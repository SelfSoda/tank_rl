# coding:utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
from env import TankMatch

# np.random.seed(1)
# tf.set_random_seed(1)

ALPHA = 0.1  # 学习率 learning rate
GAMMA = 0.9  # 衰减率 discount rate/reward decay
EPSILON = 0.9  # 策略90%靠Q值，10%靠随机


class DeepQNetworkLSTM:

    def __init__(
            self,
            n_actions,
            # n_features,

            tank_input,
            bullet_input,
            tank_max_count,
            bullet_max_count,
            hidden_size,
            batch_size,

            learning_rate=ALPHA,
            reward_decay=GAMMA,
            e_greedy=EPSILON,
            replace_target_iter=10,
            saver_iter=100,
            memory_size=200,
            e_greedy_increment=None,
            output_graph=False
    ):
        self.n_actions = n_actions
        # self.n_features = n_features

        self.tank_input = tank_input
        self.bullet_input = bullet_input
        self.time_steps_tank = tank_max_count
        self.time_steps_bullet = bullet_max_count
        self.hidden_size = hidden_size

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.saver_iter = saver_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learning_step_counter = 0

        # self.memory = np.zeros((self.memory_size, n_features*2+2))

        self.memory_tank = np.zeros((self.memory_size, self.tank_input*self.time_steps_tank))
        self.memory_bullet = np.zeros((self.memory_size, self.bullet_input*self.time_steps_bullet))
        self.memory_action = np.zeros((self.memory_size, 1))
        self.memory_reward = np.zeros((self.memory_size, 1))
        self.memory_tank_ = np.zeros((self.memory_size, self.tank_input*self.time_steps_tank))
        self.memory_bullet_ = np.zeros((self.memory_size, self.bullet_input*self.time_steps_bullet))

        self._build_net()

        self.sess = tf.Session()

        self.saver = tf.train.Saver()

        # self.saver.restore(self.sess, './sess/dqn-lstm-100')

        t_params = tf.get_collection('eval_net_params')
        e_params = tf.get_collection('target_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if output_graph:
            tf.summary.FileWriter("./logs/dqn_lstm", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # build evaluate net
        self.s_tank = tf.placeholder(tf.float32, [None, self.tank_input], name='s_tank')
        self.s_bullet = tf.placeholder(tf.float32, [None,  self.bullet_input], name="s_bullet")
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            w_initializer = tf.random_normal_initializer(0, 0.3)
            b_initializer = tf.constant_initializer(0.1)

            # --------------------input layer--------------------
            with tf.variable_scope("tank_l1"):
                tank_input = tf.reshape(self.s_tank, [-1, self.tank_input])
                print("tank_input.shape is {}".format(tank_input.shape))
                tank_w_in = tf.get_variable('tank_w_in', [self.tank_input, self.hidden_size],
                                            initializer=w_initializer, collections=c_names)
                tank_b_in = tf.get_variable('tank_b_in', [1, self.hidden_size],
                                            initializer=b_initializer, collections=c_names)
                tank_in = tf.matmul(tank_input, tank_w_in) + tank_b_in
                tank_in = tf.reshape(tank_in, [-1, self.time_steps_tank, self.hidden_size])

            with tf.variable_scope("bullet_l1"):
                bullet_input = tf.reshape(self.s_bullet, [-1, self.bullet_input])
                print("bullet_input.shape is {}".format(bullet_input.shape))
                bullet_w_in = tf.get_variable('bullet_w_in', [self.bullet_input, self.hidden_size],
                                              initializer=w_initializer, collections=c_names)
                bullet_b_in = tf.get_variable('bullet_b_in', [1, self.hidden_size],
                                              initializer=b_initializer, collections=c_names)
                bullet_in = tf.matmul(bullet_input, bullet_w_in) + bullet_b_in
                bullet_in = tf.reshape(bullet_in, [-1, self.time_steps_bullet, self.hidden_size])

            # ---------------------lstm layer--------------------

            with tf.variable_scope("tank_lstm"):
                print("tank_in.shape is {}".format(tank_in.shape))
                tank_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                tank_init_state = tank_lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
                tank_output, _ = tf.nn.dynamic_rnn(tank_lstm_cell, tank_in, initial_state=tank_init_state)
                print("tank_output.shape is {}".format(tank_output.shape))
                tank_output = tf.reshape(tank_output, [-1, self.hidden_size])
                print("tank_output.shape is {}".format(tank_output.shape))

            with tf.variable_scope("bullet_lstm"):
                print("bullet_in.shape is {}".format(bullet_in.shape))
                bullet_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                bullet_init_state = bullet_lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
                bullet_output, _ = tf.nn.dynamic_rnn(bullet_lstm_cell, bullet_in, initial_state=bullet_init_state)
                print("bullet_output.shape is {}".format(bullet_output.shape))
                bullet_output = tf.reshape(bullet_output, [-1, self.hidden_size])
                print("bullet_output.shape is {}".format(bullet_output.shape))

            # -------------------output layer--------------------

            with tf.variable_scope("output"):
                output_w = tf.get_variable('output_w', [self.hidden_size*2, self.n_actions], initializer=w_initializer)
                output_b = tf.get_variable('output_b', [1, self.n_actions], initializer=b_initializer)
                print("tank_output[-1].shape is {}".format(tank_output[-1].shape))
                print("bullet_output[-1].shape is {}".format(bullet_output[-1].shape))
                output_concat = tf.concat([tank_output[-1], bullet_output[-1]], 0)
                output_concat = tf.reshape(output_concat, [-1, self.hidden_size*2])
                print("output_concat.shape is {}".format(output_concat.shape))
                print("output_w.shape is {}".format(output_w.shape))
                xxx = tf.matmul(output_concat, output_w)
                print("xxx.shape is {}".format(xxx.shape))
                self.q_eval = tf.matmul(output_concat, output_w) + output_b
                print("q_eval.shape is {}".format(self.q_eval.shape))

        # ----------------------loss-------------------------

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.q_eval, labels=self.q_target))

        # ----------------------train------------------------

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # build target net
        self.s_tank_ = tf.placeholder(tf.float32, [None, self.tank_input], name='s_tank')
        self.s_bullet_ = tf.placeholder(tf.float32, [None, self.bullet_input], name="s_bullet")

        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            w_initializer = tf.random_normal_initializer(0, 0.3)
            b_initializer = tf.constant_initializer(0.1)

            # --------------------input layer--------------------
            with tf.variable_scope("tank_l1"):
                tank_input = tf.reshape(self.s_tank_, [-1, self.tank_input])
                tank_w_in = tf.get_variable('tank_w_in', [self.tank_input, self.hidden_size],
                                            initializer=w_initializer, collections=c_names)
                tank_b_in = tf.get_variable('tank_b_in', [1, self.hidden_size],
                                            initializer=b_initializer, collections=c_names)
                tank_in = tf.matmul(tank_input, tank_w_in) + tank_b_in
                tank_in = tf.reshape(tank_in, [-1, self.time_steps_tank, self.hidden_size])

            with tf.variable_scope("bullet_l1"):
                bullet_input = tf.reshape(self.s_bullet_, [-1, self.bullet_input])
                bullet_w_in = tf.get_variable('bullet_w_in', [self.bullet_input, self.hidden_size],
                                              initializer=w_initializer, collections=c_names)
                bullet_b_in = tf.get_variable('bullet_b_in', [1, self.hidden_size],
                                              initializer=b_initializer, collections=c_names)
                bullet_in = tf.matmul(bullet_input, bullet_w_in) + bullet_b_in
                bullet_in = tf.reshape(bullet_in, [-1, self.time_steps_bullet, self.hidden_size])

            # ---------------------lstm layer--------------------

            with tf.variable_scope("tank_lstm"):
                tank_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                tank_init_state = tank_lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
                tank_output, _ = tf.nn.dynamic_rnn(tank_lstm_cell, tank_in, initial_state=tank_init_state)

            with tf.variable_scope("bullet_lstm"):
                bullet_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                bullet_init_state = bullet_lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
                bullet_output, _ = tf.nn.dynamic_rnn(bullet_lstm_cell, bullet_in, initial_state=bullet_init_state)

            # -------------------output layer--------------------

            with tf.variable_scope("output"):
                output_w = tf.get_variable('output_w', [self.hidden_size*2, self.n_actions], initializer=w_initializer)
                output_b = tf.get_variable('output_b', [1, self.n_actions], initializer=b_initializer)
                output_concat = tf.concat([tank_output, bullet_output], 0)
                output_concat = tf.reshape(output_concat, [-1, self.hidden_size*2])
                self.q_next = tf.matmul(output_concat, output_w) + output_b

    def store_transition(self, tank_state, bullet_state, action, reward, tank_state_, bullet_state_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.memory_size
        # self.memory[index:] = transition

        self.memory_tank[index:] = tank_state
        self.memory_bullet[index:] = bullet_state
        self.memory_action[index:] = action
        self.memory_reward[index:] = reward
        self.memory_tank_[index:] = tank_state_
        self.memory_bullet_[index:] = bullet_state_

        self.memory_counter += 1

    """
        tank_observation.shape is [MAX_TANK, TANK_FEATURES]
        bullet_observation.shape is [MAX_BULLET, BULLET_FEATURES]
    """
    def choose_action(self, tank_observation, bullet_observation):
        # tank_observation = tank_observation[np.newaxis, :]
        # bullet_observation = bullet_observation[np.newaxis, :]

        print("tank_observation.shape is {}".format(tank_observation.shape))
        print("bullet_observation.shape is {}".format(bullet_observation.shape))

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval,
                                          feed_dict={
                                              self.s_tank: tank_observation,
                                              self.s_bullet: bullet_observation,
                                          })
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        if self.learning_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print("\ntarget_params_replaced")
        if self.learning_step_counter % self.saver_iter == 0:
            print("\nsave_sess")
            self.saver.save(self.sess, './sess/dqn-lstm', global_step=self.learning_step_counter)

        sample_index = np.random.choice(self.memory_size, size=self.batch_size) \
            if self.memory_counter > self.memory_size \
            else np.random.choice(self.memory_counter, size=self.batch_size)
        # batch_memory = self.memory[sample_index, :]
        batch_memory_tank = self.memory_tank[sample_index, :]
        batch_memory_bullet = self.memory_bullet[sample_index, :]
        batch_memory_tank_ = self.memory_tank_[sample_index, :]
        batch_memory_bullet_ = self.memory_bullet_[sample_index, :]
        batch_memory_action = self.memory_action[sample_index, :]
        batch_memory_reward = self.memory_reward[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                # self.s_: batch_memory[:, -self.n_features:],
                # self.s: batch_memory[:, :self.n_features]
                self.s_tank: batch_memory_tank,
                self.s_bullet: batch_memory_bullet,
                self.s_tank_: batch_memory_tank_,
                self.s_bullet_: batch_memory_bullet_
            }
        )

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # eval_act_index = batch_memory[:, self.n_features].astype(int)
        # reward = batch_memory[:, self.n_features+1]
        eval_act_index = batch_memory_action.astype(int)
        reward = batch_memory_reward

        q_target[batch_index, eval_act_index] = reward + self.gamma*np.max(q_next, axis=1)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={
                                         # self.s: batch_memory[:, :self.n_features],
                                         # self.q_target: q_target
                                         self.s_tank: batch_memory_tank,
                                         self.s_bullet: batch_memory_bullet,
                                         self.q_target: q_target
                                     })

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learning_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


if __name__=='__main__':

    rl = DeepQNetworkLSTM(
        n_actions=5,
        # n_features=10,

        tank_input=6,
        bullet_input=4,
        tank_max_count=10,
        bullet_max_count=20,
        hidden_size=64,
        batch_size=32,

        output_graph=True
    )