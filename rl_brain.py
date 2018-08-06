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


class DeepQNetwork:

    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=ALPHA,
            reward_decay=GAMMA,
            e_greedy=EPSILON,
            replace_target_iter=10,
            saver_iter=100,
            memory_size=200,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
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

        self.memory = np.zeros((self.memory_size, n_features*2+2))

        self._build_net()

        self.sess = tf.Session()

        self.saver = tf.train.Saver()

        self.saver.restore(self.sess, './sess/dqn-100')

        t_params = tf.get_collection('eval_net_params')
        e_params = tf.get_collection('target_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if output_graph:
            tf.summary.FileWriter("./logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # build evaluate net
        # 两层全链接
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s') # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target') # for calculating loss
        print(self.s.shape)
        print(self.q_target.shape)

        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            # To Do: 神经元数目
            n_l1 = 10
            w_initializer = tf.random_normal_initializer(0, 0.3)
            b_initializer = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                # To Do: 激活函数
                l1 = tf.nn.relu(tf.matmul(self.s, w1)+b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2)+b2

        with tf.variable_scope('loss'):
            # To Do: 损失函数
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            # To Do: 优化器
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # build target net
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1)+b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2)+b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index:] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
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
            self.saver.save(self.sess, './sess/dqn', global_step=self.learning_step_counter)

        sample_index = np.random.choice(self.memory_size, size=self.batch_size) \
            if self.memory_counter > self.memory_size \
            else np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],
                self.s: batch_memory[:, :self.n_features]
            }
        )

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features+1]

        q_target[batch_index, eval_act_index] = reward + self.gamma*np.max(q_next, axis=1)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={
                                         self.s: batch_memory[:, :self.n_features],
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
    env = TankMatch(8, 100)
    rl = DeepQNetwork(
        n_actions=env.n_actions,
        n_features=env.n_features,
    )
    # env.after(100, run_maze())
    # env.mainloop()
    # rl.plot_cost()