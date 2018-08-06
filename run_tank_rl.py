# coding:utf-8
import time
import threading
import json
import random

import numpy as np
from ws4py.client.threadedclient import WebSocketClient

import config
import util
from rl_brain import DeepQNetwork

name = 'lbd'

# cmd_stack = []

last_action = None

last_data = None

last_observation = None

step = 0

class RLTank(WebSocketClient):

    def opened(self):
        global name
        self.send(config.AUTH_DICT[name])
        print("Connection succeed. \n{}".format(config.AUTH_DICT[name]))

    def send_cmd(self, action):
        cmd_list = self.action2cmd(action)
        for c in cmd_list:
            self.send(c)

    def closed(self, code, reason=None):
        print("Closed down", code, reason)

    def received_message(self, m):
        if not m:
            return
        global last_action
        global last_data
        global last_observation
        global step

        # ------------get observation----------------

        if type(last_observation) != np.array:
            observation = np.zeros((n_features))
        else:
            observation = last_observation

        # ------------get result---------------------

        result = json.loads(m.data)
        data = json.loads(result['data'])
        observation_result = self.get_observation(data)

        # ---------------get reward------------------

        reward = self.get_reward(data)
        # ------------choose next action-------------

        action = self.policy(observation_result)

        self.send_cmd(action)

        # global cmd_stack
        # if cmd_stack:
        #     cmd = cmd_stack[0]
        #     self.send_cmd(cmd)
        #     print(cmd)

        # -----------------RL store------------------

        if last_action:
            dqn.store_transition(observation, last_action, reward, observation_result)

        # -----------------RL learn------------------

        if (step > 50) and (step % 10 == 0):
            print("start to learn ... (step={})".format(step))
            time_st = time.time()
            dqn.learn()
            time_end = time.time()
            print("cost {}".format(time_end-time_st))

        # -----------update last data----------------

        last_action = action
        last_data = data
        last_observation = observation_result

        step += 1

    """
        8个action，
        分别是不开火且转向到0， 90， 180， 270，
        以及先转向到0， 90， 180， 270，然后再开火
    """
    def policy(self, observation):
        action = dqn.choose_action(observation)
        # cmd_stack.append(cmd)
        return action

    def get_observation(self, data):
        observation = []
        tanks = data['tanks']
        bullets = data['bullets']
        global name

        # --------------------------me---------------------------------
        my_tank_name = config.TANK_ID[name]
        my_tank = tanks[my_tank_name]

        observation.append(my_tank['position'][0])
        observation.append(my_tank['position'][1])
        observation.append(my_tank['direction'])
        observation.append(1 if my_tank['fire'] else 0)

        self.me = (my_tank['position'][0], my_tank['position'][1], my_tank['direction'])

        # ------------------------enemy--------------------------------
        self.enemy_list = []
        for (tank_name, value) in tanks.items():
            if tank_name == my_tank_name:
                continue
            enemy = (value['position'][0], value['position'][1], value['direction'], value['score'])
            self.enemy_list.append(enemy)

        # self.enemy_list = sorted(self.enemy_list, cmp=lambda x:x[3], reverse=True)

        for enemy in self.enemy_list[:max_enemy]:
            observation += [e for e in enemy[:3]]

        while len(observation) < 3 + max_enemy*3:
            observation.append(0)

        # ------------------------bullet-------------------------------
        self.bullet_list = []
        for bullet in bullets:
            self.bullet_list.append((bullet['position'][0], bullet['position'][1], bullet['direction']))

        for bullet in self.bullet_list[:max_bullet]:
            observation += [b for b in bullet]

        while len(observation)<n_features:
            observation.append(0)

        return np.array(observation)

    def get_reward(self, data):
        global name
        global last_data
        score = data['tanks'][config.TANK_ID[name]]['score']
        last_score = last_data['tanks'][config.TANK_ID[name]]['score'] if last_data else score

        last_pos = (last_data['tanks'][config.TANK_ID[name]]['position'][0],
                    last_data['tanks'][config.TANK_ID[name]]['position'][1],
                    last_data['tanks'][config.TANK_ID[name]]['direction']) if last_data else (0, 0, 0)

        distance = util.get_distance(self.me, last_pos)

        reward = score-last_score
        if reward == 0:
            if distance < 0.15:
                reward = -5
        elif reward < 0:
            reward = -10

        print("reward is {}.".format(reward))
        return reward

    def action2cmd(self, cmd):
        if cmd < 4:
            cmd = cmd
            return [config.ACTION_DICT_4[cmd]]
        else:
            enemy = random.sample(self.enemy_list, 1)[0]
            angle = util.get_angle(self.me, enemy)

            return [config.turn(angle), config.FIRE]


if __name__ == "__main__":

    max_enemy = max_bullet = 2
    n_actions = 5
    n_features = 3 + 1 + max_enemy*3 + max_bullet*3
    # me(pos, dir) + is_fire + enemy(pos,dir)*3 + bullet(pos,dir)*3

    dqn = DeepQNetwork(n_actions=n_actions, n_features=n_features)
    try:
        ws = RLTank(config.URL)
        ws.connect()
        match = threading.Thread(target=ws.run_forever, name="match")
        match.start()

        time.sleep(3)
        print("Start match...")

    except KeyboardInterrupt:
        ws.close()

