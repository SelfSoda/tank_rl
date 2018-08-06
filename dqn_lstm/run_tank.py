# coding:utf-8
import time
import threading
import json
import random

import numpy as np
from ws4py.client.threadedclient import WebSocketClient

import config
import util
from dqn_lstm.rl_brain_lstm import DeepQNetworkLSTM

last_action = None

last_data = None

last_observation = None

step = 0

class RLTank(WebSocketClient):

    def opened(self):
        global name
        self.send(config.AUTH_DICT[name])
        self.my_tank_name = config.TANK_ID[name]
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

        if not isinstance(last_observation, dict):
            # observation = np.zeros((n_features))
            observation = {
                'tank': np.zeros((MAX_TANK, TANK_FEATURES)),
                'bullet': np.zeros((MAX_BULLET, BULLET_FEATURES))
            }
        else:
            observation = last_observation

        # ------------get result---------------------

        result = json.loads(m.data)
        data = json.loads(result['data'])
        observation_received = self.get_observation(data)

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
        return action

    def get_observation(self, data):
        # observation = []
        tank_observation = []
        bullet_observation = []
        tanks = data['tanks']
        bullets = data['bullets']
        global name

        # --------------------------me---------------------------------

        my_tank = tanks[self.my_tank_name]

        self.me = (my_tank['position'][0], my_tank['position'][1], my_tank['direction'],
                   my_tank['score'], my_tank['fireCd'], my_tank['shieldCd'],
                   my_tank['rebornCd'] if my_tank['rebornCd'] else 0)

        # ------------------------enemy--------------------------------
        self.enemy_list = []
        for (tank_name, value) in tanks.items():
            if tank_name == self.my_tank_name:
                continue
            # !!! tank feature !!!
            enemy = (value['position'][0], value['position'][1], value['direction'],
                     value['score'], value['fireCd'], value['shieldCd'], value['rebornCd'] if value['rebornCd'] else 0)
            self.enemy_list.append(enemy)

        # sorted by score from minimum to maximum
        self.enemy_list = sorted(self.enemy_list, cmp=lambda x:x[3], reverse=False)

        # for enemy in self.enemy_list[:max_enemy]:
        #     observation += [e for e in enemy[:3]]

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
            # if distance < 0.15:
            #     reward = -5
            pass
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

    name = 'lbd'            # 比赛ID
    # name = 'dtt'
    # name = 'ljz'

    MAX_TANK = 10
    MAX_BULLET = 10
    TANK_FEATURES = 7
    BULLET_FEATURES = 4

    dqn = DeepQNetworkLSTM(
        n_actions=5,        # 动作的数量
        # n_features=10,

        tank_input=TANK_FEATURES,           # 坦克的feature数
        bullet_input=BULLET_FEATURES,       # 子弹的feature数
        tank_max_count=MAX_TANK,            # 同一时间，场上存在的最大坦克数
        bullet_max_count=MAX_BULLET,        # 同一时间，场上存在的最大子弹数
        hidden_size=64,     # LSTM隐状态
    )
    try:
        ws = RLTank(config.URL)
        ws.connect()
        match = threading.Thread(target=ws.run_forever, name="match")
        match.start()

        time.sleep(3)
        print("Start match...")

    except KeyboardInterrupt:
        ws.close()

