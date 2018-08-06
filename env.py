# coding:utf-8


class TankMatch:

    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features
        self.directions = [i*(360/n_actions)%360 for i in range(n_actions)]

    # 连接环境
    def initialize(self):
        pass

    # 执行命令
    def carry_out(self):
        pass

    # 刷新环境
    def render(self):
        pass

    # 断开连接，结束游戏
    def close(self):
        pass


