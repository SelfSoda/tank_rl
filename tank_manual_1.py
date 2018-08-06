# coding:utf-8
from ws4py.client.threadedclient import WebSocketClient
import time
import threading
import config
import json

direction = -1

name = 'dtt'

class HumanTank(WebSocketClient):
    def opened(self):
        global name
        self.send(config.AUTH_DICT[name])

    def send_cmd(self, cmd):
        self.send(cmd)

    def closed(self, code, reason=None):
        print("Closed down", code, reason)

    def received_message(self, m):
        result = json.loads(m.data)
        print(result['data'])

"""
    走正方形，每次转弯的时候发炮
"""
def policy_1(ws):
    global direction
    while True:
        if direction == -1:
            direction = 0
        direction = (direction+90) % 360
        ws.send_cmd(config.turn(direction))
        ws.send_cmd(config.FIRE)
        time.sleep(1.5)


if __name__ == "__main__":
    try:
        ws = HumanTank(config.URL)
        ws.connect()
        match = threading.Thread(target=ws.run_forever, name="match")
        match.start()

        time.sleep(3)
        print("Start match...")

        tank = threading.Thread(target=policy_1, args=(ws,), name="tank")
        tank.start()

    except KeyboardInterrupt:
        ws.close()

