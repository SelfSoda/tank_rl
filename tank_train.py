# coding:utf-8
from ws4py.client.threadedclient import WebSocketClient
import keyboard
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

def keyboard_monitor(ws):
    global direction
    while True:
        if keyboard.is_pressed('left'):
            if direction == -1:
                direction = 0
            direction = (direction + config.LEFT) % 360
            ws.send_cmd(config.turn(direction))
            print('left')
            continue

        if keyboard.is_pressed('right'):
            if direction == -1:
                direction = 0
            direction = (direction + config.RIGHT) % 360
            ws.send_cmd(config.turn(direction))
            print('right')
            continue

        if keyboard.is_pressed('space'):
            ws.send_cmd(config.FIRE)
            print('space')
            continue

        if keyboard.is_pressed('down'):
            ws.send_cmd(config.STOP)
            print('down')
            continue

        time.sleep(0.2)

if __name__=="__main__":
    try:
        ws = HumanTank(config.URL)
        ws.connect()
        match = threading.Thread(target=ws.run_forever, name="match")
        match.start()

        time.sleep(3)
        print("Start match...")

        # tank = threading.Thread(target=keyboard_monitor, args=(ws,), name="tank")
        # tank.start()



    except KeyboardInterrupt:
        ws.close()

