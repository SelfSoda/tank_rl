# coding:utf-8
import time
import json
from ws4py.client.threadedclient import WebSocketClient
import threading
import keyboard


URL = "wss://tank-match.taobao.com/ai"
AUTH_DICT = {
    'commandType': 'aiEnterRoom',
    'roomId': '170300',
    'accessKey': '05bc2bebc65179498fddb61521922545',
    'employeeId': '170300'
}

FIRE = {
    'commandType': 'fire'
}

LEFT = {
    'commandType': 'direction',
    'angle': 45
}

RIGHT = {
    'commandType': 'direction',
    'angle': 45
}

direction = -1

class DummyClient(WebSocketClient):
    def opened(self):
        self.send(json.dumps(AUTH_DICT))

    def send_cmd(self, cmd):
        self.send(json.dumps(cmd))

    def closed(self, code, reason=None):
        print("Closed down", code, reason)

    def received_message(self, m):
        print(m)


def fire_forever(ws):
    while True:
        ws.send_cmd()
        print("\nfire\n")
        time.sleep(2.1)

def run_forever(ws):
    ws.run_forever




if __name__ == '__main__':
    try:
        ws = DummyClient(URL)
        ws.connect()
        t = threading.Thread(target=run_forever, args=(ws,))
        t.start()
        time.sleep(3)
        # t2 = threading.Thread(target=fire_forever, args=(ws,))
        # t2.start()


    except KeyboardInterrupt:
        ws.close()

