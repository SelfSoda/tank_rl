# coding:utf-8
from ws4py.client.threadedclient import WebSocketClient
import time
import threading
import config
import json

direction = -1

name = 'ljz'

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


if __name__=="__main__":
    try:
        ws = HumanTank(config.URL)
        ws.connect()
        match = threading.Thread(target=ws.run_forever, name="match")
        match.start()

        time.sleep(3)
        print("Start match...")

    except KeyboardInterrupt:
        ws.close()

