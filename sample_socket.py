# coding:utf-8
import asyncio
import websockets
import websocket
import json
try:
    import thread
except ImportError:
    import _thread as thread
import time


def on_message(ws, message):
    print(message)


def on_error(ws, error):
    print(error)


def on_close(ws):
    time.sleep(3)
    print("### closed ###")


def on_cont_message(ws, message):
    print(message)


def on_data(ws, data):
    print(data)


def on_open(ws):
    def run(*args):
        auth_dict = {
            'commandType': 'aiEnterRoom',
            'roomId': '170300',
            'accessKey': '05bc2bebc65179498fddb61521922545',
            'employeeId': '170300'
        }
        auth = json.dumps(auth_dict)
        # print(auth)
        ws.send(auth)
        # ws.close()
        print("thread terminating...")
    thread.start_new_thread(run, ())


if __name__ == "__main__":
    URL = "wss://tank-match.taobao.com/ai"
    AUTH_DICT = {
        'commandType': 'aiEnterRoom',
        'roomId': '170300',
        'accessKey': '05bc2bebc65179498fddb61521922545',
        'employeeId': '170300'
    }


    websocket.enableTrace(True)
    # ws = websocket.WebSocketApp(URL,
    #                             on_message=on_message,
    #                             on_error=on_error,
    #                             on_data=on_data,
    #                             on_cont_message=on_cont_message,
    #                             on_close=on_close)
    # ws.on_open = on_open
    # ws.run_forever()
    # auth = json.dumps(auth_dict)
    # # print(auth)
    # # ws.send(auth)
    # ws.close()


    # ws = websocket.WebSocketApp(URL)
    # ws.send(json.dumps(AUTH_DICT))
    # ws.run_forever()

    ws = websocket.WebSocket(sslopt={"cert_reqs": ssl.CERT_NONE})
    ws.connect("wss://echo.websocket.org")