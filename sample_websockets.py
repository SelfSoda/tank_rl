# coding:utf-8
import json
import asyncio
import pathlib
import ssl
import websockets


URL = "wss://tank-match.taobao.com/ai"
AUTH_DICT = {
    'commandType': 'aiEnterRoom',
    'roomId': '170300',
    'accessKey': '05bc2bebc65179498fddb61521922545',
    'employeeId': '170300'
}

# ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
# ssl_context.load_verify_locations(
#     pathlib.Path(__file__).with_name('localhost.pem'))

async def hello():
    async with websockets.connect(
            URL) as websocket:
        # name = input("What's your name? ")

        await websocket.send(json.dumps(AUTH_DICT))
        # print(f"> {name}")

        greeting = await websocket.recv()
        print(f"< {greeting}")

asyncio.get_event_loop().run_until_complete(hello())
