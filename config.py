import json

# ----------------Turn Command-----------------------


def turn(d):
    cmd = {
        'commandType': 'direction',
        'angle': d
    }
    return json.dumps(cmd)

# ----------------------wss--------------------------

URL = "wss://tank-match.taobao.com/ai"


# -----------------Authentication--------------------

AUTH_LJZ = {
    'commandType': 'aiEnterRoom',
    'roomId': '170300',
    'accessKey': '1d3fd308a5eb25c6d5247e4b3f34ba8b',
    'employeeId': '167915'
}

AUTH_DTT = {
    'commandType': 'aiEnterRoom',
    'roomId': '170300',
    'accessKey': 'f1dfeff9f6fcb914c29385553779c976',
    'employeeId': '151332'
}

AUTH_LBD = {
    'commandType': 'aiEnterRoom',
    'roomId': '170300',
    'accessKey': '05bc2bebc65179498fddb61521922545',
    'employeeId': '170300'
}

AUTH_LJZ = json.dumps(AUTH_LJZ)
AUTH_DTT = json.dumps(AUTH_DTT)
AUTH_LBD = json.dumps(AUTH_LBD)

AUTH_DICT = {
    "lbd": AUTH_LBD,
    "dtt": AUTH_DTT,
    "ljz": AUTH_LJZ
}


# ---------------------command-----------------------
FIRE = {
    'commandType': 'fire'
}

STOP = {
    'commandType': 'direction',
    'angle': -1
}

LEFT = -45
RIGHT = -LEFT

FIRE = json.dumps(FIRE)
STOP = json.dumps(STOP)

# ------------------4 direction----------------------

ACTION_DICT_4 = {
    0:  turn(0),
    1:  turn(90),
    2:  turn(180),
    3:  turn(270)
}

# --------------------tank_id------------------------
TANK_ID = {
    "lbd": "ai:262",
    "dtt": "ai:291",
    "ljz": "ai:293"
}





if __name__=='__main__':
    print(LEFT)
