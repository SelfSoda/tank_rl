# coding:utf-8
import json
import os

PATH = "./log/warm_up_1"
with open(PATH, 'r') as f:
    index = 0
    last_score = 0

    line = f.readline()
    while line:
        data = json.loads(line)

        tanks = data['tanks']
        bullets = data['bullets']

        for (name, value) in tanks.items():
            if name == "ai:291":
                score = value['score']

                if score > last_score:
                    print("==============================")
                    print(last_tanks['ai:291'])
                    print(last_bullets)
                    print("------------------------------")
                    print(tanks)
                    print(bullets)
                    print("==============================")

                last_score = score
                last_tanks = tanks
                last_bullets = bullets
                break


        line = f.readline()
        index += 1

