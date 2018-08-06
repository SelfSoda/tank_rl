# coding:utf-8
import keyboard
import time


def monitor():
    while True:
        if keyboard.is_pressed('left'):
            print('left')
        if keyboard.is_pressed('right'):
            print('right')
        if keyboard.is_pressed('space'):
            print('space')
        time.sleep(0.15)


def print_event(event):
    print(event.name)

# while True:
    # keyboard.on_press(callback=print_event)
    # time.sleep(0.5)
    # print("--+--")
    # keyboard.on_release(callback=print_event)
    # print("-----")
    # time.sleep(0.5)


monitor()
