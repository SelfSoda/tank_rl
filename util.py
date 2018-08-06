# coding:utf-8
from math import atan
from math import fabs
from math import pow
from math import pi
from math import sqrt


def get_angle(t1, t2):
    x1 = t1[0]
    y1 = t1[1]
    x2 = t2[0]
    y2 = t2[1]
    if (fabs(x1-x2) != 0):
        atan_angle = atan(fabs(y2 - y1) / fabs(x2 - x1)) / pi * 180
    else:
        atan_angle = atan(fabs(y2 - y1) / 0.001) / pi * 180
    if x1 <= x2 and y1 <= y2:
        return atan_angle
    if x1 <= x2 and y1 > y2:
        return 360 - atan_angle
    if x1 > x2 and y1 <= y2:
        return 180 - atan_angle
    if x1 > x2 and y1 > y2:
        return 180 + atan_angle


def get_distance(t1, t2):
    x1 = t1[0]
    y1 = t1[1]
    x2 = t2[0]
    y2 = t2[1]
    return sqrt(pow(x1-x2, 2) + pow(y1-y2, 2))

def is_shootable(t1, t2):
    return 1


if __name__ == "__main__":
    t1 = [13.491996547370965,1.9072850518774838]
    t2 = [13.790854956798492,1.933431774701781]
    print(get_distance(t1, t2))