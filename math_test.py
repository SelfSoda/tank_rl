# coding:utf-8
import math

x1, y1 = 1, 1
x2, y2 = 3, 2

r = math.atan(math.fabs(y2-y1)/math.fabs(x2-x1))
r = r/math.pi*180
print(r)

def get_angle(x1, y1, x2, y2):
    atan_angle = math.atan(math.fabs(y2-y1)/math.fabs(x2-x1))/math.pi*180
    if x1 < x2 and y1 < y2:
        return atan_angle
    if x1 < x2 and y1 > y2:
        return 360-atan_angle
    if x1 > x2 and y1 < y2:
        return 180-atan_angle
    if x1 > x2 and y1 > y2:
        return 180+atan_angle
    