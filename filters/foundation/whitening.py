# -*- coding: utf-8 -*-

import cv2
import numpy as np
from utils import img_randering


def whitenSkin(img, imgSkin, s_rate = 0.22, v_rate = 0.17):
    '''
    提亮美白
    arguments:
    rate:float,-1~1,new_V=min(255,V*(1+rate))
    confirm:whether confirm this option
    '''

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # S: Saturation: the larger the more "colorful": DECREASE
    img_hsv[:, :, 1] = \
        np.minimum(img_hsv[:, :, 1] - img_hsv[:, :, 1] * imgSkin[:, :, 1] * s_rate, 255).astype('uint8')

    # V: Value: the larger the lighter: INCREASE
    img_hsv[:, :, 2] = \
        np.minimum(img_hsv[:, :, 2] + img_hsv[:, :, 2] * imgSkin[:, :, 2] * v_rate, 255).astype('uint8')

    img_hsv = img_randering.trim256(img_hsv)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def kelvin2BGR(kTemp):  # kTemp in [1000,40000]
    kTemp = min(kTemp, 40000)
    kTemp = max(kTemp, 1000)
    kTemp /= 100

    # Red:
    if kTemp <= 66:
        r = 255
    else:
        tmp = kTemp - 60
        tmp = 329.698727446 * pow(tmp, -0.1332047592)
        r = tmp
        r = min(r, 255)
        r = max(r, 0)

    # Green:
    if kTemp <= 66:
        tmp = kTemp
        tmp = 99.4708025861 * np.log(tmp) - 161.1195681661
    else:
        tmp = kTemp - 60
        tmp = 288.1221695283 * pow(tmp, -0.0755148492)
    g = tmp
    g = min(g, 255)
    g = max(g, 0)

    # Blue:
    if kTemp >= 66:
        b = 255
    elif kTemp <= 19:
        b = 0
    else:
        tmp = kTemp - 10
        tmp = 138.5177312231 * np.log(tmp) - 305.0447927307
        b = tmp
        b = min(b, 255)
        b = max(b, 0)

    return np.array((b, g, r))


def colorBalance(img, kTemp):
    clr = kelvin2BGR(kTemp)
    return (img_randering.trim256(img * clr / 255)).astype(np.uint8)
