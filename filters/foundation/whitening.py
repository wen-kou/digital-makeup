# -*- coding: utf-8 -*-
# July/2018 LzCai

import cv2
import numpy as np


def whiten_skin(img, s_rate = 0.22, v_rate = 0.17):
    '''
    提亮美白
    arguments:
    rate:float,-1~1,new_V=min(255,V*(1+rate))
    confirm:whether confirm this option
    '''
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # S: Saturation: the larger the more "colorful": DECREASE
    img_hsv[:, :, 1] = \
        np.minimum(img_hsv[:, :, 1] - img_hsv[:, :, 1]  * s_rate, 255).astype('uint8')

    # V: Value: the larger the lighter: INCREASE
    img_hsv[:, :, 2] = \
        np.minimum(img_hsv[:, :, 2] + img_hsv[:, :, 2]  * v_rate, 255).astype('uint8')

    # Adjust Color Temperature
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    return img_bgr


def _kelvin_to_BGR(k_temp):  # kTemp in [1000,40000]
    k_temp = min(k_temp, 40000)
    k_temp = max(k_temp, 1000)
    k_temp /= 100

    # Red:
    if k_temp <= 66:
        r = 255
    else:
        tmp = k_temp - 60
        tmp = 329.698727446 * pow(tmp, -0.1332047592)
        r = tmp
        r = min(r, 255)
        r = max(r, 0)

    # Green:
    if k_temp <= 66:
        tmp = k_temp
        tmp = 99.4708025861 * np.log(tmp) - 161.1195681661
    else:
        tmp = k_temp - 60
        tmp = 288.1221695283 * pow(tmp, -0.0755148492)
    g = tmp
    g = min(g, 255)
    g = max(g, 0)

    # Blue:
    if k_temp >= 66:
        b = 255
    elif k_temp <= 19:
        b = 0
    else:
        tmp = k_temp - 10
        tmp = 138.5177312231 * np.log(tmp) - 305.0447927307
        b = tmp
        b = min(b, 255)
        b = max(b, 0)

    return np.array((b, g, r))


def color_balance(img, k_temp):
    clr = _kelvin_to_BGR(k_temp)
    return (img * clr / 255).astype(np.uint8)
