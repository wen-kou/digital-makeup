# -*- coding: utf-8 -*-

import cv2
import numpy as np


def whitenSkin(img, imgSkin, strength=10):
    '''
    提亮美白
    arguments:
    rate:float,-1~1,new_V=min(255,V*(1+rate))
    confirm:wether confirm this option
    '''

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rate = strength / 100
    # S: Saturation: the larger the more "colorful": DECREASE
    img_hsv[:, :, 1] = np.minimum(img_hsv[:, :, 1] - img_hsv[:, :, 1] * imgSkin[:, :, 1] * rate * 4, 255).astype(
        'uint8')
    # V: Value: the larger the lighter: INCREASE
    img_hsv[:, :, 2] = np.minimum(img_hsv[:, :, 2] + img_hsv[:, :, 2] * imgSkin[:, :, 2] * rate, 255).astype('uint8')
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def colorBalance(img):
    blue_ch, green_ch, red_ch = cv2.split(img)
    max_gray = 255
    min_gray = 0
    radius = 0.1
    rows, cols, channels = np.shape(img)
    n_pixels = 1
    V_min = n_pixels * radius / 2
    V_max = n_pixels * (1 - radius / 2) - 1
    return ((img - V_min) * (max_gray - min_gray) / (V_max - V_min) + min_gray).astype(np.uint8)
