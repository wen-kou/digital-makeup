# -*- coding: utf-8 -*-

import cv2
import numpy as np
# import os
# from os import listdir
# from matplotlib import pyplot as plt
# import pandas as pd
# from os.path import isfile, join
# from skimage import io
from cv2.ximgproc import guidedFilter as gf
from utils import img_randering


def detectSkin(img):
    rows, cols, channals = img.shape
    img_skin = np.zeros((rows, cols, channals), np.uint8)
    for row in range(rows):
        for col in range(cols):
            blue = img.item(row, col, 0)
            green = img.item(row, col, 1)
            red = img.item(row, col, 2)

            criteria1 = (red > 95) and (green > 40) and (blue > 20) and (
                    max(red, green, blue) - min(red, green, blue) > 15) and abs(red - green) > 15 and (
                                red > green) and (red > blue)
            criteria2 = (red > 220) and (green > 210) and (blue > 170) and (abs(red - green) <= 15) and (
                        red > blue) and (green > blue)

            if criteria1 or criteria2:
                img_skin[row, col] = (1, 1, 1)

    return img_skin


def skinRetouch(img, imgSkin, func="Gaussian", strength=50):
    strength = min(strength, 100)
    strength = max(strength, 0)

    if strength == 0:
        return img
    '''
    操作思路：
    1. 对原图层image进行双边滤波，结果存入temp1图层中。
    2. 将temp1图层减去原图层image，将结果存入temp2图层中。
    3. 对temp2图层进行高斯滤波，结果存入temp3图层中。
    4. 以原图层image为基色，以temp3图层为混合色，将两个图层进行线性光混合得到图层temp4。
    5. 考虑不透明度，修正上一步的结果，得到最终图像dst。
    '''

    dx = 10  # kernel size of the filter
    fc = 40  # delta value for the filter
    p = 50  # transparency
    blurFunc = {
        "Gaussian": lambda: cv2.GaussianBlur(temp2, (5, 5), 0, 0),
        # guidedFilter(guide, src, radius, eps[, dst[, dDepth]])
        "Guided": lambda: gf(temp2, temp2, 10, 10),

        "Surface": lambda: cv2.blur(temp2, (1, 1)),
        "Bilateral": lambda: cv2.bilateralFilter(temp2, 10, 10, 10)
    }

    temp1 = cv2.bilateralFilter(img, dx, fc, fc)  # Low Freq: Blurred
    temp2 = (temp1 - img + 128).astype(np.uint8)  # High Freq: Noise and Details => To be Reduced
    temp3 = blurFunc.get(func, lambda: None)()  # Blurring the High Freq
    temp4 = temp1 + temp3 - 135
    # Because delta = low - origin + 128 => new = low + delta - 128 (+- epsilon)

    dst = np.uint8(img * ((100 - p) / 100) + temp4 * (p / 100))
    imgskin_c = np.uint8(-(imgSkin - 1))
    dst = np.uint8(dst * imgSkin + img * imgskin_c)

    blended = img_randering.imageBlending(img, dst, strength / 100)
    return blended

# img, r, g, b = readImg('test_face3.jpg')
# filterName = "Gaussian"
# result = dermabrasion(img, detect_skin(img), filterName, value1=3, value2=2)
# showNclose(result, filterName)
