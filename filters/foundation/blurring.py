# -*- coding: utf-8 -*-
# July/2018 LzCai

import cv2
import numpy as np
from cv2.ximgproc import guidedFilter as gf
from filters import render_util


def skin_retouch(img, img_skin, func="Gaussian", strength=50, dx=10):
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

    # dx = 10  # kernel size of the filter
    fc = 40  # delta value for the filter
    p = 50  # transparency
    blur_func = {
        "Gaussian": lambda: cv2.GaussianBlur(temp2, (21, 21), 0, 0),
        # guidedFilter(guide, src, radius, eps[, dst[, dDepth]])
        "Guided": lambda: gf(temp2, temp2, 10, 10),

        "Surface": lambda: cv2.blur(temp2, (1, 1)),
        "Bilateral": lambda: cv2.bilateralFilter(temp2, 10, 10, 10)
    }

    temp1 = cv2.bilateralFilter(img, dx, fc, fc)  # Low Freq: Blurred
    temp2 = (temp1 - img + 128).astype(np.uint8)  # High Freq: Noise and Details => To be Reduced
    temp3 = blur_func.get(func, lambda: None)()  # Blurring the High Freq
    temp4 = render_util.linear_light(temp1, temp3)

    dst = np.uint8(img * ((100 - p) / 100) + temp4 * (p / 100))
    blurred_mask = cv2.blur(img_skin * 255, (51, 51)) / 255
    blurred_mask[:, :, 1] = blurred_mask[:, :, 0]
    blurred_mask[:, :, 2] = blurred_mask[:, :, 0]
    img_skin_compliment = -(blurred_mask - 1)

    dst_blend = np.uint8(dst * blurred_mask + img * img_skin_compliment)

    blended = render_util.image_blending(img, dst_blend, strength / 100)
    return blended
