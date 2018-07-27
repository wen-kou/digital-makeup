# July/2018 LzCai

import numpy as np
import cv2


def image_blending(origin, destin, percentage):
    return cv2.addWeighted(origin, 1 - percentage, destin, percentage, 0.0)


def linear_light(img_1, img_2):
    img_1.astype(float)
    img_2.astype(float)
    img_1 = img_1 / 255
    img_2 = img_2 / 255

    img = np.array(img_2 + img_1 * 2 - 1)
    mask_1 = img < 0
    mask_2 = img > 1
    img = img * (1 - mask_1)
    img = img * (1 - mask_2) + mask_2

    img *= 255
    return img.astype(np.uint8)


def hard_light(img_1, img_2):
    img_1.astype(float)
    img_2.astype(float)
    img_1 = img_1 / 255
    img_2 = img_2 / 255

    mask = img_1 < 0.5
    T1 = 2 * img_1 * img_2
    T2 = 1 - 2 * (1 - img_1) * (1 - img_2)
    img = T1 * mask + T2 * (1 - mask)

    return (img * 255).astype(np.uint8)


def tri_color(gray):
    k1 = 80
    k2 = 128
    k3 = 200
    result = gray.copy()
    result[gray < k1] = 0
    result[(gray < k3) * (gray >= k1)] = k2
    result[gray >= k3] = 255
    return result