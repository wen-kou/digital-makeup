# July/2018 LzCai

import numpy as np

ORGAN_NAME = ['jaw', 'mouth', 'nose', 'left eye', 'right eye', 'left brow', 'right brow']
ORGAN_INDEX = [list(range(0, 17)), list(range(48, 61)), list(range(27, 35)), list(range(42, 48)),
               list(range(36, 42)), list(range(22, 27)), list(range(17, 22))]
ORGAN_DETECT_LIST = [1, 3, 4, 5, 6]

JAW = 0
MOUTH = 1
NOSE = 2
LEFT_EYE = 3
RIGHT_EYE = 4
LEFT_BROW = 5
RIGHT_BROW = 6
FOREHEAD = 7

def get_checklist():
    point_checklist = [""] * 68
    for tmp in zip(ORGAN_NAME, ORGAN_INDEX):
        for i in tmp[1]:
            point_checklist[i] = tmp[0]
    return point_checklist


LEFT_CHEEK = 0
RIGHT_CHEEK = 1
FOREHEAD = 2

RIGHT_CHEEK_POINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 57, 58, 59, 60, 31, 30, 29, 28, 39, 40, 41, 36, 17]
LEFT_CHEEK_POINTS = [16, 15, 14, 13, 12, 11, 10, 9, 8, 57, 56, 55, 54, 35, 30, 29, 28, 42, 47, 46, 45, 26]


def getKsize(img, size):
    tmprate = 80
    tmpsize = max([int(np.sqrt(size / 3) / tmprate), 1])
    tmpsize = (tmpsize if tmpsize % 2 == 1 else tmpsize + 1)
    ksize = (tmpsize, tmpsize)
    return ksize