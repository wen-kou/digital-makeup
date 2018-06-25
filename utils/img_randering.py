import numpy as np
import cv2
import warnings


def trim256(array):
    if len(np.shape(array)) == 1:
        for tmp in range(len(array)):
            array[tmp] = min(array[tmp], 255)
            array[tmp] = max(array[tmp], 0)
        return array
    elif len(np.shape(array)) == 2:
        for r in range(len(array)):
            for c in range(len(array[r])):
                array[r, c] = min(array[r, c], 255)
                array[r, c] = max(array[r, c], 0)
    elif len(np.shape(array)) == 3:
        for r in range(len(array)):
            for c in range(len(array[r])):
                for clr in range(len(array[r, c])):
                    array[r, c, clr] = min(array[r, c, clr], 255)
                    array[r, c, clr] = max(array[r, c, clr], 0)
    else:
        warnings.warn("trim256: dim>3")
        return array
    return array


def imageBlending(origin, destin, percentage):
    return cv2.addWeighted(origin, 1 - percentage, destin, percentage, 0.0)
