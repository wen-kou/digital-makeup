import numpy as np
import cv2


def imageBlending(origin, destin, percentage):
    return cv2.addWeighted(origin, 1 - percentage, destin, percentage, 0.0)
