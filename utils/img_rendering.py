import numpy as np
import cv2


def imageBlending(origin, destin, percentage):
    return cv2.addWeighted(origin, 1 - percentage, destin, percentage, 0.0)


def Hard_light(img_1, img_2):
    img_1.astype(float)
    img_2.astype(float)
    img_1 = img_1 / 255
    img_2 = img_2 / 255

    mask = img_1 < 0.5
    T1 = 2 * img_1 * img_2
    T2 = 1 - 2 * (1 - img_1) * (1 - img_2)
    img = T1 * mask + T2 * (1 - mask)

    return (img * 255).astype(np.uint8)


def kmeansColor(img, n):
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = n
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2
