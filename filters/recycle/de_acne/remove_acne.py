import cv2
import numpy as np
from filters.facemask import facemaskMgr
from filters import render_util as render


def tri_color(gray):
    k1 = 80
    k2 = 128
    k3 = 200
    result = gray.copy()
    result[gray < k1] = 0
    result[(gray < k3) * (gray >= k1)] = k2
    result[gray >= k3] = 255
    return result


def get_acne_mask_on_patch(img, in_mask):
    img = img*in_mask
    g = cv2.split(img)[1]
    hard_g = render.hard_light(g, g)
    hard_g2 = render.hard_light(hard_g, hard_g)
    hard_g3 = render.hard_light(hard_g2, hard_g2)
    tri_hg3 = tri_color(hard_g3)

    params = cv2.SimpleBlobDetector_Params()
    params.minArea = 20
    params.maxArea = 1000
    params.minConvexity = 0.3
    params.minDistBetweenBlobs = 0
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(tri_hg3)
    kpLocation = [kp.pt for kp in keypoints]
    kpSize = [kp.size for kp in keypoints]

    mask = np.zeros(img.shape[:-1], np.uint8)
    for i, kploc in enumerate(kpLocation):
        cv2.circle(mask, (int(kploc[0]), int(kploc[1])), int(kpSize[i]), 1, -1)  # Last -1 means filled, return 0-1 mask

    return mask

