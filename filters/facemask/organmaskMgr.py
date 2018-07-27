# July/2018 LzCai

import cv2
import numpy as np
from filters.facemask import constants
from filters import render_util


def update_one_organ(mask, organ_landmark):
    organ_whole_mask = get_organ_whole_mask(organ_landmark,mask.shape)
    return np.clip(mask + organ_whole_mask, 0, 1)


def _get_organ_location(landmark):
    ys, xs = landmark[:, 1], landmark[:, 0]
    top, bottom, left, right = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
    shape = (int(bottom - top), int(right - left))
    size = shape[0] * shape[1] * 3
    move = int(np.sqrt(size / 3) / 20)
    location = [top, bottom, left, right, shape, size, move]
    return location


def _get_organ_patch_mask(landmark, location, img_shape):
    top, bottom, left, right, shape, size, move = location[:]

    landmark_re = landmark.copy()
    landmark_re[:, 1] -= np.max([top - move, 0])
    landmark_re[:, 0] -= np.max([left - move, 0])
    mask = np.zeros(img_shape[:2], dtype=np.float64)
    mask = mask[np.max([top - move, 0]):np.min([bottom + move, img_shape[0]]),
           np.max([left - move, 0]):np.min([right + move, img_shape[1]])]
    points = cv2.convexHull(landmark_re)
    mask = cv2.fillConvexPoly(mask, points, color=1).astype(np.uint8)
    mask = np.array([mask, mask, mask]).transpose((1, 2, 0))

    return mask


def get_organ_whole_mask(landmark, image_shape):
    location = _get_organ_location(landmark)
    patch_mask = _get_organ_patch_mask(landmark, location, image_shape)
    mask = np.zeros(image_shape, dtype=np.float64)
    top, bottom, left, right, shape, size, move = location[:]

    rowStart = np.max([top - move, 0])
    rowEnd = np.min([bottom + move, image_shape[0]])
    colStart = np.max([left - move, 0])
    colEnd = np.min([right + move, image_shape[1]])

    mask[rowStart:rowEnd, colStart:colEnd] += patch_mask

    return mask


def get_acne_mask(faces, img):
    masks = []  # For all faces

    for face in faces:
        # tmp: LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD
        for tmp in [0, 1, 2]:
            # for tmp in [1]:
            if tmp == constants.LEFT_CHEEK:
                partial_landmark = face.landmark[constants.LEFT_CHEEK_POINTS]

            elif tmp == constants.RIGHT_CHEEK:
                partial_landmark = face.landmark[constants.RIGHT_CHEEK_POINTS]

            else:
                partial_landmark = face.organs[-1].landmark

            tmp_mask = np.zeros(img.shape[:-1], dtype=np.uint8)
            points = np.array([[i] for i in partial_landmark.tolist()])
            tmp_mask = cv2.fillConvexPoly(tmp_mask, points, color=1).astype(np.uint8)
            tmp_mask = np.array([tmp_mask, tmp_mask, tmp_mask]).transpose((1, 2, 0))

            tmp_mask = _get_acne_mask_on_patch(img, tmp_mask)

            masks.append(tmp_mask)  # Append one face mask on all faces

    return np.clip(sum(masks), 0, 1)


def _get_acne_mask_on_patch(img, in_mask):
    img = img * in_mask
    g = cv2.split(img)[1]
    hard_g = render_util.hard_light(g, g)
    hard_g2 = render_util.hard_light(hard_g, hard_g)
    hard_g3 = render_util.hard_light(hard_g2, hard_g2)
    tri_hg3 = render_util.tri_color(hard_g3)

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
        cv2.circle(mask, (int(kploc[0]), int(kploc[1])), int(kpSize[i]), 1,
                   -1)  # Last -1 means filled, return 0-1 mask
    mask = np.array([mask, mask, mask]).transpose((1, 2, 0))

    return mask
