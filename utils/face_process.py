import cv2
import copy
import dlib
import numpy as np

from scipy.linalg import inv
from utils import knn_matte
from facial_attributes.facial_landmarks import facial_landmarks_detection_dlib

global detector
detector = dlib.get_frontal_face_detector()


def get_face_rect(image):
    detected_faces = detector(image, 1)
    largest_area = 0
    res = None
    if len(detected_faces) > 0:
        for i, rect in enumerate(detected_faces):
            area = (rect.bottom() - rect.top()) * (rect.right() - rect.left())
            if area > largest_area:
                largest_area = area
                res = rect

    return res


def get_face_image(image):
    rect = get_face_rect(image)
    if rect is None:
        return None
    face_rect = image[rect.top():rect.bottom(), rect.left():rect.right()]
    return face_rect


def _whiteness_analysis(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    whiteness = np.sort(np.reshape(image_gray, (image_gray.shape[0]*image_gray.shape[1])))
    num = image_gray.shape[0]*image_gray.shape[1]
    threds0 = whiteness[-1]
    threds1 = whiteness[int(1*num - 1)]
    threds2 = whiteness[int(0.2*num)]

    return threds0, threds1, threds2


def get_new_face_mask(image, landmarks=None, segment=False):
    face = get_face_image(image)
    max_, threds1, threds2 = _whiteness_analysis(face)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = copy.copy(image_gray) / 255
    mask = 1.5 - mask
    mask = np.tanh(mask*1.5)

    if segment is True:
        face_rect = {
            'left_top': tuple([0, 0]),
            'bottom_right': tuple([image.shape[1], image.shape[0]])
        }
        image_rescale, _, t = calc_image_transform(face_rect, image, ratio=1, resolution=256)
        if landmarks is None:
            landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(image_rescale)
        else:
            landmarks = rescale_landmarks(t, landmarks)
        contour = landmarks[0:17]

        left_eye_brow = np.asarray(list(reversed(landmarks[22:27])))
        left_eye_brow[:,1] = left_eye_brow[:,1] - 10
        left_eye_brow[np.where(left_eye_brow<0)] = 0
        contour.extend(left_eye_brow.tolist())

        right_eye_brow = np.asarray(list(reversed(landmarks[17:22])))
        right_eye_brow[:, 1] = right_eye_brow[:, 1] - 10
        right_eye_brow[np.where(right_eye_brow<0)] = 0
        contour.extend(right_eye_brow)
        trimap= knn_matte.get_face_trimap(image_rescale, contour, np.ones((8,8)))

        # cv2.imshow('trimap', np.asarray(trimap, dtype=np.uint8))

        matting = knn_matte.knn_matte(image_rescale, trimap)

        matting = cv2.resize(matting, dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow('alpha', np.asarray(255 * matting, dtype=np.uint8))
        # cv2.imshow('fusion_weights', np.asarray(255 * mask, dtype=np.uint8))
        # cv2.waitKey(500)
        mask = np.multiply(matting, mask)
    scalar = np.repeat(mask, 3, axis=1)
    scalar = np.reshape(scalar, image.shape)
    res = np.multiply(scalar, image)
    return mask, res


def _calc_translate(original_center, new_center, scale):
    """
    :param original_center: tuple
    :param new_center: tuple
    :return:
    """
    return [new_center[0] - scale * original_center[0],
            new_center[1] - scale * original_center[1]]


def _calc_scale(origin_resolution, new_resolution):
    return new_resolution / origin_resolution


def _calc_transfer_mat(origin_center, new_center, origin_resolution, new_resolution=256):
    transform = np.eye(3)
    scale = _calc_scale(origin_resolution, new_resolution)
    translate = _calc_translate(origin_center, new_center, scale)

    transform[0, 2] = translate[0]
    transform[1, 2] = translate[1]
    transform[0, 0] = scale
    transform[1, 1] = scale
    return transform


def _calc_origin_center(face_rect):
    center = [(face_rect['bottom_right'][0] + face_rect['left_top'][0]) / 2,
              (face_rect['bottom_right'][1] + face_rect['left_top'][1]) / 2]
    # center[1] = center[1] - 0.1 * (face_rect['bottom_right'][1] - face_rect['left_top'][1])
    return tuple(center)


def _select_face_area(face_rect, image_size, ratio):
    """
    :param face_rect: dict {'top_left': [x,y], 'bottom_right': [x,y]}
    :param image_size: tuple (height, width)
    :param ratio: float
    :return:
    """
    center = _calc_origin_center(face_rect)

    origin_height = face_rect['bottom_right'][1] - face_rect['left_top'][1]
    origin_width = face_rect['bottom_right'][0] - face_rect['left_top'][0]

    new_height = origin_height * ratio
    new_width = origin_width * ratio

    if center[0] + int(new_width / 2) > image_size[1]:
        new_right = image_size[1] - 1
    else:
        new_right = center[0] + int(new_width / 2)

    if center[1] + int(new_height / 2) > image_size[0]:
        new_bottom = image_size[0] - 1
    else:
        new_bottom = center[1] + int(new_height / 2)
    new_bottom_right = tuple([int(new_right), int(new_bottom)])

    if center[0] - int(new_width / 2) < 0:
        new_left = 0
    else:
        new_left = center[0] - int(new_width / 2)

    if center[1] - int(new_height / 2) < 0:
        new_top = 0
    else:
        new_top = center[1] - int(new_height / 2)
    new_top_left = tuple([int(new_left), int(new_top)])
    return {'left_top': new_top_left, 'bottom_right': new_bottom_right}


def calc_transform_pts(pts, transform, invert=False):
    if invert is True:
        transform = inv(transform)
    origin_points = np.ones(3)
    origin_points[0] = pts[0]
    origin_points[1] = pts[1]
    return np.array(np.matmul(transform, origin_points), dtype=int)[0:2]


def calc_image_transform(face_rect, image, ratio=1.1, resolution=256):
    """
    :param face_rect:
    :param image: color image with shape (height, width, channel=3)
    :param ratio:
    :param resolution:
    :return:
    """
    image_size = tuple([image.shape[0], image.shape[1]])
    new_rect = _select_face_area(face_rect, image_size, ratio)
    new_height = int(new_rect['bottom_right'][1] - new_rect['left_top'][1])
    new_width = int(new_rect['bottom_right'][0] - new_rect['left_top'][0])
    origin_center = _calc_origin_center(new_rect)

    new_image = image[new_rect['left_top'][1]: new_rect['left_top'][1] + new_height,
                      new_rect['left_top'][0]: new_rect['left_top'][0] + new_width, :]

    origin_resolution = max(new_height, new_width)
    fx, fy = resolution/origin_resolution, resolution/origin_resolution
    new_image = cv2.resize(new_image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

    scaled_center = tuple([new_image.shape[1]/2, new_image.shape[0]/2])
    t = _calc_transfer_mat(origin_center, scaled_center, origin_resolution, new_resolution=resolution)
    return new_image, new_rect, t


def select_face_rect(landmarks):
    """
    :param landmarks: list (tuple)
    :param image_size: tuple
    :return:
    """
    left_top = np.min(np.asarray(landmarks), axis=0)
    bottom_right = np.max(np.asarray(landmarks), axis=0)

    face_height = bottom_right[1] - left_top[1]

    left_top[1] = (left_top[1] - face_height / 3) if (left_top[1] - face_height / 3) > 0 else 0

    face_rect = {'left_top': tuple([int(left_top[0]), int(left_top[1])]),
                 'bottom_right': tuple([int(bottom_right[0]), int(bottom_right[1])])}

    return face_rect


def rescale_landmarks(transform_t, landmarks, invert=False):
    new_landmarks = []
    for landmark in landmarks:
        new_landmark = calc_transform_pts(landmark, transform_t, invert=invert)
        new_landmarks.append(tuple(new_landmark.tolist()))
    return new_landmarks