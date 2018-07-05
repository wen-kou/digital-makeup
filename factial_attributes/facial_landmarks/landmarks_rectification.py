import numpy as np
import cv2
import copy

from utils import face_process
from utils.face_process import select_face_rect, rescale_landmarks


def landmarks_rectify(image, landmarks, gmm_color, verify_dir='horizontal', resolution_list=tuple([256])):

    t_list = list()
    face_rect = select_face_rect(landmarks)

    tmp_landmarks = landmarks
    tmp_image = image
    window_size = 8
    for i, resolution in enumerate(resolution_list):
        window_size = int(window_size*resolution/256)

        face_image, face_image_rect, t = face_process.calc_image_transform(face_rect,
                                                                           tmp_image,
                                                                           resolution=resolution)
        t_list.append(t)

        new_landmarks = rescale_landmarks(t, tmp_landmarks)
        rectified_new_landmarks = _landmark_rectify(np.array(face_image, dtype=np.uint8),
                                                    new_landmarks,
                                                    gmm_color,
                                                    search_dir=verify_dir,
                                                    window_size=window_size)

        tmp_image = face_image

        face_rect = {'left_top': tuple([0, 0]),
                     'bottom_right': tuple([tmp_image.shape[0], tmp_image.shape[1]])}
        tmp_landmarks = rectified_new_landmarks

    result = tmp_landmarks
    for t in reversed(t_list):
        result = rescale_landmarks(t, result, invert=True)

    return result


def _landmark_rectify(image, landmarks, color_model,
                      filter_func='laplacian',
                      search_dir=None,
                      window_size=15):
    new_landmarks = copy.copy(landmarks)
    # image_size = tuple([image.shape[0], image.shape[1]])
    for i, landmark in enumerate(landmarks):
        if search_dir == 'horizontal':
            sub_images = image[landmark[1]:landmark[1] + 1,
                               landmark[0] - window_size: landmark[0] + window_size + 1]
        elif search_dir == 'vertical':
            sub_images = image[landmark[1] - window_size:landmark[1] + window_size + 1,
                               landmark[0]:landmark[0] + 1]
        else:
            sub_images = image[landmark[1] - window_size:landmark[1] + window_size + 1,
                               landmark[0] - window_size: landmark[0] + window_size + 1]
        x = np.arange(0, sub_images.shape[1])
        y = np.arange(0, sub_images.shape[0])
        xv,yv = np.meshgrid(x, y)

        if search_dir == 'horizontal':
            center_point = tuple([landmark[0] if (landmark[0] - window_size) < 0 else window_size, 0])
        elif search_dir == 'vertical':
            center_point = tuple([0, landmark[1] if (landmark[1] - window_size) < 0 else window_size])
        else:
            center_point = tuple([landmark[1] if (landmark[1] - window_size) < 0 else window_size,
                                  landmark[0] if (landmark[0] - window_size) < 0 else window_size])

        dist_score = _calc_distance(np.asarray([xv,yv]), center_point)
        gradient_score = _calc_gradient(sub_images, func=filter_func)
        color_score, weights = _calc_color_score(color_model, sub_images, center_point, weighted=True)
        if weights is not None:
            score = np.multiply(weights, dist_score + gradient_score + color_score)
        else:
            score = dist_score + gradient_score + color_score
        ind = np.argmax(score)
        coord = np.unravel_index(ind, (sub_images.shape[0], sub_images.shape[1]))
        row, col = coord[0], coord[1]

        diff_row = row - center_point[1]
        diff_col = col - center_point[0]
        new_landmark = tuple([diff_col + landmark[0], diff_row + landmark[1]])
        new_landmarks[i] = new_landmark

    return new_landmarks


def _calc_distance(coordinates, ref_pt, sigma=10):
    coordinates[0] = coordinates[0] - ref_pt[0]
    coordinates[1] = coordinates[1] - ref_pt[1]
    dist = np.linalg.norm(coordinates, axis=0)
    dist = np.reshape(dist, (dist.shape[0]*dist.shape[1], 1))
    dist_score = np.exp(-dist/(sigma*sigma))
    return dist_score


def _calc_gradient(image, func='laplacian'):
    if image.ndim >= 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.blur(image,ksize=(3,3))
    if func == 'laplacian':
        gradient_score = np.abs(cv2.Laplacian(image,cv2.CV_64F)) + np.finfo(float).eps

    elif func == 'sobel':
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        gradient_score = np.abs(sobelx + sobely) + np.finfo(float).eps
    else:
        raise ValueError('No such filter')

    gradient_score = 1 / gradient_score
    sigma = np.median(gradient_score)
    gradient_score = np.exp(- gradient_score / (sigma * sigma))
    if image.ndim >= 2:
        gradient_score = np.resize(gradient_score,
                                  (gradient_score.shape[0]*gradient_score.shape[1], 1))

    return gradient_score


def _calc_color_score(color_model, image, center_point, weighted=False):
    color_samples = np.reshape(image, (image.shape[0] * image.shape[1], 3)) / 255
    color_score = color_model.score(color_samples).reshape((image.shape[0] * image.shape[1],1))
    color_score = color_score - np.min(color_score) + np.finfo(float).eps
    color_score = 1 / color_score
    sigma = np.median(color_score)
    color_score = np.exp(-color_score / (sigma*sigma))

    weight = None
    if weighted is True:
        color_score_map = np.reshape(color_score, (image.shape[0], image.shape[1]))
        left_weight = np.sum(color_score_map[:, 0:center_point[0] + 1])
        right_weight = np.sum(color_score_map[:, center_point[0] + 1:])
        weight = np.zeros((image.shape[0], image.shape[1]))
        weight[:, 0:center_point[0] + 1] = left_weight
        weight[:, center_point[0] + 1:] = right_weight
        weight = np.reshape(weight, (image.shape[0] * image.shape[1], 1))
    return color_score, weight







