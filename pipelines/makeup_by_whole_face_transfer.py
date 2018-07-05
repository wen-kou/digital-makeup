import cv2
import numpy as np
import utils.face_process

from scipy.linalg import inv
from facial_attributes.facial_landmarks import facial_landmarks_detection_dlib
from facial_attributes.facial_landmarks import landmarks_rectification
from facial_attributes.facial_segment import face_segmentation
from facial_attributes.head_pose import head_pose_estimation
from utils import face_process
from blending.makeup_transfer_by_example import color_blending


def _get_facial_gmm_model(image, landmarks):
    image_size = tuple([image.shape[0], image.shape[1]])
    facial_region_index = face_segmentation.get_facial_indices(landmarks, image_size)
    gmm_color = face_segmentation.gmm_color_model(image, facial_region_index)
    return gmm_color


def add_landmarks(landmarks, image_size, num_intervals=8):
    interval = int(image_size[1] / num_intervals)
    for i in range(num_intervals):
        landmarks.append(tuple([interval * i, 0]))
    return landmarks


def pre_proc_example(example_image):
    example_landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(example_image)
    resolution = 512
    example_face_rect = face_process.select_face_rect(example_landmarks)
    new_example_face, new_example_face_rect, t_example = face_process.calc_image_transform(example_face_rect,
                                                                                           example_image,
                                                                                           resolution=resolution)

    new_example_landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(new_example_face)
    new_example_landmarks = add_landmarks(new_example_landmarks,
                                              tuple([new_example_face.shape[0], new_example_face.shape[1]]))

    alpha_map_example, _ = face_process.get_new_face_mask(new_example_face, segment=True)

    return new_example_face, alpha_map_example, new_example_landmarks


def makeup_by_whole_face_transfer(target_image, example_face, example_alpha, example_landmarks, rectify_landmarks=False):
    target_landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(target_image)
    resolution = 512
    if rectify_landmarks is True:
        target_gmm_color = _get_facial_gmm_model(target_image, target_landmarks)
        target_contour_landmarks = target_landmarks[0:17]
        target_contour_landmarks = landmarks_rectification.landmarks_rectify(target_image,
                                                                             target_contour_landmarks,
                                                                             target_gmm_color,
                                                                             resolution_list=tuple([516])
                                                                             )
        target_landmarks[0:17] = target_contour_landmarks
        example_gmm_color = _get_facial_gmm_model(example_face, example_landmarks)
        example_contour_landmarks = example_landmarks[0:17]
        example_contour_landmarks = landmarks_rectification.landmarks_rectify(example_face,
                                                                              example_contour_landmarks,
                                                                              example_gmm_color,
                                                                              resolution_list=tuple([516])
                                                                              )
        example_landmarks[0:17] = example_contour_landmarks

    target_face_rect = face_process.select_face_rect(target_landmarks)
    new_target_face, new_target_face_rect, t_target = face_process.calc_image_transform(target_face_rect,
                                                                                        target_image,
                                                                                        resolution=resolution)
    target_landmarks_rescale = utils.face_process.rescale_landmarks(t_target, target_landmarks)
    origin_new_target_face_shape = tuple([new_target_face.shape[1], new_target_face.shape[0]])

    t_head_pose_rectify, rectify_head_pose_image_shape = \
        head_pose_estimation.landmarks_rectify_by_head_pose(origin_new_target_face_shape, target_landmarks_rescale)

    if np.array_equal(t_head_pose_rectify, np.eye(3)) is False:
        new_target_face = cv2.warpAffine(new_target_face, t_head_pose_rectify[0:2], rectify_head_pose_image_shape)
        target_landmarks_rescale = utils.face_process.rescale_landmarks(t_head_pose_rectify, target_landmarks_rescale)

    # cv2.imshow('rectify_face', new_target_face)
    # cv2.waitKey(500)

    target_landmarks_rescale = add_landmarks(target_landmarks_rescale,
                                             tuple([new_target_face.shape[0], new_target_face.shape[1]]))

    alpha_map_target, _ = face_process.get_new_face_mask(new_target_face, segment=True)
    alpha_map_example = cv2.cvtColor(example_alpha, cv2.COLOR_BGR2GRAY) / 255

    # cv2.imshow('alpha_map',
    #            np.concatenate((np.asarray(alpha_map_example * 255, dtype=np.uint8),
    #                            np.asarray(alpha_map_target * 255, dtype=np.uint8)), axis=1))
    # cv2.imshow('origin_image',
    #            np.concatenate((new_example_face,
    #                            new_target_face), axis=1))
    # cv2.waitKey(500)
    triangle_index_path = '../resources/landmark_triangle_index.txt'
    triangle_index = np.loadtxt(triangle_index_path, dtype=int)
    # _, target_triangle, target_triangle_mesh = face_segmentation.get_triangle_landmarks(target_landmarks_rescale)

    target_triangle_mesh = face_segmentation.get_triangle_mesh(target_landmarks_rescale, triangle_index)
    example_triangle_mesh = face_segmentation.get_triangle_mesh(example_landmarks, triangle_index)

    target_pts, ref_pts = face_segmentation.get_pixels_warp(target_triangle_mesh,
                                                            new_target_face.shape,
                                                            example_triangle_mesh)

    fusion_face = color_blending(new_target_face,
                                 target_pts,
                                 example_face,
                                 ref_pts,
                                 alpha_map=alpha_map_example,
                                 alpha=0.7)
    alpha_map_target = alpha_map_target.reshape((alpha_map_target.shape[0], alpha_map_target.shape[1], 1))

    # cv2.imshow('face', np.asarray(np.multiply(np.repeat(alpha_map_target,3,axis=2), fusion_face),dtype=np.uint8))
    # cv2.imshow('back', np.asarray(np.multiply(1 - np.repeat(alpha_map_target, 3, axis=2), new_target_face), dtype=np.uint8))
    # cv2.waitKey(300)
    fusion_face = np.multiply(np.repeat(alpha_map_target, 3, axis=2), fusion_face) + \
                  np.multiply(np.ones((alpha_map_target.shape[0], alpha_map_target.shape[1], 3)) -
                              np.repeat(alpha_map_target, 3, axis=2), new_target_face)
    # cv2.imshow('fusion_face', np.asarray(fusion_face, dtype=np.uint8))
    # cv2.waitKey(500)

    origin_coords = new_target_face_rect['left_top']
    origin_height = new_target_face_rect['bottom_right'][1] - new_target_face_rect['left_top'][1]
    origin_width = new_target_face_rect['bottom_right'][0] - new_target_face_rect['left_top'][0]

    if np.array_equal(t_head_pose_rectify, np.eye(3)) is False:
        fusion_face = cv2.warpAffine(fusion_face, inv(t_head_pose_rectify)[0:2], origin_new_target_face_shape)

    fusion_face_rescale = cv2.resize(fusion_face,
                                     dsize=(origin_width, origin_height),
                                     interpolation=cv2.INTER_LINEAR)

    target_image[int(origin_coords[1]):int(origin_coords[1]) + origin_height,
    int(origin_coords[0]):int(origin_coords[0]) + origin_width, :] = fusion_face_rescale

    return target_image
