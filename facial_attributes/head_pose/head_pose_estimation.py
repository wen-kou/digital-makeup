import cv2
import numpy as np
import copy
import math
import os


def get_camera_intrinsic(image_size):
    """
    :param image_size: tuple (width)
    :return:
    """
    center = tuple([image_size[0] / 2, image_size[1] / 2])
    focal_len = image_size[0]
    camera_model = np.eye(3)
    camera_model[0, 0] = focal_len
    camera_model[1, 1] = focal_len
    camera_model[0, 2] = center[0]
    camera_model[1, 2] = center[1]

    return np.asarray(camera_model, dtype=np.float32)


def get_initial_landmarks_2d(camera_model):
    facial_landmarks_3d = get_3d_landmarks_68()
    initial_2d_landmarks = np.matmul(camera_model, facial_landmarks_3d)

    return initial_2d_landmarks


def get_3d_landmarks_68():
    current_path = os.path.abspath(__file__)
    model_path = os.path.abspath('../resources/3d_facial_landmarks_model.txt')
    facial_landmarks_3d = np.loadtxt(model_path, dtype=float)
    facial_landmarks_3d = np.reshape(facial_landmarks_3d, (3, 68))
    return facial_landmarks_3d.transpose()


def head_pose_estimation(image_size, landmarks):
    """
    :param image_size: tuple(width, height)
    :param landmarks:
    :return: rotation (pitch, yaw, roll), translation
    """
    landmarks_copy = copy.copy(landmarks)
    landmarks_copy = np.asarray(landmarks_copy, dtype=np.float32)
    camera_model = get_camera_intrinsic(image_size)
    facial_landmarks_3d = get_3d_landmarks_68()
    (_, rotation_vector, translation_vector) = cv2.solvePnP(
        facial_landmarks_3d,
        landmarks_copy,
        camera_model,
        distCoeffs=np.zeros((4, 1)))
    return rotation_vector, translation_vector


def landmarks_rectify_by_head_pose(image_size, landmarks):
    """
    :param image_size: tuple (width, height)
    :param landmarks: landmark list
    :return: rectify_matrix
    """
    r, t = head_pose_estimation(image_size, landmarks)
    # only handle roll
    # roll = -(180 - r[2] * 180 / np.pi)
    roll = r[2] * 180 / np.pi
    if (np.abs(roll) > 20) & (np.abs(180 - roll) > 20):
        scale = 1.0
        t = np.eye(3)
        roll = roll - 180
        m = cv2.getRotationMatrix2D((image_size[0] / 2, image_size[1] / 2), roll, scale)
        t[0:2, :] = m
        new_size = calc_new_image_size_by_rotation(image_size, t)
        t[0, 2] += (new_size[0] - image_size[0] + 1)/2
        t[1, 2] += (new_size[1] - image_size[1] + 1)/2
        return t, new_size
    else:
        return np.eye(3), image_size


def calc_new_image_size_by_rotation(origin_image_size, t):
    corner_coords = np.array([[0, 0, 1],
                             [0, origin_image_size[1],  1],
                             [origin_image_size[0], 0, 1],
                             [origin_image_size[0], origin_image_size[1],   1]]).transpose()
    new_corner_coords = np.matmul(t, corner_coords)[0:2]
    left = np.min(new_corner_coords[0, :])
    right = np.max(new_corner_coords[0, :])
    new_width = int(right - left)
    top = np.min(new_corner_coords[1, :])
    bottom = np.max(new_corner_coords[1, :])
    new_height = int(bottom - top)
    return tuple([new_width, new_height])


def _is_rotation_ratrix(R):
    should_be_identity = np.dot(np.transpose(R), R)
    eye = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(eye - should_be_identity)
    return n < 1e-6


def rotation_matrix2euler_angles(R):
    assert (_is_rotation_ratrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])