from numpy.linalg import inv
from multiprocessing.pool import ThreadPool
from scipy.spatial import Delaunay
from sklearn.mixture import GMM

import numpy as np
import cv2

global pool
pool = ThreadPool(1024)


def draw_landmark(face_img, landmark_list, color=(0, 255, 255)):
    res = face_img
    for landmark in landmark_list:
        cv2.circle(res, (int(landmark[0]),int(landmark[1])), 3, color, thickness=-1)
    return res


def find_face_region(landmark_list_for_one_region, image_size):
    '''
    :param landmark_list_for_one_region: list of landmark points
    :param image_size: tuple with 3 elements
    :return:
    '''
    mask = np.zeros(image_size, dtype=np.uint8)
    landmark_list_for_one_region = np.array([list(landmark) for landmark in landmark_list_for_one_region], dtype=np.int32)
    mask = cv2.fillPoly(mask, [landmark_list_for_one_region], (255))
    return np.where(mask == 255)[0:2]


def remove_region(origin_index, remove_region_list, image_size):
    mask = np.zeros(image_size, dtype=np.uint8)
    mask[origin_index[0], origin_index[1]] = 255
    for remove_region in remove_region_list:
        mask[remove_region[0], remove_region[1]] = 0
    return np.where(mask == 255)[0:2]

def get_triangle_landmarks(landmarks):
    if isinstance(landmarks, dict):
        landmark_coord = []
        for key, value in landmarks.items():
            landmark_coord.extend(value)
    landmark_coord = landmarks
    triangles = Delaunay(landmark_coord)
    return landmark_coord, triangles.simplices, get_triangle_mesh(landmark_coord, triangles.simplices)


def get_triangle_mesh(coords, triangles_indices):
    return list(map(lambda tri_index: [coords[tri_index[0]],
                                       coords[tri_index[1]],
                                       coords[tri_index[2]]], triangles_indices))


def calc_triangle_affine_transformation(pts_source, pts_target):
    pts_source = np.asarray(pts_source)
    ones = np.ones((3, 1))
    pts_source = np.hstack((pts_source, ones)).transpose()

    pts_target = np.asarray(pts_target)
    pts_target = np.hstack((pts_target, ones)).transpose()

    try:
        affine_transformation = np.matmul(pts_target, inv(pts_source))
    except:
        affine_transformation = None
        print("Input source points at least two {} are in the same point".format(pts_source))

    return affine_transformation


def get_ref_pixels_by_affine_transformation(source_pixels, affine_matrix):
    tmp_source_pixels = np.asarray(source_pixels)
    one = np.ones((1, tmp_source_pixels.shape[1]))
    input = np.vstack((tmp_source_pixels, one))
    output = np.matmul(affine_matrix, input)
    return output


def pixel_transfer(affine_trans, source_pixels):
    tmp_source_pixels = np.asarray(source_pixels)
    one = np.ones((1,tmp_source_pixels.shape[1]))
    input = np.vstack((tmp_source_pixels, one))
    output = np.matmul(affine_trans, input)

    scalar = np.repeat(output[2,:], 3, axis=0)
    scalar = np.reshape(scalar, output.shape)
    output = np.divide(output, scalar)

    return output[0:2,:]


def _triangle_pixels_warp(ptSource, sourceImgSize, ptTarget):
    affine_h = calc_triangle_affine_transformation(ptSource,ptTarget)

    source_pixels = find_face_region(ptSource, sourceImgSize)
    source_pixels = [source_pixels[1], source_pixels[0]]
    output_pixels = pixel_transfer(affine_h, source_pixels)

    source_pixels = tuple([source_pixels[1].tolist(), source_pixels[0].tolist()])
    target_pixels = tuple([output_pixels[1,:].tolist(), output_pixels[0,:].tolist()])
    return source_pixels, target_pixels


def get_pixels_warp(ptSource_list, sourceImgSize, ptTarget_list):
    inputs = [ptSource_list, ptTarget_list]
    inputs = np.asarray(list(map(list, zip(*inputs))))
    results = \
        pool.map(lambda input:
                 _triangle_pixels_warp(input[0], sourceImgSize, input[1]),
                 inputs)
    sourcePixels = []
    outputPixels = []

    for res in results:
        sourcePixels.extend(list(map(list,zip(*res[0]))))
        outputPixels.extend(list(map(list,zip(*res[1]))))

    return sourcePixels, outputPixels


def gmm_color_model(image, sample_region):
    color_list = image[sample_region[0], sample_region[1], :] / 255
    model = GMM(n_components=3)
    model.fit(color_list)
    return model


def get_facial_indices(landmarks, image_size):
    contour = landmarks[0:17]
    contour.extend(reversed(landmarks[42:46]))
    contour.extend(reversed(landmarks[36:40]))
    facial_region = find_face_region(contour, image_size)
    left_eye_landmarks = landmarks[36: 42]
    left_eye_region = find_face_region(left_eye_landmarks, image_size)
    right_eye_landmarks = landmarks[42:48]
    right_eye_region = find_face_region(right_eye_landmarks, image_size)
    lip_landmarks = landmarks[48:60]
    lip_region = find_face_region(lip_landmarks, image_size)
    remove_regions = [left_eye_region, right_eye_region, lip_region]
    facial_segments = remove_region(facial_region, remove_regions, image_size)
    return facial_segments
