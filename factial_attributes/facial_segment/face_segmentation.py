from numpy.linalg import inv
from multiprocessing.pool import ThreadPool
from scipy.spatial import Delaunay

import numpy as np
import cv2

global pool
pool = ThreadPool(1024)


def draw_landmark(face_img, landmark_list):
    for landmark in landmark_list:
        cv2.circle(face_img, (int(landmark[0]),int(landmark[1])), 3, (0, 255, 255), thickness=False)
    return face_img


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


def get_triangle_landmarks(landmark_dict):
    landmark_coord = []
    for key, value in landmark_dict.items():
        landmark_coord.extend(value)
    #
    # for i, value in enumerate(landmark_coord):
    #     landmark_coord[i] = tuple([value[1], value[0]])

    triangles = Delaunay(landmark_coord)

    return landmark_coord, triangles.simplices


def calc_triangle_affine_transformation(ptSource, ptTarget):
    ptSource = np.asarray(ptSource)
    ones = np.ones((3, 1))
    ptSource = np.hstack((ptSource, ones)).transpose()

    ptTarget = np.asarray(ptTarget)
    ptTarget = np.hstack((ptTarget, ones)).transpose()

    try:
        affine_transformation = np.matmul(ptTarget, inv(ptSource))
    except:
        affine_transformation = None
        print("Input source points {} is in the same line".format(ptSource))

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
    output = np.divide(output, scalar)

    return output


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



