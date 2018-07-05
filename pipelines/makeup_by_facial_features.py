import numpy as np

from blending import makeup_transfer_by_example
from facial_attributes.facial_landmarks import facial_landmarks_detection_dlib
from facial_attributes.facial_segment import face_segmentation
from utils import face_process


def _get_facial_feature_landmarks(landmark_mode=68):
    feature_index_dict = {}
    if landmark_mode == 68:
        left_eye_area_index = np.arange(0, 6)
        left_eye_area_index = np.hstack((left_eye_area_index, np.arange(17, 22)))
        left_eye_area_index = np.hstack((left_eye_area_index, np.arange(36, 42)))
        feature_index_dict.update({'left_eye_area': left_eye_area_index.tolist()})

        right_eye_area_index = np.arange(11, 17)
        right_eye_area_index = np.hstack((right_eye_area_index, np.arange(22, 27)))
        right_eye_area_index = np.hstack((right_eye_area_index, np.arange(42, 48)))
        feature_index_dict.update({'right_eye_area': right_eye_area_index})

        up_lip_index = np.arange(48, 55)
        up_lip_index = np.hstack((up_lip_index, np.arange(60, 65))).tolist()
        feature_index_dict.update({'up_lip': up_lip_index})

        bottom_lip_index = [48]
        bottom_lip_index.extend(np.arange(54, 61).tolist())
        bottom_lip_index.extend(np.arange(64, 68).tolist())
        feature_index_dict.update({'bottom_lip': bottom_lip_index})
    else:
        raise ValueError('There is no such landmark mode')

    return feature_index_dict


def get_triangle(landmarks, landmark_mode=68):
    feature_index_dict = _get_facial_feature_landmarks(landmark_mode=landmark_mode)
    triangle_indices = list()
    triangle_mesh = list()
    for key in feature_index_dict.keys():
        index = feature_index_dict[key]
        tmp = np.array(landmarks)[index]
        _, triangles_tmp, triangle_mesh_tmp = face_segmentation.get_triangle_landmarks(tmp)

        if key == 'up_lip':
            bottom_contour = [0]
            bottom_contour.extend(np.arange(6, 12).tolist())
            selected_triangle_index = list()
            for i, triangle_tmp in enumerate(triangles_tmp):
                if (triangle_tmp[0] in bottom_contour) & \
                    (triangle_tmp[1] in bottom_contour) & \
                        (triangle_tmp[2] in bottom_contour):
                    continue
                selected_triangle_index.append(i)

            triangle_mesh_tmp = np.array(triangle_mesh_tmp)[selected_triangle_index]
            triangle_mesh_tmp = triangle_mesh_tmp.tolist()
        if key == 'bottom_lip':
            up_contour = np.arange(7, 12).tolist()
            selected_triangle_index = list()
            for i, triangle_tmp in enumerate(triangles_tmp):
                if (triangle_tmp[0] in up_contour) & \
                        (triangle_tmp[1] in up_contour) & \
                        (triangle_tmp[2] in up_contour):
                    continue
                selected_triangle_index.append(i)

            triangle_mesh_tmp = np.array(triangle_mesh_tmp)[selected_triangle_index]
            triangle_mesh_tmp = triangle_mesh_tmp.tolist()

            triangles_tmp = np.array(triangles_tmp)[selected_triangle_index]
            triangles_tmp = triangles_tmp.tolist()

        origin_triangles = list()
        for triangle_tmp in triangles_tmp:
            origin_triangles.append([index[triangle_tmp[0]], index[triangle_tmp[1]], index[triangle_tmp[2]]])

        triangle_indices.extend(origin_triangles)
        triangle_mesh.extend(triangle_mesh_tmp)
    return np.array(triangle_indices), np.array(triangle_mesh)


def get_triangle_indices(triangle_indices, landmark_mode=68):
    res = list()
    feature_index_dict = _get_facial_feature_landmarks(landmark_mode=landmark_mode)
    for key in feature_index_dict.keys():
        values = feature_index_dict[key]
        for i, triangle_index in enumerate(triangle_indices):
            if (triangle_index[0] in values) & (triangle_index[1] in values) & (triangle_index[2] in values):
                res.append(i)
    return res


def makeup_by_facial_feature(target_image, example_image):
    target_landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(target_image)
    example_landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(example_image)
    target_triangle, target_triangle_mesh = get_triangle(target_landmarks)
    example_triangle_mesh = face_segmentation.get_triangle_mesh(example_landmarks, target_triangle)

    target_pts, example_pts = face_segmentation.get_pixels_warp(target_triangle_mesh,
                                                                target_image.shape, example_triangle_mesh)
    alpha_map, _ = face_process.get_new_face_mask(example_image)
    fusion_image = \
        makeup_transfer_by_example.color_blending(target_image,
                                                  target_pts,
                                                  example_image,
                                                  example_pts, alpha_map=alpha_map)

    return fusion_image
