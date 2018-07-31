# July/2018 LzCai

import cv2
import numpy as np
import dlib
from filters.facemask import constants
from filters.facemask import organmaskMgr



class Image:
    def __init__(self, img):
        self.img = img
        self.shape = img.shape
        self.faceList = self.detect_all_faces()

    def detect_all_faces(self):
        face_landmark_list = self._get_all_face_landmark()
        face_list = []
        for temp_landmark in face_landmark_list:
            face_list.append(Face(self.img, temp_landmark))
        return face_list

    def _get_all_face_landmark(self):  # Will detect and return ALL faces in the image
        predictor_path = '../../resources/shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        rects = detector(self.img, 1)
        result = [np.matrix([[p.x, p.y] for p in predictor(self.img, rect).parts()]) for rect in rects]
        return result


class Face:
    def __init__(self, img, face_landmark):
        # self.landmark: A list of landmarks (68 points) for THIS FACE (Only one face)
        # self.organs: A list of Organ objects of this face
        self.landmark = face_landmark
        self.organs = [Organ(np.array(self.landmark[points]), name, img) for name, points in
                       zip(constants.ORGAN_NAME, constants.ORGAN_INDEX)]
        self._add_forehead_organ(img)
        self.img_shape = img.shape

    def _add_forehead_organ(self, img):
        # 画椭圆
        radius = (np.linalg.norm(self.landmark[0] - self.landmark[16]) / 2).astype('int32')
        center_abs = tuple(((self.landmark[0] + self.landmark[16]) / 2).astype('int32'))

        angle = np.arctan((lambda l: l[1] / l[0])(np.array(self.landmark[16] - self.landmark[0])[0])).astype(
            'int32')
        forehead_mask = np.zeros(img.shape[:2], dtype=np.float64)
        cv2.ellipse(forehead_mask, center=tuple(np.array(center_abs).flatten()),
                    axes=(radius, radius), angle=angle, startAngle=180, endAngle=360, color=1, thickness=-1)

        # 剔除与五官重合部分
        # forehead_mask[self.mask[:, :, 0] > 0] = 0

        # 根据鼻子的肤色判断真正的额头面积
        nose_index = 2
        nose_mask = organmaskMgr.get_organ_whole_mask(self.organs[nose_index].landmark, img.shape)

        index_bool = []
        for ch in range(3):
            mean = np.mean(img[:, :, ch][nose_mask[:, :, ch] > 0])
            std = np.std(img[:, :, ch][nose_mask[:, :, ch] > 0])
            up, down = mean + 0.5 * std, mean - 0.5 * std
            index_bool.append((img[:, :, ch] < down) | (img[:, :, ch] > up))
        index_zero = ((forehead_mask > 0) & index_bool[0] & index_bool[1] & index_bool[2])
        forehead_mask[index_zero] = 0
        index_abs = np.array(np.where(forehead_mask > 0)[::-1]).transpose()
        landmark = cv2.convexHull(index_abs).squeeze()

        self.organs.append(Organ(landmark, "Forehead", img.shape))

    def get_facemask(self):
        organ_mask = np.zeros(self.img_shape, dtype=np.uint8)
        for organ_i in constants.ORGAN_DETECT_LIST:
            organ = self.organs[organ_i]
            tmp_mask = organmaskMgr.get_organ_whole_mask(organ.landmark, self.img_shape)
            organ_mask = np.clip(tmp_mask + organ_mask, 0, 1)
        forehead = self.organs[-1]

        forehead_mask = organmaskMgr.get_organ_whole_mask(forehead.landmark, self.img_shape)
        face_mask = organmaskMgr.get_organ_whole_mask(self.organs[0].landmark, self.img_shape)
        whole_face_mask = np.clip(forehead_mask + face_mask, 0, 1)
        whole_face_mask = whole_face_mask - organ_mask
        return whole_face_mask


class Organ:
    def __init__(self, landmark, name, img_shape):
        self.landmark = landmark
        self.name = name
        self.img_shape = img_shape
