import cv2
import numpy as np
import dlib
from filters.facemask import constants
from filters.recycle.de_acne import acne_mask as am


class Face:
    def __init__(self, img, face_landmark):
        self.img = img
        self.landmark = face_landmark
        self.organs = [Organ(np.array(self.landmark[points]), name, self.img) for name, points in
                       zip(constants.ORGAN_NAME, constants.ORGAN_INDEX)]
        self.mask = None
        self.facialZone = None

    def update_mask(self, organ_index):
        if self.mask is None:
            self.mask = np.zeros(self.img.shape, dtype=np.float64)

        organ_mask = self.organs[organ_index].wholeMask
        self.mask = np.clip(self.mask + organ_mask, 0, 1)

    def get_forehead_landmark(self):
        # 画椭圆
        radius = (np.linalg.norm(self.landmark[0] - self.landmark[16]) / 2).astype('int32')
        center_abs = tuple(((self.landmark[0] + self.landmark[16]) / 2).astype('int32'))

        angle = np.arctan((lambda l: l[1] / l[0])(np.array(self.landmark[16] - self.landmark[0])[0])).astype(
            'int32')
        forehead_mask = np.zeros(self.mask.shape[:2], dtype=np.float64)
        cv2.ellipse(forehead_mask, center=tuple(np.array(center_abs).flatten()),
                    axes=(radius, radius), angle=angle, startAngle=180, endAngle=360, color=1, thickness=-1)

        # 剔除与五官重合部分
        forehead_mask[self.mask[:, :, 0] > 0] = 0

        # 根据鼻子的肤色判断真正的额头面积
        nose_index = 2
        nose_mask = self.organs[nose_index].wholeMask

        index_bool = []
        for ch in range(3):
            mean = np.mean(self.img[:, :, ch][nose_mask[:, :, ch] > 0])
            std = np.std(self.img[:, :, ch][nose_mask[:, :, ch] > 0])
            up, down = mean + 0.5 * std, mean - 0.5 * std
            index_bool.append((self.img[:, :, ch] < down) | (self.img[:, :, ch] > up))
        index_zero = ((forehead_mask > 0) & index_bool[0] & index_bool[1] & index_bool[2])
        forehead_mask[index_zero] = 0
        index_abs = np.array(np.where(forehead_mask > 0)[::-1]).transpose()
        landmark = cv2.convexHull(index_abs).squeeze()

        self.add_organ(landmark, "Forehead")

    def add_organ(self, landmark, name):
        self.organs.append(Organ(landmark, name, self.img))


class Organ:
    def __init__(self, landmark, name, img):
        self.landmark = landmark
        self.name = name
        self.imgShape = img.shape
        self.location = self._get_location()
        self.patchImg = self._get_patch(img)
        self.patchMask = self._get_patch_mask()
        self.wholeMask = self._get_whole_mask()

    def _get_location(self):
        ys, xs = self.landmark[:, 1], self.landmark[:, 0]
        top, bottom, left, right = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
        shape = (int(bottom - top), int(right - left))
        size = shape[0] * shape[1] * 3
        move = int(np.sqrt(size / 3) / 20)
        organ_location = [top, bottom, left, right, shape, size, move]
        return organ_location

    def _get_patch(self, img):
        img_shape = self.imgShape[0:2]
        top, bottom, left, right, shape, size, move = self.location[:]
        patch = img[np.max([top - move, 0]):np.min([bottom + move, img_shape[0]]),
                np.max([left - move, 0]):np.min([right + move, img_shape[1]])]
        return patch

    def _get_patch_mask(self):
        top, bottom, left, right, shape, size, move = self.location[:]
        ksize = constants.getKsize(self.patchImg, size)

        landmark_re = self.landmark.copy()
        landmark_re[:, 1] -= np.max([top - move, 0])
        landmark_re[:, 0] -= np.max([left - move, 0])
        mask = np.zeros(self.patchImg.shape[:2], dtype=np.float64)

        points = cv2.convexHull(landmark_re)
        mask = cv2.fillConvexPoly(mask, points, color=1).astype(np.uint8)

        mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
        # mask = (cv2.GaussianBlur(mask, ksize, 0) > 0) * 1.0
        # mask = cv2.GaussianBlur(mask, ksize, 0)[:]

        return mask

    def _get_whole_mask(self):
        mask = np.zeros(self.imgShape, dtype=np.float64)
        top, bottom, left, right, shape, size, move = self.location[:]

        rowStart = np.max([top - move, 0])
        rowEnd = np.min([bottom + move, self.imgShape[0]])
        colStart = np.max([left - move, 0])
        colEnd = np.min([right + move, self.imgShape[1]])

        mask[rowStart:rowEnd, colStart:colEnd] += self.patchMask

        return mask


def _get_faces(img):
    all_face_landmark = _get_all_face_landmark(img)
    if not all_face_landmark:  # If no face detected
        print("NOT FOUND")
        return [np.zeros(img.shape, dtype=np.uint8) + 1]

    faces = []
    for tmp_landmark in all_face_landmark:
        faces.append(Face(img, tmp_landmark))

    return faces


def get_img_face_mask(img):
    '''
     1. Scan through the image and find all the faces inside;
     2. For each face, obtain its landmark
     3. Draw the mask from the landmark:
        3.1. Draw the lower mask
        3.2. Dig out the unnecessary organs from the lower mask
        3.3. Draw the forehead mask
        3.4. Combine the forehead mask and the lower mask
    '''

    faces = _get_faces(img)

    whole_img_mask = np.zeros(img.shape, dtype=np.uint8)
    if not type(faces[0]) is Face:
        return whole_img_mask

    for face in faces:
        mask = get_one_face_mask(face)
        whole_img_mask = update_face_mask(mask, whole_img_mask)

    return whole_img_mask


def get_one_face_mask(face):
    assert type(face) is Face
    for organ_i in constants.ORGAN_DETECT_LIST:
        face.update_mask(organ_i)

    face.get_forehead_landmark()

    forehead = face.organs[-1]

    whole_face_mask = np.clip(forehead.wholeMask + face.organs[0].wholeMask, 0, 1)
    whole_face_mask = whole_face_mask - face.mask
    return whole_face_mask

# Copied
def _get_all_face_landmark(img):  # Will detect and return ALL faces in the image
    predictor_path = '../../resources/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    rects = detector(img, 1)
    result = [np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()]) for rect in rects]
    return result


def get_dig_out_organs(face):
    dig_out_organ_list = []
    for organ_i in constants.ORGAN_DETECT_LIST:
        dig_out_organ_list.append(face.organs[organ_i])
    return dig_out_organ_list


def update_face_mask(new_mask, whole_img_mask):
    return np.clip(new_mask + whole_img_mask, 0, 1)


def get_cheek_forehead_acne_mask(img):
    faces = _get_faces(img)
    if not type(faces[0]) is Face:
        return np.zeros(img.shape, dtype=np.uint8)
    masks = []

    for face in faces:
        # choice: LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD
        for choice in [0, 1, 2]:
        # for choice in [1]:
            if choice == constants.LEFT_CHEEK:
                partial_landmark = face.landmark[constants.LEFT_CHEEK_POINTS]

            elif choice == constants.RIGHT_CHEEK:
                partial_landmark = face.landmark[constants.RIGHT_CHEEK_POINTS]

            else:
                face.mask = np.zeros(face.img.shape, dtype=np.uint8)
                face.get_forehead_landmark()
                partial_landmark = face.organs[-1].landmark

            mask = np.zeros(img.shape[:-1], dtype=np.uint8)
            points = np.array([[i] for i in partial_landmark.tolist()])
            mask = cv2.fillConvexPoly(mask, points, color=1).astype(np.uint8)
            mask = np.array([mask, mask, mask]).transpose((1, 2, 0))

            mask = am.get_acne_mask_on_patch(img, mask)
            masks.append(mask)

    return np.clip(sum(masks), 0, 1)

