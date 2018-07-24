# NOT IN USE... FOR BACK UP ONLY


import cv2
import numpy as np
import dlib


def getChecklist():
    organs_name = ['jaw', 'mouth', 'nose', 'left eye', 'right eye', 'left brow', 'right brow']
    organs_points = [list(range(0, 17)), list(range(48, 61)), list(range(27, 35)), list(range(42, 48)),
                     list(range(36, 42)), list(range(22, 27)), list(range(17, 22))]

    point_checklist = [""] * 68
    for tmp in zip(organs_name, organs_points):
        for i in tmp[1]:
            point_checklist[i] = tmp[0]
    return organs_name, organs_points, point_checklist


def getAllFaceLandmark(img):
    predictor_path = '../../resources/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    rects = detector(img, 1)
    result = [np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()]) for rect in rects]
    return result


# result: ALL faces in the image


def getLocation(img, landmarks):
    ys, xs = landmarks[:, 1], landmarks[:, 0]
    top, bottom, left, right = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
    shape = (int(bottom - top), int(right - left))
    size = shape[0] * shape[1] * 3
    move = int(np.sqrt(size / 3) / 20)
    organLocation = [top, bottom, left, right, shape, size, move]
    return organLocation


def _patchOrgan(img, landmarks):
    organLocation = getLocation(img, landmarks)
    img_shape = np.shape(img)[0:2]
    top, bottom, left, right, shape, size, move = organLocation[:]
    patch = img[np.max([top - move, 0]):np.min([bottom + move, img_shape[0]]),
            np.max([left - move, 0]):np.min([right + move, img_shape[1]])]
    return patch, organLocation


def drawLandmark(img, landmark, point_checklist, name, size=15, haveNum=True, haveWord=True):
    pointed_img = img.copy()
    for tmp in enumerate(np.array(landmark)):
        i = tmp[0]
        img = cv2.circle(pointed_img, tuple(np.array(landmark)[i]), size, color=(0, 0, 255),
                         thickness=max(0, int(size / 5)))
        if haveNum and haveWord:
            text = str(i) + ' ' + point_checklist[i]
        elif haveNum and not haveWord:
            text = str(i)
        elif not haveNum and haveWord:
            text = point_checklist[i]
        else:
            text = ""
        tmp_point = tuple(np.array(landmark)[i] + 20)
        pointed_img = cv2.putText(pointed_img, text, tmp_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1)

    cv2.imwrite("face5-" + name + "-drawLM.jpg", pointed_img)


def getKsize(img, size):
    tmprate = 80
    tmpsize = max([int(np.sqrt(size / 3) / tmprate), 1])
    tmpsize = (tmpsize if tmpsize % 2 == 1 else tmpsize + 1)
    ksize = (tmpsize, tmpsize)
    return ksize


def drawOrganMask(img, landmark, organ_patch, organ_location):
    top, bottom, left, right, shape, size, move = organ_location[:]
    ksize = getKsize(organ_patch, size)

    landmark_re = landmark.copy()
    landmark_re[:, 1] -= np.max([top - move, 0])
    landmark_re[:, 0] -= np.max([left - move, 0])
    mask = np.zeros(organ_patch.shape[:2], dtype=np.float64)

    points = cv2.convexHull(landmark_re)
    mask = cv2.fillConvexPoly(mask, points, color=1).astype(np.uint8)

    mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
    # mask = (cv2.GaussianBlur(mask, ksize, 0) > 0) * 1.0
    # mask = cv2.GaussianBlur(mask, ksize, 0)[:]

    return mask


def drawOrganMaskOnWholeImg(wholeImg, organ_location, organ_mask, mask=None):
    if mask is None: mask = np.zeros(wholeImg.shape, dtype=np.float64)
    img_shape = np.shape(wholeImg)
    top, bottom, left, right, shape, size, move = organ_location[:]

    rowStart = np.max([top - move, 0])
    rowEnd = np.min([bottom + move, img_shape[0]])
    colStart = np.max([left - move, 0])
    colEnd = np.min([right + move, img_shape[1]])

    mask[rowStart:rowEnd, colStart:colEnd] += organ_mask

    return mask


def updateOrganMask(img, organ_element, wholeImg_mask=None):
    if wholeImg_mask is None: wholeImg_mask = np.zeros(img.shape, dtype=np.float64)
    organ_landmark = organ_element[0]
    organ_name = organ_element[1]
    organ_patch, organ_location = _patchOrgan(img, organ_landmark)
    organ_mask = drawOrganMask(img, organ_landmark, organ_patch, organ_location)
    updated_wholeImg_mask = drawOrganMaskOnWholeImg(img, organ_location, organ_mask, mask=wholeImg_mask)

    return updated_wholeImg_mask


def getOrganMask(img, organ_element):
    organ_landmark = organ_element[0]
    organ_name = organ_element[1]
    organ_patch, organ_location = _patchOrgan(img, organ_landmark)
    organ_mask = drawOrganMask(img, organ_landmark, organ_patch, organ_location)
    return organ_mask, organ_patch, organ_location


def getForeHeadLandmark(img, one_face_landmarks, organ_mask, nose_mask):
    # 画椭圆
    radius = (np.linalg.norm(one_face_landmarks[0] - one_face_landmarks[16]) / 2).astype('int32')
    center_abs = tuple(((one_face_landmarks[0] + one_face_landmarks[16]) / 2).astype('int32'))

    angle = np.arctan((lambda l: l[1] / l[0])(np.array(one_face_landmarks[16] - one_face_landmarks[0])[0])).astype(
        'int32')
    forehead_mask = np.zeros(organ_mask.shape[:2], dtype=np.float64)
    cv2.ellipse(forehead_mask, center=tuple(np.array(center_abs).flatten()),
                axes=(radius, radius), angle=angle, startAngle=180, endAngle=360, color=1, thickness=-1)

    # 剔除与五官重合部分
    forehead_mask[organ_mask[:, :, 0] > 0] = 0

    # 根据鼻子的肤色判断真正的额头面积
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

    return landmark


def getFaceMask(img):
    all_faces_landmarks = getAllFaceLandmark(img)
    organs_name, organs_points, point_checklist = getChecklist()
    if all_faces_landmarks == [] :
        print ("NOT FOUND")
        return np.zeros(img.shape, dtype=np.uint8)+1

    one_face_landmarks = all_faces_landmarks[0]
    # landmark: features on ONE person


    all_points = one_face_landmarks[0]
    all_organs_landmark = [[np.array(one_face_landmarks[points]), name] for name, points in
                           zip(organs_name, organs_points)]

    # Find Organ Mask
    organ_mask = np.zeros(img.shape, dtype=np.float64)
    for org_i in [1,3,4,5,6]:
    # for organ_element in all_organs_landmark[1:]:
        organ_element = all_organs_landmark[org_i]
        organ_mask = updateOrganMask(img, organ_element, organ_mask)

    # Find Face Mask
    nose_index = 2
    nose_element = all_organs_landmark[nose_index]
    nose_mask = updateOrganMask(img, nose_element)
    forehead_landmark = getForeHeadLandmark(img, one_face_landmarks, organ_mask, nose_mask)
    face_mask = np.zeros(img.shape, dtype=np.float64)
    forehead_element = [forehead_landmark, "forehead"]
    face_mask = updateOrganMask(img, forehead_element, face_mask)
    face_mask = updateOrganMask(img, all_organs_landmark[0], face_mask)
    face_mask = face_mask - organ_mask
    # face_mask = cv2.GaussianBlur(face_mask, (15, 15), 0)
    face_mask_binary = np.where(face_mask > 0, 1, 0)
    return face_mask_binary
