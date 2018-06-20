import cv2
import dlib
import numpy as np

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
    threds2 = whiteness[int(0*num)]

    return threds0, threds1, threds2


def get_new_face_mask(image):
    face = get_face_image(image)
    max_, threds1, threds2 = _whiteness_analysis(face)

    mask = np.ones((image.shape[0], image.shape[1]))

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = np.where(image_gray >= threds1)
    mask[rows, cols] = 0
    rows, cols = np.where(np.logical_and(image_gray >= threds2, image_gray < threds1))
    mask[rows, cols] = 1 - np.divide(image_gray[np.where(np.logical_and(image_gray >= threds2, image_gray < threds1))],max_)

    mask = np.tanh(mask * 2.5)
    mask = np.divide(mask, np.max(mask))
    # cv2.imshow('mask', np.asarray(255 * mask,dtype=np.uint8))
    # cv2.waitKey(200)

    scalar = np.repeat(mask, 3, axis=1)
    scalar = np.reshape(scalar, image.shape)
    res = np.multiply(scalar, image)

    return mask, res


