import dlib

from utils import face_process

predictor_path = '../resources/shape_predictor_68_face_landmarks.dat'
face_shape_predictor = dlib.shape_predictor(predictor_path)


def get_facial_landmarks(image, add_points=False, rect=None):
    if rect is None:
        rect = face_process.get_face_rect(image)
    landmarks = face_shape_predictor(image, rect)
    landmarks_as_tuples = [tuple([landmarks.part(i).x, landmarks.part(i).y]) for i in range(68)]
    if add_points is True:
        cols = image.shape[1]
        intervals = 8
        intervals_dist = int(cols / 8)
        for i in range(intervals):
            landmarks_as_tuples.append(tuple([i*intervals_dist, 0]))
    return landmarks_as_tuples
