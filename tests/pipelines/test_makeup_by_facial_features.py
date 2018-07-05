import unittest
import cv2
import numpy as np

from pipelines import makeup_by_facial_features
from facial_attributes.facial_segment import face_segmentation
from facial_attributes.facial_landmarks import facial_landmarks_detection_dlib

class MyTestCase(unittest.TestCase):
    def test_facial_features(self):
        image_path = '../assets/targets/target_12.jpg'
        image = cv2.imread(image_path)
        landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(image)
        _, triangle, triangle_mesh = face_segmentation.get_triangle_landmarks(landmarks)

        triangle, triangle_mesh = makeup_by_facial_features.get_triangle(landmarks)
        image = face_segmentation.draw_landmark(image, landmarks)

        for triangle in triangle_mesh:
            cv2.polylines(image, [np.asarray(triangle)], True, (0, 255, 255))
        cv2.imwrite('facial_features.jpg', image)
        pass

    def test_makeup_by_facial_features(self):
        example_path = '../assets/examples/example_8.jpeg'
        example = cv2.imread(example_path)

        target_path = '../assets/targets/target_8.jpg'
        target = cv2.imread(target_path)

        res = makeup_by_facial_features.makeup_by_facial_feature(target, example)

        cv2.imwrite('facial_feature_makeup.jpg', res)
        pass


if __name__ == '__main__':
    unittest.main()
