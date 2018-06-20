import unittest
import cv2

from utils import face_process
from factial_attributes.facial_landmarks import facial_landmarks_detection_dlib
from factial_attributes.facial_segment import face_segmentation


class MyTestCase(unittest.TestCase):
    def test_get_facial_landmarks_dlib(self):
        image_path = '../assets/examples/after-makeup3.jpeg'
        image = cv2.imread(image_path)
        rect = face_process.get_face_rect(image)
        landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(image, rect)
        landmark_image = face_segmentation.draw_landmark(image, landmarks)
        cv2.imwrite('facial_landmarks.jpg', landmark_image)
        pass


if __name__ == '__main__':
    unittest.main()
