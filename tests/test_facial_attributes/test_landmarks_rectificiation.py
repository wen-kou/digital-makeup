import unittest
import numpy as np
import cv2

from factial_attributes.facial_landmarks import landmarks_rectification
from factial_attributes.facial_landmarks import facial_landmarks_detection_dlib
from factial_attributes.facial_segment import face_segmentation
from utils import face_process

class MyTestCase(unittest.TestCase):
    def test_calc_gradient(self):
        image_path = '../assets/targets/target_11.jpeg'
        image = cv2.imread(image_path)
        mask, mask_image = face_process.get_new_face_mask(image)
        mask = mask*255
        image_gradient = landmarks_rectification._calc_gradient(mask)
        image_gradient = image_gradient/image_gradient.max()
        cv2.imwrite('test_gradient.jpg', np.asarray(255*image_gradient, dtype=np.uint8))
        val = np.mean(image_gradient)
        self.assertGreater(1, val)

    def test__landmark_rectify(self):
        image_path = '../assets/examples/example_8.jpeg'
        image = cv2.imread(image_path)
        landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(image)
        face_contour_landmarks = landmarks[0:17]
        image_size = tuple([image.shape[0], image.shape[1]])
        facial_region_index = face_segmentation.get_facial_indices(landmarks, image_size)
        gmm_color = face_segmentation.gmm_color_model(image, facial_region_index)
        rectified_landmark = landmarks_rectification._landmark_rectify(image,
                                                                       face_contour_landmarks,
                                                                       gmm_color,
                                                                       search_dir='horizontal',
                                                                       window_size=25)

        image = face_segmentation.draw_landmark(image, landmarks)
        image = face_segmentation.draw_landmark(image, rectified_landmark, color=(0, 0, 255))
        cv2.imwrite('test_rectify.jpg', image)

    def test_landmarks_rectification(self):
        image_path = '../assets/targets/target_10.jpeg'
        image = cv2.imread(image_path)
        landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(image)
        face_contour_landmarks = landmarks[0:17]
        image_size = tuple([image.shape[0], image.shape[1]])
        facial_region_index = face_segmentation.get_facial_indices(landmarks, image_size)
        gmm_color = face_segmentation.gmm_color_model(image, facial_region_index)
        rectified_landmarks = landmarks_rectification.landmarks_rectify(image,
                                                                        face_contour_landmarks,
                                                                        gmm_color,
                                                                        resolution_list=tuple([512]))
        image = face_segmentation.draw_landmark(image, face_contour_landmarks)
        image = face_segmentation.draw_landmark(image, rectified_landmarks, (0, 0, 255))
        cv2.imwrite('test_rectify.jpg', image)

if __name__ == '__main__':
    unittest.main()
