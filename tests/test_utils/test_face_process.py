import unittest
import cv2
import numpy as np
import time

from utils import face_process
from facial_attributes.facial_landmarks import facial_landmarks_detection_dlib
from facial_attributes.facial_segment import face_segmentation


class MyTestCase(unittest.TestCase):
    def test_get_face_rect(self):
        image_path = '../assets/examples/example_0.jpeg'
        image = cv2.imread(image_path)
        start = time.time()
        mask, face_template = face_process.get_new_face_mask(image, segment=True)
        print('Process time is {}s'.format(time.time() - start))
        mask = 255*mask

        cv2.imwrite('test_face_matting.jpg', np.asarray(mask, dtype=np.uint8))
        pass

    def test_transfer_image(self):
        image_path = '../assets/examples/example_8.jpeg'
        image = cv2.imread(image_path)
        landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(image)
        face_rect = face_process.select_face_rect(landmarks)
        face_image = image[face_rect['left_top'][1]:face_rect['bottom_right'][1],
                           face_rect['left_top'][0]:face_rect['bottom_right'][0]]
        cv2.imwrite('test.jpg', face_image)
        new_image, t = face_process.calc_image_transform(face_rect, image, resolution=512)
        cv2.imwrite('test1.jpg', new_image)
        new_landmarks = []
        for landmark in landmarks:
            new_landmarks.append(face_process.calc_transform_pts(landmark, t))

        new_image = face_segmentation.draw_landmark(new_image, new_landmarks)

        cv2.imwrite('test2.jpg', new_image)
        self.assertEquals(True,True)

    def test_calc_transform_mat(self):
        image_path = '../assets/examples/after-makeup1.jpeg'
        image = cv2.imread(image_path)
        landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(image)
        height, width = image.shape[0], image.shape[1]
        origin_resolution = min(height, width)
        origin_center = tuple([origin_resolution /2, origin_resolution/2])
        image = image[0: origin_resolution, 0:origin_resolution]
        new_resolution = 256
        image = cv2.resize(image, dsize=(int(new_resolution),int(new_resolution)))
        new_center = tuple([new_resolution / 2, new_resolution/ 2])

        transform = face_process._calc_transfer_mat(origin_center, new_center, origin_resolution, new_resolution)

        new_landmarks = []
        for landmark in landmarks:
            new_landmarks.append(face_process.calc_transform_pts(landmark, transform))

        new_image = face_segmentation.draw_landmark(image, new_landmarks)
        cv2.imwrite('test1.jpg', new_image)


if __name__ == '__main__':
    unittest.main()
