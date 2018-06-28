import unittest
import cv2

from blending import makeup_transfer_by_example
from factial_attributes.facial_segment import face_segmentation
from factial_attributes.facial_landmarks import facial_landmarks_detection_dlib


class MyTestCase(unittest.TestCase):
    def test_color_blending(self):
        example_path = '../assets/targets/target_12.jpg'
        example_image = cv2.imread(example_path)
        example_landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(example_image,add_points=True)
        target_path = '../assets/targets/target_9.jpg'
        target = cv2.imread(target_path)
        target_landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(target, add_points=True)

        _, target_triangle_indices, target_triangle_meshes = \
            face_segmentation.get_triangle_landmarks(target_landmarks)

        example_triangle_meshes = face_segmentation.get_triangle_mesh(example_landmarks,
                                                                      target_triangle_indices)
        target_pts, example_pts = face_segmentation.get_pixels_warp(target_triangle_meshes,
                                                                    target.shape, example_triangle_meshes)

        fusion_image = makeup_transfer_by_example.color_blending(target, target_pts, example_image, example_pts)
        cv2.imwrite('test_blending_result.jpg', fusion_image)

        pass


if __name__ == '__main__':
    unittest.main()
