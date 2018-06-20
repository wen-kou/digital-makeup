import unittest
import cv2
import numpy as np

from factial_attributes.facial_landmarks import facial_landmarks_detection_dlib
from factial_attributes.facial_segment import face_segmentation


class MyTestCase(unittest.TestCase):
    def test_get_triangle_mesh(self):
        image_path = '../assets/examples/after-makeup3.jpeg'
        image = cv2.imread(image_path)

        landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(image, add_points=True)
        landmarks_coords, triangle_indices, triangles = face_segmentation.get_triangle_landmarks(landmarks)

        for triangle in triangles:
            cv2.polylines(image, [np.asarray(triangle)],True, (0,255,255))
        cv2.imwrite('test1.jpg', image)
        pass

    def test_calc_triangle_affine_transformation(self):
        sources = [(2,2), (3,3), (0,1)]
        targets = [(2,2), (3,3), (0,1)]
        affine_h = face_segmentation.calc_triangle_affine_transformation(sources, targets)

        self.assertEquals(np.eye(3).tolist(), affine_h.tolist())

        # TODO: more cases should be added

    def test_get_ref_pixels_by_affine_transformation(self):
        sources = np.array([[2, 3, 3], [3, 0, 1]], dtype=np.float32)
        gt = np.array([[0,1,1], [1,-2,-1]])
        affine_h = 0.5 * np.eye(3)
        affine_h[0,2] = -1
        affine_h[1,2] = -1

        res = face_segmentation.pixel_transfer(affine_h, sources)

        self.assertEquals(gt.tolist(), res.tolist())

    def test_get_facial_segments(self):
        image_path = '../assets/examples/after-makeup3.jpeg'
        image = cv2.imread(image_path)

        new_image = np.zeros(image.shape)
        landmarks = facial_landmarks_detection_dlib.get_facial_landmarks(image, add_points=True)

        chin_landmarks = landmarks[0:17]
        coords = face_segmentation.find_face_region(chin_landmarks,image.shape )
        new_image[coords[0], coords[1], :] = image[coords[0], coords[1], :]
        cv2.imwrite('test1.jpg', new_image)
        pass

if __name__ == '__main__':
    unittest.main()
