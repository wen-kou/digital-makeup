import unittest
import cv2

from utils import face_process


class MyTestCase(unittest.TestCase):
    def test_get_face_rect(self):
        image_path = '../assets/examples/after-makeup3.jpeg'
        image = cv2.imread(image_path)
        mask, face_template = face_process.get_new_face_mask(image)

        cv2.imwrite('face.jpg', face_template)
        # cv2.waitKey(100)
        pass


if __name__ == '__main__':
    unittest.main()
