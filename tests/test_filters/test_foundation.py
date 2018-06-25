from filters.foundation import blurring
from filters.foundation import whitening
from utils import img_randering
import cv2
import os
import unittest
import time
from unittest import TestCase


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


class TestFoundation(TestCase):
    def test_skinRetouch(self):
        import_path = '../assets/natural_look/'
        tmp_list = listdir_nohidden(import_path)
        # tmp_list = ["test_face3.jpg"]
        for img_name in tmp_list:
            img_path = import_path + img_name
            img = cv2.imread(img_path)
            img_skin = blurring.detectSkin(img)

            # Gaussian, Guided, Surface, Bilateral
            for filter_name in ["Gaussian", "Guided", "Surface", "Bilateral"]:
                start = time.clock()
                blur_result = blurring.skinRetouch(img, img_skin, filter_name, strength=100)
                white_result = whitening.whitenSkin(blur_result, img_skin)
                blend_result = img_randering.imageBlending(blur_result, white_result, 0.8)

                end1 = time.clock()-start
                print("Foundation Time:", img_name, filter_name, end1)

                export_path = './result/' + img_name[:-4] + '/'
                save_name = filter_name + '-' + img_name
                if not os.path.exists(export_path):
                    os.makedirs(export_path)
                cv2.imwrite(export_path + save_name, blend_result)

                end2 = time.clock()-start
                print("Save Time:", img_name, filter_name, end2-end1)
        pass


if __name__ == '__main__':
    unittest.main()
