from filters.foundation import blurring
from filters.foundation import whitening
from filters.foundation.facemask import facemask
from utils import img_randering
import cv2
import os
import unittest
import time
import numpy as np
from unittest import TestCase


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


class TestFoundation(TestCase):
    def test_skinRetouch(self):
        import_path = '../assets/natural_look/'
        # tmp_list = listdir_nohidden(import_path)
        tmp_list = ["face6.jpeg"]
        # tmp_list = ["test_34756300_176754833015550_7307187577633636352_n.jpg"]
        for img_name in tmp_list:
            print(img_name)
            img_path = import_path + img_name
            img = cv2.imread(img_path)
            # img_skin = blurring.detectSkin(img)
            img_skin = facemask.get_img_face_mask(img)

            # Gaussian, Guided, Surface, Bilateral
            # for filter_name in ["Gaussian", "Guided", "Surface", "Bilateral"]:
            for filter_name in ["Gaussian"]:
                start = time.clock()
                white_result = whitening.whitenSkin(img)
                blur_result = blurring.skinRetouch(white_result, img_skin, filter_name, strength=80, dx=30)
                # white_result2 = whitening.whitenSkin(blur_result)
                blend_result = img_randering.imageBlending(blur_result, white_result, 0.7)
                blend_result = whitening.colorBalance(blend_result, 10000)

                end1 = time.clock()-start
                print("Foundation Time:", img_name, filter_name, end1)

                # export_path = './result-White_First-0629-4.4/' + img_name[:-4] + '/'
                export_path = './result-0703/'
                save_name = img_name[:-4] + filter_name + '-blurdx30-1.jpg'
                if not os.path.exists(export_path):
                    os.makedirs(export_path)
                cv2.imwrite(export_path + save_name, img_skin*255)
                end2 = time.clock()-start
                print("Save Time:", img_name, filter_name, end2-end1)
        pass


if __name__ == '__main__':
    unittest.main()
