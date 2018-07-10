import cv2
import os
import unittest
import time
import numpy as np
from unittest import TestCase
from filters.de_acne import acne_mask as am
from filters.foundation.facemask import facemask


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


class TestFoundation(TestCase):
    def test_skinRetouch(self):
        import_path = '../assets/natural_look/'
        tmp_list = listdir_nohidden(import_path)
        # tmp_list = ["face6.jpeg"]
        # tmp_list = ["test_34756300_176754833015550_7307187577633636352_n.jpg"]
        for img_name in tmp_list:
            print(img_name)
            img_path = import_path + img_name
            img = cv2.imread(img_path)

            cheek_mask = facemask.get_cheek_forehead_mask(img) # This is the acne mask: circle out the acne
            if len(cheek_mask.shape) == 3:
                cheek_mask = cheek_mask[:, :, 0]

            face_mask = facemask.get_img_face_mask(img)
            if len(face_mask.shape) == 3:
                face_mask = face_mask[:, :, 0]

            cheek_mask = cheek_mask * face_mask # remove the organ zone

            acne_mask = np.zeros(img.shape, dtype=np.uint8)
            for i in range(3):
                acne_mask[:, :, i] = cheek_mask


            export_path = './result-0710-1/'
            save_name = img_name[:-4] + '-cheek_mask.jpg'
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            cv2.imwrite(export_path + save_name, acne_mask)
        pass


if __name__ == '__main__':
    unittest.main()
