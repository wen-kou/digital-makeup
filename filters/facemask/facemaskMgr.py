# July/2018 LzCai

import numpy as np


def get_img_face_mask(img_object):  # Main function
    whole_img_mask = np.zeros(img_object.img.shape, dtype=np.uint8)
    faces = img_object.faceList
    if faces is None or faces == []:
        print("!!!!NOT FOUND!!!!")
        return whole_img_mask

    for face in faces:
        mask = face.get_facemask()
        whole_img_mask = np.clip(mask+whole_img_mask, 0, 1)

    return whole_img_mask

