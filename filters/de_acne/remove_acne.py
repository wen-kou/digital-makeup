import cv2
import numpy as np
from filters.de_acne import acne_mask
from filters.foundation.facemask import facemask


def remove_acne(img):
    cheek_mask = facemask.get_cheek_forehead_acne_mask(img)  # This is the acne mask: circle out the acne
    if len(cheek_mask.shape) == 3:
        cheek_mask = cheek_mask[:, :, 0]

    face_mask = facemask.get_img_face_mask(img)
    if len(face_mask.shape) == 3:
        face_mask = face_mask[:, :, 0]

    cheek_mask = cheek_mask * face_mask  # remove the organ zone

    acne_mask = np.zeros(img.shape, dtype=np.uint8)
    for i in range(3):
        acne_mask[:, :, i] = cheek_mask

    img_removed_acne = cv2.inpaint(img, acne_mask[:, :, 0], 3, cv2.INPAINT_TELEA)

    return img_removed_acne
