# July/2018 LzCai

from filters.facemask import facemaskMgr
from filters.foundation import whitening
from filters.foundation import blurring
from filters.facemask import classes
from filters.facemask import organmaskMgr
from filters import render_util
import numpy as np
import cv2


def filter(img: np.array):
    # 1. Initialize Image objects
    img_object = classes.Image(img)
    if img_object.faceList is None:
        return img

    # 2. De_acne for all faces on the image
    acne_mask = organmaskMgr.get_acne_mask(img_object.faceList, img_object.img)
    face_mask = facemaskMgr.get_img_face_mask(img_object)
    acne_mask = acne_mask * face_mask
    tmp_mask = acne_mask[:,:,0].astype(np.uint8)
    img_removed_acne = cv2.inpaint(img, tmp_mask, 3, cv2.INPAINT_TELEA)

    # 3. Whiten the de_acned image
    white_result = whitening.whiten_skin(img_removed_acne)

    # 4. Blurring the texture of all faces of the de_acned image
    blur_result = blurring.skin_retouch(white_result, face_mask, 'Gaussian', strength=100)

    # 5. Blend the whitened result and the blurring result
    blend_result = render_util.image_blending(blur_result, white_result, 0.8)

    # 6. Adjust the color temperature
    blend_result = whitening.color_balance(blend_result, 9000)

    return blend_result
