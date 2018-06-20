from filters.foundation import blurring
from filters.foundation import whitening
import cv2
# from tkinter import *

img = cv2.imread('face3.jpg')

# Gaussian, Guided, Surface, Bilateral
filterName = "Bilateral"
img_skin = blurring.detectSkin(img)
result = blurring.skinRetouch(img, img_skin, filterName, strength=100)
result1 = whitening.whitenSkin(result, img_skin)
result2 = blurring.imageBlending(result,result1,0.8)


result1_1 = whitening.whitenSkin(img, img_skin)
result1_2 = blurring.skinRetouch(result1_1, img_skin, filterName, strength=100)
result1_3 = blurring.imageBlending(result1_1,result1_2,0.8)

# cv2.imshow("skin", img_skin*255)
cv2.imshow("original", img)
cv2.imshow(filterName, result)
# cv2.imshow("lightened", result1)
cv2.imshow("result filter first", result2)
cv2.imshow("just white", result1_1)
cv2.imshow("result white first", result1_3)



# balancedimg = whitening.colorBalance(img)
# cv2.imshow("colorbalance", balancedimg)
cv2.waitKey()
# cv2.destroyAllWindows()
