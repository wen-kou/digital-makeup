# filter.py
## Overview
1. Initialize Image objects
    1. Detect __all__ faces in the image
2. De_acne for __all__ faces on the image
    1. Find the "face" area (cheek and forehead), return it as a Mask object
    2. Find the acne in that area
    3. Inpaint the acne
    4. Return the blurred image
3. Whiten the de_acned image
    1. Decrease the saturation
    2. Decrease the contrast
    3. Increase the brightness
4. Blurring the texture of __all__ faces of the de_acned image
    1. Find the face mask of the face, return as a Mask object
    2. Separate the base and the texture of the whitened image
    3. Blur the texture
    4. Combine the blurred texture and the base
5. Blend the whitened result and the blurring result
6. Adjust the color temperature

# render_util.py
- __image_blending(origin, destin, percentage)__:
    - Linear probing of two image with certain ratio
    - Input:
        - _origin_: The original image
        - _destin_: The destination image
        - _percentage_: The percentage of transformation from the original image
    - Output:
        - Blended image.

- __linear_light(img_1, img_2)__:
    - Linear-light blending of two image

- __hard_light(img_1, img_2)__:
    - Hard-light blending of two image

- __tri_color(gray)__:
    - Tri-color (0, 128, 200) clustering of a gray image (1-channel)
    - Input:
        - _gray_: 1 channel gray style image
    - Output:
        - _result_: 1 channel gray style image with three colors (black, white, gray)

 © July/2018 LzCai