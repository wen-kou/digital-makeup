# Foundation Algorithm
## 1. Blurring

- __skin_retouch(img, img_skin, func="Gaussian", strength=50, dx=10)__:
    - Separate the "base" layer and the "detail" layer of the image.
    - Blur the "detail" layer
    - Mixed the blurred "detail" layer with the "base" layer using linear-light blending
    - Adjust the strength and opacity and linear probe the result on the original
    - Input:
        - _img_: The original image
        - _img_skin_: The mask of the areas that considered as faces
        - _func_: Blurring function to be used, options are:
            - _"Gaussian"_: Gaussian Filter
            - _"Guided"_: Guided Filter
            - _"Surface"_: Surface Blurring
            - _"Bilateral"_: Bilateral Filter
        - _strength_: [0, 100], Percentile that the original image to be change into the filtered image:
            - 0: Original Image
            - 100: Fully Filtered Image
        - _dx_: The window size parameter for extracting the base:
            - Larger window size -> More blurry "base" layer -> Less detail preserved in the "base" layer -> More blurry final result.
    - Output:
        - _blended_: The blended image.
        
## 2. Whitening
- __whiten_skin(img, s_rate = 0.22, v_rate = 0.17)__:
    - Transform the BGR image to HSV image
    - Decrease the Saturation by s_rate
    - Increase the Value (brightness) by v_rate
    - Input:
        - _img_: The original image
        - _s_rate_: [0, 1], Ratio of decreasing the Saturation
        - _v_rate_: [0, 1], Ratio of increasing the Value
    - Output:
        - _img_bgr_: The result whitened image in BGR format.
        
- ___kelvin_to_BGR(k_temp)__:
    - Input the Kelvin Temperature value, return the BGR value of "white" color of that color temperature.
    - Input:
        - _k_temp_: Kelvin Temperature value, Usually [700, 10000]
    - Output:
        - _(b,g,r)_: The BGR value of the "white" color under that color temperature. ((255,255,255) <=> 6500K)

- __color_balance(img, k_temp)__:
    - Input the image and the target color temperature, assume the image is of 6500K, return the adjusted image
    - Input:
        - _img_: The original image, treated as under 6500K
        - _k_temp_: The desired Kelvin Temperature value
    - Output:
        - Adjusted image.
 
 © July/2018 LzCai