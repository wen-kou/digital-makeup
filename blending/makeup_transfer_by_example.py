import numpy as np
import cv2
import copy

from multiprocessing.pool import ThreadPool

global pool
pool = ThreadPool(2048)


def calc_cosmetic_map(before_makeup_img, after_makeup_img):
    before_makeup_gray = cv2.cvtColor(before_makeup_img, cv2.COLOR_BGR2GRAY)

    after_makeup_gray = cv2.cvtColor(after_makeup_img, cv2.COLOR_BGR2GRAY)

    cosmetic_map = np.divide((before_makeup_gray + np.finfo(float).eps), (after_makeup_gray + np.finfo(float).eps))
    return cosmetic_map


def makeup_transfer(target_image,
                    target_pixels,
                    makeup_img,
                    transfer_pixels,
                    cosmetic_map=None, k=0.5, alpha=0.4):
    # cosmetics = list(map(lambda pix : cosmetic_map[pix[1], pix[0]], transfer_pixels))
    if cosmetic_map is None:
        cosmetic_map = np.ones((target_image.shape[0], target_image.shape[1]))
    cosmetics = cosmetic_map[np.asarray(transfer_pixels[0], dtype=int), np.asarray(transfer_pixels[1], dtype=int)]
    source_colors = target_image[np.asarray(target_pixels[0], dtype=int), np.asarray(target_pixels[1], dtype=int), :]
    # source_colors = list(map(lambda pix : target_image[pix[1], pix[0],:], transfer_pixels))
    scale = k * np.asarray(cosmetics) - k + 1
    scale = np.reshape(np.repeat(scale, 3), (transfer_pixels[0].shape[0], 3))
    target_color = np.multiply(source_colors, scale)
    target_color[np.where(target_color > 255)] = 255
    target_color[np.where(target_color < 0)] = 0

    makeup_colors = makeup_img[np.asarray(transfer_pixels[0], dtype=int), np.asarray(transfer_pixels[1], dtype=int)]
    target_color = (1 - alpha) * target_color + alpha * makeup_colors

    return np.asarray(target_color, dtype=np.uint8)


def color_blending(target_image_origin,
                   target_pixels,
                   ref_image,
                   transfer_pixels,
                   simple_mode=True, cos_map=None, alpha_map=None, k=0.5, alpha=0.6):
    if cos_map is None:
        cos_map = np.ones((ref_image.shape[0], ref_image.shape[1]))

    ref_image_size = ref_image.shape
    if alpha_map is not None:
        alpha_map = np.repeat(alpha_map, 3, axis=1)
        alpha_map = np.reshape(alpha_map, ref_image_size)
        ref_image = np.multiply(ref_image, alpha_map)
    target_image = copy.copy(target_image_origin)
    target_pixels = np.asarray(target_pixels, dtype=int).transpose()
    origin_color_values = target_image[target_pixels[0], target_pixels[1]]
    transfer_pixels = np.asarray(transfer_pixels)
    zip_inputs = zip(origin_color_values, transfer_pixels)

    if simple_mode is True:
        transfer_pixels = np.asarray(np.around(transfer_pixels), dtype=int).transpose()
        transfer_pixels[np.where(transfer_pixels < 0)] = int(0)
        rows = np.where(transfer_pixels[0,:] >= ref_image_size[0])[0]
        if rows.shape[0] > 0:
            cols = np.zeros(rows.shape, dtype=int)
            transfer_pixels[rows,cols] = ref_image_size[0] - 1
        rows = np.where(transfer_pixels[1,:] >= ref_image_size[1])[0]
        if rows.shape[0] > 0:
            cols = np.ones(rows.shape, dtype=int)
            transfer_pixels[rows,cols] = ref_image_size[1] - 1
        transferred_color_values = alpha * ref_image[transfer_pixels[0], transfer_pixels[1], :]

        if alpha_map is not None:
            alpha_values = alpha_map[transfer_pixels[0], transfer_pixels[1], :]
            alpha_values = 1 - alpha * alpha_values
            transferred_color_values = transferred_color_values + \
                                       np.multiply(alpha_values,target_image[target_pixels[0], target_pixels[1], :])
        else:
            transferred_color_values = transferred_color_values + \
                                       (1 - alpha) * target_image[target_pixels[0], target_pixels[1], :]
    else:
        transferred_color_values = pool.map(lambda zip_input:
                                            _color_blending(zip_input[0], zip_input[1], ref_image,
                                                            ref_image_size, cos_map, k=k, alpha=alpha),
                                            zip_inputs)

    target_image[target_pixels[0], target_pixels[1], :] = transferred_color_values
    return np.asarray(target_image, dtype=np.uint8)


def _color_blending(original_color_value, transfer_pixel, ref_image, ref_image_size, cosmetic_map, k=0.5,
                    alpha=0.5):
    import math
    transfer_pixel_y, transfer_pixels_x = transfer_pixel[0], transfer_pixel[1]
    transfer_pixel_y_decimal, transfer_pixel_y_int = math.modf(transfer_pixel_y)
    transfer_pixels_x_decimal, transfer_pixels_x_int = math.modf(transfer_pixels_x)
    transfer_pixel_y_int = int(transfer_pixel_y_int)
    transfer_pixels_x_int = int(transfer_pixels_x_int)
    coeff = 0.0
    cosmetic_value = 0.0
    blended_color_value = np.zeros((1, 3))
    if (transfer_pixel_y_int >= 0) & (transfer_pixel_y_int < ref_image_size[0]) & \
            (transfer_pixels_x_int >= 0) & (transfer_pixels_x_int < ref_image_size[1]):
        tmp = math.sqrt(math.pow(transfer_pixels_x_decimal, 2) + math.pow(transfer_pixel_y_decimal, 2)) + np.finfo(
            float).eps
        tmp = 1.0 / tmp
        coeff = coeff + tmp
        cosmetic_value = cosmetic_value + tmp * cosmetic_map[transfer_pixel_y_int, transfer_pixels_x_int]
        blended_color_value = blended_color_value + tmp * ref_image[transfer_pixel_y_int, transfer_pixels_x_int]

    if (transfer_pixel_y_int + 1 >= 0) & (transfer_pixel_y_int + 1 < ref_image_size[0]) & \
            (transfer_pixels_x_int >= 0) & (transfer_pixels_x_int < ref_image_size[1]):
        tmp = math.sqrt(math.pow(transfer_pixels_x_decimal, 2) + math.pow(transfer_pixel_y_decimal + 1, 2)) + np.finfo(
            float).eps
        tmp = 1.0 / tmp
        coeff = coeff + tmp
        cosmetic_value = cosmetic_value + tmp * cosmetic_map[transfer_pixel_y_int + 1, transfer_pixels_x_int]
        blended_color_value = blended_color_value + tmp * ref_image[transfer_pixel_y_int + 1, transfer_pixels_x_int]

    if (transfer_pixel_y_int + 1 >= 0) & (transfer_pixel_y_int + 1 < ref_image_size[0]) & \
            (transfer_pixels_x_int + 1 >= 0) & (transfer_pixels_x_int + 1 < ref_image_size[1]):
        tmp = math.sqrt(
            math.pow(1 - transfer_pixels_x_decimal, 2) + math.pow(1.0 - transfer_pixel_y_decimal, 2)) + np.finfo(
            float).eps
        tmp = 1.0 / tmp
        coeff = coeff + tmp
        cosmetic_value = cosmetic_value + tmp * cosmetic_map[transfer_pixel_y_int + 1, transfer_pixels_x_int + 1]
        blended_color_value = blended_color_value + tmp * ref_image[transfer_pixel_y_int + 1, transfer_pixels_x_int + 1]

    if (transfer_pixel_y_int >= 0) & (transfer_pixel_y_int < ref_image_size[0]) & \
            (transfer_pixels_x_int + 1 >= 0) & (transfer_pixels_x_int + 1 < ref_image_size[1]):
        tmp = math.sqrt(math.pow(1 - transfer_pixels_x_decimal, 2) + math.pow(transfer_pixel_y_decimal, 2)) + np.finfo(
            float).eps
        tmp = 1.0 / tmp
        coeff = coeff + tmp
        cosmetic_value = cosmetic_value + tmp * cosmetic_map[transfer_pixel_y_int + 1, transfer_pixels_x_int + 1]
        blended_color_value = blended_color_value + tmp * ref_image[
            transfer_pixel_y_int, transfer_pixels_x_int + 1]
    cosmetic_value = cosmetic_value / coeff
    blended_color_value = blended_color_value / coeff

    cosmetic_value = k * (cosmetic_value - 1) + 1
    res = cosmetic_value * original_color_value
    res = (1 - alpha) * res + alpha * blended_color_value

    return res


