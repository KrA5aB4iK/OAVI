import numpy as np


def simple_bin(img: np.array, thres: int = 40):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = semitone(img)

    res_img = np.empty_like(img)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            res_img[x, y] = 255 if img[x, y] > thres else 0
    return res_img


def semitone(img):
    return (0.3 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.11 *
            img[:, :, 2]).astype(np.uint8)
