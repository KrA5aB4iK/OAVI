import numpy as np
from PIL import Image
from os import path


def calculate_profile(img: np.array, axis: int) -> np.array:
    return np.sum(img, axis=1 - axis)


def cut_black(img: np.array, profile: np.array, axis: int) -> np.array:
    start = profile.nonzero()[0][0]
    end = profile.nonzero()[0][-1] + 1

    if axis == 0:
        return img[start:end, :], profile[start:end]
    elif axis == 1:
        return img[:, start:end], profile[start:end]


def image_to_np_array(image_name: str) -> np.array:
    img_src = Image.open(path.join('results', image_name)).convert('L')
    return np.array(img_src)
