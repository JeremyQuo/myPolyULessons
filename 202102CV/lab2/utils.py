from PIL import Image, ImageDraw  # pillow package
import numpy as np

def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr


def save_array_as_img(arr, file):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min) / (max - min) * 255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)


def show_array_as_img(arr, title=None):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min) / (max - min) * 255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.show(title=title)


def rgb2gray(arr):
    R = arr[:, :, 0]  # red channel
    G = arr[:, :, 1]  # green channel
    B = arr[:, :, 2]  # blue channel
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return gray