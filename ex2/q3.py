import cv2
import numpy as np

from src.cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeHDR, read_colorchecker_gm, xyY_to_XYZ


def tone_map(image_illumination: np.ndarray, N: int, K: float=0.15, B: float=0.95, eps: float=1e7):
    """
    Perform the tone-mapping on an input image.
    :param image_illumination: The illumination of the input image to tone-map(equivalent to the Y in xyY format).
    :param K: The intensity of the shift acted upon the input img mean.
    :param B: The amount of contrast suppression that will be performed in the output image.
    :param N: The number of pixels in the image.
    :param eps: A small constant being used for numerical stability purposes.
    :return:
    """
    i_mean = np.exp(1 / N * np.log(image_illumination + eps).sum())
    i_hdr = K / i_mean * image_illumination
    i_white = B * image_illumination.max()
    i_tm = (i_hdr + (i_hdr / i_white) ** 2) / (1 + i_hdr)

    return i_tm


def XYZ_to_xyY(XYZ_img: np.ndarray):
    """
    Parse the image from a XYZ to xyY format.
    Both input and output are np.ndarray
    :param XYZ_img: input image as a np.ndarray
    :return: np.ndarray of the image in xyY format
    """
    X = XYZ_img[:, :, 0]
    Y = XYZ_img[:, :, 1]
    Z = XYZ_img[:, :, 2]

    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)

    xyY_img = np.dstack((x, y, Y))
    return xyY_img


def photographic_tone_mapping(img: np.ndarray, K: float=0.15, B: float=0.95, eps: float=1e7):
    """
    Perform the tone-mapping on an input image.
    :param img: The input image to tone-map.
    :param K: The intensity of the shift acted upon the input img mean.
    :param B: The amount of contrast suppression that will be performed in the output image.
    :param N: The number of pixels in the image.
    :param eps: A small constant being used for numerical stability purposes.
    :return: a new tone-mapped image as a np.ndarray.
    """
    # parse image from rgb to XYZ
    xyz_img = lRGB2XYZ(img)

    # parse image from rgb to xyY
    xyY_img = XYZ_to_xyY(xyz_img)
    x = xyY_img[:, :, 0]
    y = xyY_img[:, :, 1]
    Y = xyY_img[:, :, 2]
    N = Y.shape[0] * Y.shape[1]

    # tone-map on the Y(illumination)
    Y_tone_mapped_img = tone_map(Y, N, K, B, eps)

    # parse the image with the newly tone-mapped Y from xyY to XYZ
    xyz_tone_mapped_img = xyY_to_XYZ(x, y, Y_tone_mapped_img)

    # parse the image from XYZ to RGB
    tone_mapped_img = XYZ2lRGB(xyz_tone_mapped_img)

    return tone_mapped_img


def main():
    hdr_img = cv2.imread("uniform.HDR")
    tone_mapped_img = photographic_tone_mapping(hdr_img)


if __name__ == '__main__':
    main()
