import cv2
import numpy as np

from src.cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeHDR, read_colorchecker_gm, xyY_to_XYZ


def tone_map(image_illumination: np.ndarray, N: int, K: float=0.15, B: float=0.95, eps: float = np.finfo(float).eps):
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


def XYZ_to_xyY(XYZ_img: np.ndarray, eps: float = np.finfo(float).eps):
    """
    Parse the image from a XYZ to xyY format.
    Both input and output are np.ndarray
    :param XYZ_img: input image as a np.ndarray
    :return: np.ndarray of the image in xyY format
    """
    X = XYZ_img[:, :, 0]
    Y = XYZ_img[:, :, 1]
    Z = XYZ_img[:, :, 2]

    denom = X + Y + Z
    denom = np.where(denom == 0, eps, denom)

    x = X / denom
    y = Y / denom

    xyY_img = np.dstack((x, y, Y))
    return xyY_img


def q3_rgb(img: np.ndarray, K: float=0.15, B: float=0.95, eps: float = np.finfo(float).eps):
    """
    Perform the tone-mapping on an input image.
    :param img: The input image to tone-map.
    :param K: The intensity of the shift acted upon the input img mean.
    :param B: The amount of contrast suppression that will be performed in the output image.
    :param eps: A small constant being used for numerical stability purposes.
    :return: a new tone-mapped image as a np.ndarray.
    """
    # extract RGB channels
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    N = r.shape[0] * r.shape[1]

    # tone-map on the RGB channels
    r_tone_mapped = tone_map(r, N, K, B, eps)
    g_tone_mapped = tone_map(g, N, K, B, eps)
    b_tone_mapped = tone_map(b, N, K, B, eps)

    # re-stack the rgb channels
    tone_mapped_img = np.dstack([r_tone_mapped, g_tone_mapped, b_tone_mapped])

    return tone_mapped_img


def q3_luminance(img: np.ndarray, K: float=0.15, B: float=0.95, eps: float = np.finfo(float).eps):
    """
    Perform the tone-mapping on an input image.
    :param img: The input image to tone-map.
    :param K: The intensity of the shift acted upon the input img mean.
    :param B: The amount of contrast suppression that will be performed in the output image.
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
    # Avoid division by zero by adding a small epsilon where y is zero
    # y_w_eps = np.where(y == 0, eps, y)
    xyz_tone_mapped_img = np.dstack(xyY_to_XYZ(x, y, Y_tone_mapped_img))

    # parse the image from XYZ to RGB
    tone_mapped_img = XYZ2lRGB(xyz_tone_mapped_img)

    return tone_mapped_img


def main():
    hdr_img = cv2.imread("gaussian.HDR")
    tone_mapped_img = q3_luminance(hdr_img, eps=1e7)

    writeHDR('gaussian_tone_mapped.HDR', tone_mapped_img)


if __name__ == '__main__':
    main()
