import cv2
import numpy as np

from src.cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeHDR, read_colorchecker_gm, xyY_to_XYZ


def main():
    hdr_img = cv2.imread("gaussian.HDR")
    tone_mapped_img = q3(hdr_img, eps=1e7)

    writeHDR('gaussian_tone_mapped.HDR', tone_mapped_img)


if __name__ == '__main__':
    main()