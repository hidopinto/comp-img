import numpy as np
from scipy.optimize import minimize

from coords import square_coords
import cv2
from src.cp_hw2 import read_colorchecker_gm
import matplotlib

matplotlib.use("TkAgg")


def w_uniform(x, tk, zmin, zmax):
    x_new = np.zeros_like(x)
    x_new[np.where((x >= zmin) & (x <= zmax))] = 1
    return x_new


def w_tent(x, tk, zmin, zmax):
    x_new = x  ##assume x is the minimum between x and 1-x
    x_new[np.where((1 - x) < x)] = (1 - x)[np.where((1 - x) < x)]  ##for all the places where 1-x is less, fill them in
    x_new[~np.logical_and(x <= zmax, x >= zmin)] = 0
    return x_new


def w_gaussian(x, tk, zmin, zmax):
    x_new = np.exp((-4) * ((x - 0.5) ** 2) / (0.5 ** 2))
    x_new[~np.logical_and(x <= zmax, x >= zmin)] = 0
    return x_new


def w_photon(x, tk, zmin, zmax):
    x_new = np.ones_like(x) * tk
    x_new[~np.logical_and(x <= zmax, x >= zmin)] = 0
    return x_new


def read_tiff(i):
    scaling_factor = 2 ** 16 - 1
    return cv2.imread(f"TIFF_files/exposure{i}.tiff", -1) / scaling_factor


def form_hdr(ldr_files, w_func, Z_min, Z_max, tks):
    num = sum(list(map(lambda x: w_func(x[0], x[1], Z_min, Z_max) * x[0] / x[1], zip(ldr_files, tks))))
    denom = sum(list(map(lambda x: w_func(x[0], x[1], Z_min, Z_max), zip(ldr_files, tks))))
    ##to make the computation work, clip the zeros so this works
    nonzero_mask = np.where(denom > 0)
    all_uncorrected = np.zeros_like(num)
    all_uncorrected[nonzero_mask] = num[nonzero_mask] / denom[nonzero_mask]
    valid_only = num[nonzero_mask] / denom[nonzero_mask]
    ##grab the maximum and minimum valid pixel
    valid_only_min, valid_only_max = valid_only.min(), valid_only.max()
    ##we have an image that includes over and under exposed images
    ##This is okay because all the operations are done pixel wise
    ##We still haven't done the correction yet
    ##replace the under and over exposed values in the final image with the maximum they can be
    all_corrected = np.clip(all_uncorrected, a_min=valid_only_min, a_max=valid_only_max)
    return all_corrected


def q_1():
    Z_min = 0.02
    Z_max = 0.95
    tks = [1 / 2048 * (2 ** (i - 1)) for i in range(1, 17)]
    files = list(map(lambda x: read_tiff(x), list(range(1, 17))))
    saved_uniform = form_hdr(files, w_uniform, Z_min, Z_max, tks)
    cv2.imwrite("uniform.HDR", saved_uniform[:, :, ::-1].astype(np.float32))
    saved_tent = form_hdr(files, w_tent, Z_min, Z_max, tks)
    cv2.imwrite("tent.HDR", saved_tent[:, :, ::-1].astype(np.float32))
    saved_gaussian = form_hdr(files, w_gaussian, Z_min, Z_max, tks)
    cv2.imwrite("gaussian.HDR", saved_gaussian[:, :, ::-1].astype(np.float32))
    saved_photon = form_hdr(files, w_photon, Z_min, Z_max, tks)
    cv2.imwrite("photon.HDR", saved_photon[:, :, ::-1].astype(np.float32))
    ## pick the best image
    hdr_img = saved_gaussian

    return hdr_img


def make_single_long_rgb_matrix(avg_rgb_list):
    res = []
    for i in range(3):
        temp = np.zeros((3, 4), dtype=np.float64)
        temp[i, :] = avg_rgb_list
        res.append(temp)
    res = np.concatenate(res, axis=1)
    return res


def do_colour_solver(avg_rgb, target):
    tall_long_rgb_matrix = np.concatenate([make_single_long_rgb_matrix(avg_rgb[:, i]) for i in range(24)], axis=0)
    tall_target = target.transpose(2, 1, 0).ravel()
    return np.linalg.pinv(
        tall_long_rgb_matrix.transpose() @ tall_long_rgb_matrix) @ tall_long_rgb_matrix.transpose() @ tall_target


def objective(M_flat, pred, label):
    """
    Objective function: minimize the Frobenius norm
    :param M_flat:
    :param pred:
    :param label:
    :return:
    """
    M = M_flat.reshape(4, 4)
    diff = (M @ pred.T).T[:, :3] - label
    return np.linalg.norm(diff, ord='2') ** 2


def affine_constraint(M_flat):
    """
    Constraint: enforce the last row of M to be [0, 0, 0, 1]
    :param M_flat:
    :return:
    """
    M = M_flat.reshape(4, 4)
    return M[3, :] - np.array([0, 0, 0, 1])


def affine_color_solver(avg_rgb, target):
    # pred shape (24, 4)
    pred = avg_rgb.transpose(1, 0)
    # label shape (24, 3)
    label = target.reshape(3, 24).transpose(1, 0)

    # Initial guess for M with shape (4, 4)
    M_initial = np.eye(4).flatten()

    constraint = {'type': 'eq', 'fun': affine_constraint}
    result = minimize(objective, M_initial, args=(pred, label), constraints=[constraint])

    affine_matrix = result.x.reshape(4, 4)

    return affine_matrix


def q_2(hdr_img):
    avg_rgb = np.ones((4, 24), dtype=np.float64)
    for i in range(24):
        x = square_coords[:, 0, i]
        y = square_coords[:, 1, i]
        avg_rgb[:, i] = list(hdr_img[min(x):max(x), min(y):max(y), :].mean(axis=(0, 1))) + [1]
    target = np.array(read_colorchecker_gm())
    T = affine_color_solver(avg_rgb, target)[:3, :].reshape(3, 4)
    hdr_img_extra_channel = np.concatenate(
        [hdr_img, np.ones((hdr_img.shape[0], hdr_img.shape[1], 1), dtype=np.float64)],
        axis=-1
    )
    new_hdr = np.einsum("ij,xyj->xyi", T, hdr_img_extra_channel)
    new_hdr = np.clip(new_hdr, a_min=0, a_max=1)
    white_square_values = list(new_hdr[min(square_coords[:, 0, 23]):max(square_coords[:, 0, 23]),
                               min(square_coords[:, 1, 23]): max(square_coords[:, 0, 23]), :].mean(axis=(0, 1)))
    for i in range(3):
        new_hdr[:, :, i] /= white_square_values[i]
    cv2.imwrite(f"test.HDR", new_hdr.astype(np.float32))


def main():
    matplotlib.use("TkAgg")
    hdr_img = q_1()
    q_2(hdr_img)


if __name__ == "__main__":
    main()
