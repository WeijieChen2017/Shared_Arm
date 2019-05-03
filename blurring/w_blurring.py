#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import nibabel as nib
# from copy import deepcopy
from scipy import ndimage
from scipy.stats import multivariate_normal
from skimage.filters import gaussian
from global_dict.w_global import gbl_get_value


def write_XY(temp_x, data):

    # n_slice, n_pixel, n_pixel, slice_x
    slice_x = gbl_get_value("slice_x")
    n_pixel = gbl_get_value("img_shape")[0]
    n_slice = gbl_get_value("img_shape")[2]
    X = np.zeros((n_slice, n_pixel, n_pixel, slice_x))
    Y = np.zeros((n_slice, n_pixel, n_pixel, 1))

    # write X
    if slice_x == 1:
        for idx in range(n_slice):
            X[idx, :, :, 0] = temp_x[:, :, idx]
            Y[idx, :, :, 0] = data[:, :, idx]

    if slice_x == 3:
        for idx in range(n_slice):
            idx_0 = idx - 1 if idx > 0 else 0
            idx_1 = idx
            idx_2 = idx + 1 if idx < n_slice - 1 else n_slice - 1
            X[idx, :, :, 0] = temp_x[:, :, idx_0]
            X[idx, :, :, 1] = temp_x[:, :, idx_1]
            X[idx, :, :, 2] = temp_x[:, :, idx_2]
            Y[idx, :, :, 0] = data[:, :, idx]

    return X, Y


def gaussian_kernel(ksize=7):
    x, y = np.mgrid[-ksize:ksize:.01, -ksize:ksize:.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    mu = [0, 0]
    cov = [[ksize, 0], [0, ksize]]

    rv = multivariate_normal(mu, cov)

    psf = np.zeros((ksize * 2 + 1, ksize * 2 + 1))
    for idx_x in range(ksize * 2 + 1):
        for idx_y in range(ksize * 2 + 1):
            real_x = idx_x - ksize
            real_y = idx_y - ksize
            psf[idx_x, idx_y] = rv.pdf([real_x, real_y])
    psf = psf / np.sum(psf)

    return psf


def data_generator(data, blur_method, blur_para):

    temp_x = None

    # kernel convolution
    if blur_method == 'kernel_conv':
        ksize = int(blur_para)
        kernel = gaussian_kernel(ksize)
        temp_x = np.zeros(data.shape)
        for idx in range(data.shape[2]):
            temp_x[:, :, idx] = ndimage.convolve(data[:, :, idx], kernel, mode='constant')

    # skimage gaussian
    if blur_method == 'skimage_gaus':
        sigma = float(blur_para)
        temp_x = gaussian(data, sigma=sigma, multichannel=True)

    # nibabel smooth
    if blur_method == 'nib_smooth':
        fwhm = float(blur_para)
        dir_mri = gbl_get_value("dir_mri")
        file_X = nib.processing.smooth_image(nib.load(dir_mri), fwhm=fwhm, mode='nearest')
        temp_x = file_X.get_fdata()

    X, Y = write_XY(temp_x, data)

    X = X / np.amax(X)
    Y = Y / np.amax(Y)

    return X, Y

def enhance_data_generator(data):

    # # kernel convolution
    # ksize = 7
    # kernel = gaussian_kernel(ksize)
    # temp_x = np.zeros(data.shape)
    # for idx in range(data.shape[2]):
    #     temp_x[:, :, idx] = ndimage.convolve(data[:, :, idx], kernel, mode='constant')
    # X1, Y1 = write_XY(temp_x, data)
    #
    # # skimage gaussian
    # sigma = 3
    # temp_x = gaussian(data, sigma=sigma, multichannel=True)
    # X2, Y2 = write_XY(temp_x, data)

    # nibabel smooth
    fwhm = 4
    dir_mri = gbl_get_value("dir_mri")
    file_X = nib.processing.smooth_image(nib.load(dir_mri), fwhm=fwhm, mode='nearest')
    temp_x = file_X.get_fdata()
    X1, Y1 = write_XY(temp_x, data)

    # nibabel smooth
    fwhm = 5
    dir_mri = gbl_get_value("dir_mri")
    file_X = nib.processing.smooth_image(nib.load(dir_mri), fwhm=fwhm, mode='nearest')
    temp_x = file_X.get_fdata()
    X2, Y2 = write_XY(temp_x, data)

    # nibabel smooth
    fwhm = 6
    dir_mri = gbl_get_value("dir_mri")
    file_X = nib.processing.smooth_image(nib.load(dir_mri), fwhm=fwhm, mode='nearest')
    temp_x = file_X.get_fdata()
    X3, Y3 = write_XY(temp_x, data)

    # nibabel smooth
    fwhm = 7
    dir_mri = gbl_get_value("dir_mri")
    file_X = nib.processing.smooth_image(nib.load(dir_mri), fwhm=fwhm, mode='nearest')
    temp_x = file_X.get_fdata()
    X4, Y4 = write_XY(temp_x, data)

    # nibabel smooth
    fwhm = 8
    dir_mri = gbl_get_value("dir_mri")
    file_X = nib.processing.smooth_image(nib.load(dir_mri), fwhm=fwhm, mode='nearest')
    temp_x = file_X.get_fdata()
    X5, Y5 = write_XY(temp_x, data)

    X = np.vstack((X1, X2, X3, X4, X5))
    Y = np.vstack((Y1, Y2, Y3, Y4, Y5))

    X = X / np.amax(X)
    Y = Y / np.amax(Y)

    return X, Y

