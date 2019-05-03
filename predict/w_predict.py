#!/usr/bin/python
# -*- coding: UTF-8 -*-

import nibabel as nib
import numpy as np
from keras import backend as K
from global_dict.w_global import gbl_get_value


def mean_squared_error_1e6(y_true, y_pred):
    loss = K.mean(K.square(y_pred - y_true), axis=-1)

    # Energy conservation
    # reg_term = K.square(K.sum(y_pred) - K.sum(y_true))
    # loss += reg_term
    return loss


def psnr(y_true, y_pred):
    mse = K.mean(K.square(y_pred - y_true))
    return (20 - 10 * K.log(mse) / K.log(10.0))


def predict(model, data):
    dir_pet = gbl_get_value("dir_pet")
    n_pixel = gbl_get_value("img_shape")[0]
    n_slice = gbl_get_value("img_shape")[2]
    slice_x = gbl_get_value("slice_x")
    model_id = gbl_get_value("model_id")
    # dir_syn = gbl_get_value('dir_syn')
    dir_syn = './'

    run_aim = gbl_get_value("run_aim")
    flag_save = True
    if run_aim == 'see_aug':
        flag_save = False


    file_pet = nib.load(dir_pet)
    factor = np.sum(data)
    data = data / np.amax(data)
    y_hat = np.zeros(data.shape)

    X = np.zeros((1, n_pixel, n_pixel, slice_x))

    if slice_x == 1:
        for idx in range(n_slice):
            X[0, :, :, 0] = data[:, :, idx]
            y_hat[:, :, idx] = np.squeeze(model.predict(X))

    if slice_x == 3:
        for idx in range(n_slice):
            idx_0 = idx-1 if idx > 0 else 0
            idx_1 = idx
            idx_2 = idx+1 if idx < n_slice-1 else n_slice - 1
            X[0, :, :, 0] = data[:, :, idx_0]
            X[0, :, :, 1] = data[:, :, idx_1]
            X[0, :, :, 2] = data[:, :, idx_2]
            y_hat[:, :, idx] = np.squeeze(model.predict(X))

    dif = y_hat - data
    y_hat = y_hat / np.sum(y_hat) * factor

    if flag_save:

        # save nifty file
        affine = file_pet.affine
        header = file_pet.header
        nii_file = nib.Nifti1Image(y_hat, affine, header)
        nib.save(nii_file, dir_syn + 'syn_'+model_id+'.nii')

        # save difference

        dif_file = nib.Nifti1Image(dif, affine, header)
        nib.save(dif_file, dir_syn + 'dif_' + model_id + '.nii')
