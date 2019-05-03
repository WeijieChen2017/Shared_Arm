#!/usr/bin/python
# -*- coding: UTF-8 -*-

from keras.models import load_model
from model.w_train import mean_squared_error_1e6, psnr


def load_existing_model(model_id):
    dir_model = model_id + '.hdf5'
    model = load_model(dir_model, custom_objects={'mean_squared_error_1e6': mean_squared_error_1e6, 'psnr': psnr})
    return model
