#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import gc
import glob
import datetime
import argparse
import numpy as np
import nibabel as nib
from global_dict.w_global import gbl_set_value, gbl_get_value
from blurring.w_blurring import data_generator, enhance_data_generator
from model.w_train import train_a_unet
from model.w_load import load_existing_model
from predict.w_predict import predict
from notification.w_emails import send_emails




np.random.seed(591)


def usage():
    print("Error in input argv")


def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for Partial Volume Correction in PET/MRI system. ''',
        epilog="""All's well that ends well.""")
    parser.add_argument('--dir_folder', metavar='', type=str, default="crohns",
                        help='Name of dataset.(crohns)<str>')
    parser.add_argument('--blur_method', metavar='', type=str, default="nib_smooth",
                        help='The blurring method of syn PET(nib_smooth)<str> [kernel_conv/skimage_gaus/nib_smooth]')
    parser.add_argument('--blur_para', metavar='', type=str, default="4",
                        help='Parameters of blurring data(4)<str>')
    parser.add_argument('--slice_x', metavar='', type=int, default="1",
                        help='Slices of input(1)<int>[1/3]')
    parser.add_argument('--enhance_blur', metavar='', type=bool, default=False,
                        help='Whether stack different blurring methods to train the model')
    parser.add_argument('--id', metavar='', type=str, default="eeVee",
                        help='ID of the current model.(eeVee)<str>')

    parser.add_argument('--epoch', metavar='', type=int, default=500,
                        help='Number of epoches of training(2000)<int>')
    parser.add_argument('--n_filter', metavar='', type=int, default=64,
                        help='The initial filter number(64)<int>')
    parser.add_argument('--depth', metavar='', type=int, default=4,
                        help='The depth of U-Net(4)<int>')
    parser.add_argument('--batch_size', metavar='', type=int, default=10,
                        help='The batch_size of training(10)<int>')

    parser.add_argument('--model_name', metavar='', type=str, default='',
                        help='The name of model to be predicted. ()<str>')


    args = parser.parse_args()

    # common setting
    model_name = args.model_name
    enhance_blur = args.enhance_blur
    gbl_set_value("depth", args.depth)
    gbl_set_value("n_epoch", args.epoch + 1)
    gbl_set_value("n_filter", args.n_filter)
    gbl_set_value("depth", args.depth)
    gbl_set_value("batch_size", args.batch_size)
    gbl_set_value("slice_x", args.slice_x)

    # file-specific
    dir_folder = './data/dataset/' + args.dir_folder + '/'
    list_pet = glob.glob(dir_folder+'pet/*.nii.gz')
    list_mri = glob.glob(dir_folder+'mri/*.nii.gz')


    n_files = len(list_pet)
    print(n_files)
    union_mri = []

    for idx in range(n_files):
        filename_start = list_pet[idx].rfind('/')
        filename_end = list_pet[idx].find('_')
        filename = list_pet[idx][filename_start+1:filename_end]

        dir_pet = list_pet[idx]
        dir_mri = dir_folder + 'mri/' + filename + '_water.nii.gz'

        # Load data

        file_mri = nib.load(dir_mri)
        data_mri = file_mri.get_fdata()

        union_mri.append(data_mri)

        del file_mri
        del data_mri

    print("Loading Completed!")

    union_mri = np.asarray(union_mri)
    print(union_mri.shape)
    time_stamp = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M")
    model_id = filename + time_stamp

    dir_syn = './union_results/' + args.dir_folder + '/synthesized/'
    if not os.path.exists(dir_syn):
        os.makedirs(dir_syn)

    dir_model = './union_results/' + args.dir_folder + '/models/'
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)

    gbl_set_value("dir_mri", dir_mri)
    gbl_set_value("dir_pet", dir_pet)
    gbl_set_value('dir_syn', dir_syn)
    gbl_set_value('dir_model', dir_model)
    gbl_set_value("model_id", model_id)
    gbl_set_value("img_shape", data_mri.shape)

    if model_name == '':

        if not enhance_blur:
            X, Y = data_generator(union_mri, args.blur_method, args.blur_para)
        else:
            X, Y = enhance_data_generator(union_mri)
            print(X.shape)

        print("Blurring Completed!")
        model = train_a_unet(X, Y)
        print("Training Completed!")

        # predict(model, union_pet)
        # print("Predicting Completed!")

    else:
        gbl_set_value("model_id", model_name[5:])
        model = load_existing_model(model_name)

        # predict(model, union_pet)
        # print("Predicting Completed!")


    # prediction
    for idx in range(n_files):
        filename_start = list_pet[idx].rfind('/')
        filename_end = list_pet[idx].find('_')
        filename = list_pet[idx][filename_start+1:filename_end]
        gbl_set_value("model_id", filename)

        dir_pet = list_pet[idx]
        file_pet = nib.load(dir_pet)
        data_pet = file_pet.get_fdata()

        print(filename)
        predict(model, data_pet)
        print("Predicting Completed!")

    send_emails(model_id)
    print("Notification completed!")

if __name__ == "__main__":
    main()
