#!/usr/bin/python
# -*- coding: UTF-8 -*-
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
    parser.add_argument('--dir_pet', metavar='', type=str, default="breast1_pet",
                        help='Name of PET subject.(breast1_pet)<str>')
    parser.add_argument('--dir_mri', metavar='', type=str, default="breast1_water",
                        help='Name of MRI subject.(breast1_water)<str>')
    parser.add_argument('--blur_method', metavar='', type=str, default="nib_smooth",
                        help='The blurring method of synthesizing PET(nib_smooth)<str> [kernel_conv/skimage_gaus/nib_smooth]')
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
    parser.add_argument('--run_aim', metavar='', type=str, default='train',
                        help='Why do you run this program? (train)<str>')

    args = parser.parse_args()

    model_name = args.model_name

    dir_mri = './data/' + args.dir_mri + '.nii'
    dir_pet = './data/' + args.dir_pet + '.nii'

    time_stamp = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M")
    model_id = args.id + time_stamp
    enhance_blur = args.enhance_blur
    gbl_set_value("depth", args.depth)
    gbl_set_value("dir_mri", dir_mri)
    gbl_set_value("dir_pet", dir_pet)
    gbl_set_value("model_id", model_id)
    gbl_set_value("n_epoch", args.epoch + 1)
    gbl_set_value("n_filter", args.n_filter)
    gbl_set_value("depth", args.depth)
    gbl_set_value("batch_size", args.batch_size)
    gbl_set_value("slice_x", args.slice_x)
    gbl_set_value("run_aim", args.run_aim)

    # Load data
    file_pet = nib.load(dir_pet)
    file_mri = nib.load(dir_mri)

    data_pet = file_pet.get_fdata()
    data_mri = file_mri.get_fdata()

    gbl_set_value("img_shape", data_pet.shape)

    print("Loading Completed!")

    if model_name == '':
        if not enhance_blur:
            X, Y = data_generator(data_mri, args.blur_method, args.blur_para)
        else:
            X, Y = enhance_data_generator(data_mri)
            print(X.shape)

        print("Blurring Completed!")
        model = train_a_unet(X, Y)
        print("Training Completed!")

        predict(model, data_pet)
        print("Predicting Completed!")

        # send_emails(model_id)
        # print("Notification completed!")

    else:
        gbl_set_value("model_id", model_name[5:])
        model = load_existing_model(model_name)

        predict(model, data_pet)
        print("Predicting Completed!")


if __name__ == "__main__":
    main()
