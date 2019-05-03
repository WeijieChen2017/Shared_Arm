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
from blurring.sa_gen import sa_data_generator




np.random.seed(591)


def usage():
    print("Error in input argv")


def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for Partial Volume Correction in PET/MRI system. ''',
        epilog="""All's well that ends well.""")
    parser.add_argument('--X', metavar='', type=str, default="X_mnist",
                        help='X file name.(X_mnist)<str>')
    parser.add_argument('--Y', metavar='', type=str, default="Y_mnist",
                        help='Y file name.(Y_mnist)<str>')
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

    args = parser.parse_args()

    model_name = args.model_name

    dir_X = './data/' + args.dir_X + '.npy'
    dir_Y = './data/' + args.dir_Y + '.npy'

    time_stamp = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M")
    model_id = args.id + time_stamp

    gbl_set_value("depth", args.depth)
    gbl_set_value("dir_X", dir_X)
    gbl_set_value("dir_Y", dir_Y)
    gbl_set_value("model_id", model_id)
    gbl_set_value("n_epoch", args.epoch + 1)
    gbl_set_value("n_filter", args.n_filter)
    gbl_set_value("depth", args.depth)
    gbl_set_value("batch_size", args.batch_size)
    gbl_set_value("slice_x", args.slice_x)
    gbl_set_value("flag_bypass", False)

    # Load data
    file_X = np.load(dir_X)
    file_Y = np.load(dir_Y)

    gbl_set_value("img_shape", file_X.shape)

    print("Loading Completed!")

    X, Y = sa_data_generator(file_X)
    print("Data Preparation Completed!")

    model = train_a_unet(X, Y)
    print("Training Completed!")

    # predict(model, X)
    # print("Predicting Completed!")

    send_emails(model_id)
    print("Notification completed!")


if __name__ == "__main__":
    main()
