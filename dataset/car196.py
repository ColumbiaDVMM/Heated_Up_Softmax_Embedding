#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: car196.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-07-2018
#  Last Modified: Fri Sep  7 14:48:41 2018
#
#  Usage: python car196.py -h
#  Description:
#
#  Copyright (C) 2018 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================

import numpy as np
import scipy.io as sio
import os
from PIL import Image, ImageChops
from tqdm import tqdm

import urllib
import tarfile

import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', default='/home/xuzhang/project/Medifor/data/car196/',
        help='Folder to download the image data, need to be a non-existent folder')
parser.add_argument('--save_dir', default='./data/', help='Folder to save the processed data')
args = parser.parse_args()

data_dir = args.data_dir

try:
    os.stat(data_dir)
    print('Data floder exists, exit')
    os._exit(1)
except:
    os.makedirs(data_dir)

try:
    os.stat(args.save_dir)
except:
    os.makedirs(args.save_dir)

image_url = "http://imagenet.stanford.edu/internal/car196/car_ims.tgz"
annotation_url = "http://imagenet.stanford.edu/internal/car196/cars_annos.mat"
for download_url in [image_url, annotation_url]:
    filename = download_url.split('/')[-1]
    download_filename = "{}/{}".format(data_dir, filename)
    try:
        print("Download: {}".format(download_url))
        urllib.urlretrieve(download_url,download_filename)
        if download_filename[-3:] != 'mat':
            tar = tarfile.open(download_filename)
            tar.extractall('{}/'.format(data_dir))
            tar.close()
            os.remove(download_filename)
    except:
        print('Cannot download from {}.'.format(download_url))

training_img_list = []
validation_img_list = []

training_label_list = []
validation_label_list = []
fix_image_width = 256
fix_image_height = 256

print('Preprocessing.')
annotation = sio.loadmat(data_dir+'/cars_annos.mat')
annotation = annotation['annotations'][0]
for label in tqdm(annotation):
    image_name, left, top, right, bottom, class_id, test_flag = label
    image_name = image_name[0]
    class_id = class_id[0][0]
    img = Image.open(data_dir+image_name)
    img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
    pix_array = np.array(img)
    if len(pix_array.shape) == 2:
        pix_array.resize((pix_array.shape[0], pix_array.shape[1], 1))
        pix_array = np.repeat(pix_array, 3, 2)
    if pix_array.shape[2]>3:
        pix_array = pix_array[:,:,:3]
    if class_id <=98:
        training_img_list.append(pix_array)
        training_label_list.append(class_id)
    else:
        validation_img_list.append(pix_array)
        validation_label_list.append(class_id)

training_img = np.array(training_img_list)
training_label = np.array(training_label_list)
print("Training Image Array Size: {}".format(training_img.shape))
np.save(args.save_dir + '/training_car196_256resized_img.npy', training_img)
np.save(args.save_dir + '/training_car196_256resized_label.npy', training_label)
validation_img = np.array(validation_img_list)
validation_label = np.array(validation_label_list)
print("Test Image Array Size: {}".format(validation_img.shape))
np.save(args.save_dir + '/validation_car196_256resized_img.npy', validation_img)
np.save(args.save_dir + '/validation_car196_256resized_label.npy', validation_label)
