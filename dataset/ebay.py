#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: ebay.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-09-2018
#  Last Modified: Mon Sep 10 13:57:17 2018
#
#  Usage: python ebay.py -h
#  Description:
#
#  Copyright (C) 2018 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================

import numpy as np
import os
from PIL import Image, ImageChops
from tqdm import tqdm

import urllib
import zipfile

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', default='/home/xuzhang/project/Medifor/data/',
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

image_url = "ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip"
for download_url in [image_url]:
    filename = download_url.split('/')[-1]
    download_filename = "{}/{}".format(data_dir, filename)
    try:
        print("Download: {}".format(download_url))
        urllib.urlretrieve(download_url,download_filename)
        tar = zipfile.ZipFile(download_filename, 'r')
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
index = 0
with open(data_dir+'/ebay/Stanford_Online_Products/Ebay_train.txt', 'r') as label_file:
    for info in tqdm(label_file):
        if index == 0:
            index = index + 1
            continue
        img_idx, class_id, _, file_name = info.split(' ')
        class_id = int(class_id)
        file_name = file_name[:-1]
        #print(class_id, file_name)
        img = Image.open(data_dir+'/ebay/Stanford_Online_Products/'+file_name)
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        #img = trim(img)
        #img = pad(img)
        pix_array = np.array(img)
        if len(pix_array.shape) == 2:
            pix_array.resize((pix_array.shape[0], pix_array.shape[1], 1))
            pix_array = np.repeat(pix_array, 3, 2)
        training_img_list.append(pix_array)
        training_label_list.append(class_id)

training_img = np.array(training_img_list)
training_label = np.array(training_label_list)
print(training_img.shape)
print(training_label.shape)
np.save(args.save_dir +  '/training_ebay_256resized_img.npy', training_img)
np.save(args.save_dir + '/training_ebay_256resized_label.npy', training_label)
training_img = None
training_label = None

index = 0
with open(data_dir+'/ebay/Stanford_Online_Products/Ebay_test.txt', 'r') as label_file:
    for info in tqdm(label_file):
        if index == 0:
            index = index + 1
            continue
        img_idx, class_id, _, file_name = info.split(' ')
        class_id = int(class_id)
        file_name = file_name[:-1]
#       print(class_id, file_name)
        img = Image.open(data_dir+'/ebay/Stanford_Online_Products/'+file_name)
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        #img = trim(img)
        #img = pad(img)
        pix_array = np.array(img)
        if len(pix_array.shape) == 2:
            pix_array.resize((pix_array.shape[0], pix_array.shape[1], 1))
            pix_array = np.repeat(pix_array, 3, 2)
        validation_img_list.append(pix_array)
        validation_label_list.append(class_id)

validation_img = np.array(validation_img_list)
validation_label = np.array(validation_label_list)
print(validation_img.shape)
print(validation_label.shape)
np.save(args.save_dir + '/validation_ebay_256resized_img.npy', validation_img)
np.save(args.save_dir + '/validation_ebay_256resized_label.npy', validation_label)
