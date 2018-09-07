#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: bird200.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 07-09-2018
#  Last Modified: Fri Sep  7 14:31:57 2018
#
#  Usage: python bird200.py -h
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
import tarfile

import argparse

def trim(im):
    bg = Image.new(im.mode, im.size, 'white')
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -55)
    bbox = diff.getbbox()
    if bbox:
        a = max(0,bbox[0]-20)
        b = max(0,bbox[1]-20)
        c = min(im.size[0],bbox[2]+20)
        d = min(im.size[1],bbox[3]+20)
        bbox = (a,b,c,d)
        return im.crop(bbox)
    return im

def pad(im):
    if im.size[0]>im.size[1]:
        im = im.resize((fix_image_width, fix_image_height*im.size[1]/im.size[0]), Image.ANTIALIAS)
    elif im.size[1]>im.size[0]:
        im = im.resize((fix_image_width*im.size[0]/im.size[1], fix_image_height), Image.ANTIALIAS)
    else:
        im = im.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)

    new_im = Image.new(im.mode,(fix_image_width, fix_image_height), 'white')

    new_im.paste(im, ((fix_image_width-im.size[0])/2,
                      (fix_image_height-im.size[1])/2))
    return new_im

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', default='/home/xuzhang/project/Medifor/data/bird200/',
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

image_url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
for download_url in [image_url]:
    filename = download_url.split('/')[-1]
    download_filename = "{}/{}".format(data_dir, filename)
    try:
        print("Download: {}".format(download_url))
        urllib.urlretrieve(download_url,download_filename)
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
with open('{}/CUB_200_2011/image_class_labels.txt'.format(args.data_dir), 'r') as label_file:
    with open('{}/CUB_200_2011/images.txt'.format(args.data_dir), 'r') as image_file:
        for label, image in tqdm(zip(label_file,image_file)):
            idx, class_id = label.split(' ')
            idx, file_name = image.split(' ')
            class_id = int(class_id[:-1])
            file_name = file_name[:-1]
            img = Image.open(data_dir+'/CUB_200_2011/images/'+file_name)
            img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
            #img = trim(img)
            #img = pad(img)
            pix_array = np.array(img)
            if len(pix_array.shape) == 2:
                pix_array.resize((pix_array.shape[0], pix_array.shape[1], 1))
                pix_array = np.repeat(pix_array, 3, 2)
            if class_id <=100 :
                training_img_list.append(pix_array)
                training_label_list.append(class_id)
            else:
                validation_img_list.append(pix_array)
                validation_label_list.append(class_id)

training_img = np.array(training_img_list)
training_label = np.array(training_label_list)
print("Training Image Array Size: {}".format(training_img.shape))
np.save(args.save_dir + '/training_bird200_256resized_img.npy', training_img)
np.save(args.save_dir + '/training_bird200_256resized_label.npy', training_label)
validation_img = np.array(validation_img_list)
validation_label = np.array(validation_label_list)
print("Test Image Array Size: {}".format(validation_img.shape))
np.save(args.save_dir + '/validation_bird200_256resized_img.npy', validation_img)
np.save(args.save_dir + '/validation_bird200_256resized_label.npy', validation_label)
