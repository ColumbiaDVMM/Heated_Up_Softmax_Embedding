#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: deep_metric_learning_Inception.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-07-2018
#  Last Modified: Mon Sep 10 16:03:28 2018
#
#  Usage: python deep_metric_learning_Inception.py -h
#  Description:
#
#  Copyright (C) 2018 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================

from __future__ import print_function

import h5py
import numpy as np
import tensorflow as tf
import scipy.io
import collections
import random
import sys
import os
import layers
from tqdm import tqdm
import argparse
from Loggers import Logger, FileLogger
from nets import inception_v1
import nets.inception_utils
import utils

slim = tf.contrib.slim
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', default='car196', help='folder to image data')
parser.add_argument('--data_dir', default='./data/car196/', help='folder to image data')
parser.add_argument('--log_dir', default='./Inception_log/', help='folder to output log')
parser.add_argument('--model_dir', default='./Inception_model/', help='folder to output log')
parser.add_argument('--base_network', default='InceptionV1', help='folder to output log')
parser.add_argument('--base_network_file', default='./data/model/inception_v1.ckpt', help='folder to output log')
parser.add_argument('--optimizer', default='rmsprop', help='folder to image data')
parser.add_argument("--l2_norm", action="store_true",
                    help="L2 Norm or not")
parser.add_argument("--decay_lr", action="store_true",
                    help="decayed learning rate or not")
parser.add_argument("--data_augment", action="store_true",
                    help="data augmentation or not")
parser.add_argument("--normed_test", action="store_true",
                    help="test with normed feature or not")
parser.add_argument("--norm_weights", action="store_true",
                    help="test with normed feature or not")
parser.add_argument("--bn", action="store_true",
                    help="use batch norm to the bottleneck feature or not.")
parser.add_argument("--learn_norm", action="store_true",
                    help="learning feature norm (alpha) or not")
parser.add_argument("--better_init", action="store_true",
                    help="use feature mean to initialize or not")
parser.add_argument("--label_smoothing", action="store_true",
                    help="use label smoothing or not")
parser.add_argument("--alpha_decay", action="store_true",
                    help="reduce alpha during training")
parser.add_argument("--fast_kmeans", action="store_true",
                    help="use fast kmeans to calculate NMI (for product dataset)")
parser.add_argument("--heat_up", action="store_true",
                    help="use heat up or not")
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--embedding_dim', default=64, type=int, help='embedding dim')
parser.add_argument('--alpha', default=1.0, type=float, help='alpha')
parser.add_argument('--bn_decay', default=0.95, type=float, help='parameter for bn decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate')
parser.add_argument('--training_batch_size', default=32, type=int, help='batch size for training')
parser.add_argument('--test_batch_size', default=100, type=int, help='batch size for test')
parser.add_argument('--nb_epoch', default=60, type=int, help='number of epoch for training')
parser.add_argument('--nb_hu_epoch', default=20, type=int, help='number of epoch for training')

np.set_printoptions(precision=2)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

embedding_dim = args.embedding_dim
training_batch_size = args.training_batch_size
test_batch_size = args.test_batch_size

default_image_size = 224
offset = (256-default_image_size)/2


def configure_learning_rate(num_samples_per_epoch, global_step):
    decay_steps = int(6*num_samples_per_epoch/args.training_batch_size)
    return tf.train.exponential_decay(args.learning_rate,
                    global_step, decay_steps,
                    0.94, staircase=True, name='exponential_decay_learning_rate')

def get_feature(img_data, num_category, alpha):
    ptr = 0
    num_sample = img_data.shape[0]
    embedding = np.zeros((num_sample,args.embedding_dim))
    logits = np.zeros((num_sample,num_category))
    for step in tqdm(range(int(num_sample / args.test_batch_size)+1)):
        real_size = min(args.test_batch_size, num_sample-ptr)
        if real_size <= 0:
            break
        img = np.zeros((real_size, default_image_size, default_image_size, 3),\
                dtype = np.float32)
        #center crop for test
        img[:,:,:,:] = img_data[ptr:ptr+real_size,\
                offset:(offset+default_image_size),\
                offset:(offset+default_image_size),:]

        if 'Inception' in args.base_network:
            img = (np.float32(img)/255.0-0.5)*2.0
        else:
            img = np.float32(img)
            img[:,:,:,0] = img[:,:,:,0] - _R_MEAN
            img[:,:,:,1] = img[:,:,:,1] - _G_MEAN
            img[:,:,:,2] = img[:,:,:,2] - _B_MEAN

        feature, tmp_logits = sess.run([test_net_output, test_out_layer],\
                feed_dict={img_place_holder: img, alpha_place_holder:1.0})

        embedding[ptr:ptr + real_size, :] = feature
        logits[ptr:ptr + real_size, :] = tmp_logits
        ptr += real_size

    #normalization for test
    norm_embedding = np.copy(embedding)
    for i in range(norm_embedding.shape[0]):
        norm_embedding[i,:] = norm_embedding[i,:]/np.sqrt(np.sum(norm_embedding[i,:] ** 2)+1e-4)

    return embedding, norm_embedding, logits

def read_data(ndataset,name):
    original_img_data = np.load(args.data_dir + '{}_{}_256resized_img.npy'.format(name,ndataset))
    class_id = np.load(args.data_dir + '{}_{}_256resized_label.npy'.format(name,ndataset))
    #remove label offset
    class_id = class_id-np.amin(class_id)
    num_training_sample = original_img_data.shape[0]
    num_training_category = np.unique(class_id).shape[0]
    print('{} dataset shape: {}'.format(name, original_img_data.shape))
    return original_img_data, class_id

suffix = args.dataset + '_lr_{:1.1e}'.format(args.learning_rate)
nets.inception_utils.set_bn_decay(args.bn_decay)
suffix = suffix + '_alpha_{:1.1f}'.format(args.alpha)

if not args.heat_up:
    args.nb_hu_epoch = 0
if args.l2_norm:
    suffix = suffix + '_l2n'
if args.decay_lr:
    suffix = suffix + '_dl'
if args.learn_norm:
    suffix = suffix + '_ln'
if args.better_init:
    suffix = suffix + '_bi'
if args.label_smoothing:
    suffix = suffix + '_ls'
if args.alpha_decay:
    suffix = suffix + '_ad'
if args.norm_weights:
    suffix = suffix + '_nw'
if args.bn:
    suffix = suffix + '_bn'
    args.alpha = args.alpha/np.sqrt(args.embedding_dim)
if args.heat_up:
    suffix = suffix + '_heat'
if args.embedding_dim != 64:
    suffix = suffix + '_ed_{}'.format(args.embedding_dim)
suffix = suffix + '_' + args.optimizer 

alpha = args.alpha

img_data, class_id = read_data(args.dataset, 'training')
valid_img_data, valid_class_id = read_data(args.dataset, 'validation')

num_training_sample = img_data.shape[0]
num_training_category = np.unique(class_id).shape[0]
num_valid_category = np.unique(valid_class_id).shape[0]
print('constructing model')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

#build the graph
img_place_holder = tf.placeholder(tf.float32, [None, default_image_size, default_image_size, 3])
label_place_holder = tf.placeholder(tf.float32, [None, num_training_category])
alpha_place_holder = tf.placeholder(tf.float32, shape=())
lr_place_holder = tf.placeholder(tf.float32, shape=())

#build backbone, InceptionV1
if args.base_network == 'InceptionV1':
    with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
        net_output, _ = inception_v1.inception_v1(img_place_holder, 
                embedding_dim = args.embedding_dim, use_bn = args.bn)
        test_net_output, _ = inception_v1.inception_v1(img_place_holder,
                embedding_dim = args.embedding_dim, reuse = True,
                is_training = False, use_bn = args.bn)#
else:
    print('Unknown network.')
    quit()

#build final classifier
with tf.variable_scope('retrieval'):
    retrieval_layer = layers.retrieval_layer_2(embedding_dim, num_training_category)

out_layer, bottleneck = retrieval_layer.get_output(net_output,\
        alpha = alpha_place_holder, l2_norm = args.l2_norm,\
        learn_norm = args.learn_norm, norm_weights = args.norm_weights)
test_out_layer, test_bottleneck = retrieval_layer.get_output(test_net_output,\
        alpha = alpha_place_holder, l2_norm = args.l2_norm,\
        learn_norm = args.learn_norm, norm_weights = args.norm_weights)

prediction = tf.nn.softmax(out_layer)

if args.label_smoothing:
    loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(
        logits=out_layer, onehot_labels=label_place_holder, label_smoothing = 0.1))
else:
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	logits=out_layer, labels=label_place_holder))

#dummy loss to heat up the network
dummy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    	    logits=label_place_holder, labels=label_place_holder ))

global_step = tf.train.get_or_create_global_step()
if args.decay_lr:
    learning_rate = configure_learning_rate(num_training_sample, global_step)
else:
    learning_rate = args.learning_rate

for var in tf.trainable_variables():
    if args.base_network in var.op.name:
        if 'BottleNeck' in var.op.name:
            print(var.op.name)

variables_to_restore = []
for var in slim.get_model_variables():
    if args.base_network in var.op.name:
        if 'BottleNeck' in var.op.name:
            print(var.op.name)
            continue
        variables_to_restore.append(var)

optimizer_0 = tf.train.RMSPropOptimizer(0, decay=0.94)
train_op_dummy = slim.learning.create_train_op(loss_op, optimizer_0)

#set optimizer
if args.optimizer == 'rmsprop':
    optimizer_2 = tf.train.RMSPropOptimizer(lr_place_holder, decay=0.90,  epsilon = 1.0,  momentum=0.9)
elif args.optimizer == 'SGD':
    optimizer_2 = tf.train.MomentumOptimizer(lr_place_holder,momentum=0.9)
elif args.optimizer == 'ADAM':
    optimizer_2 = tf.train.AdamOptimizer(learning_rate=lr_place_holder, epsilon = 0.01)# beta1=0.9, beta2=0.999, epsilon = 1.0
else:
    print('Unknown Optimizer')

train_op_2 = slim.learning.create_train_op(loss_op, optimizer_2)
train_op = train_op_2

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(label_place_holder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print('done constructing model')

restorer = tf.train.Saver(variables_to_restore)

sess.run(tf.global_variables_initializer())
if args.base_network == 'InceptionV1':
    restorer.restore(sess, args.base_network_file)
print('done initializing')

if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir + args.dataset)
logger = Logger(args.log_dir + args.dataset + '/' + suffix)

if not os.path.isdir('{}/{}/{}/'.format(args.model_dir, args.dataset, suffix)):
    os.makedirs('{}/{}/{}/'.format(args.model_dir, args.dataset, suffix))

saver = tf.train.Saver()

#learn one epoch to heat up the bn parameter
ptr = 0
random_ind = np.random.permutation(num_training_sample)
for step in tqdm(range(int(num_training_sample / training_batch_size)+1)):
    real_size = min(training_batch_size, num_training_sample-ptr)
    if real_size <= 0:
        break
    img = np.zeros((real_size, default_image_size,\
            default_image_size, 3), dtype = np.float32)
    label =  np.zeros((real_size, num_training_category), dtype = np.float32)
    img[:,:,:,:] = img_data[ptr:ptr+real_size,\
        offset:(offset+default_image_size),\
        offset:(offset+default_image_size),:] 

    img = (np.float32(img)/255.0-0.5)*2.0

    for i in range(real_size):
        label[i, int(class_id[ptr + i])] = 1.0

    _, cost, accu, weights = sess.run([train_op_dummy, loss_op, accuracy,\
            retrieval_layer.out],\
            feed_dict={img_place_holder: img, label_place_holder: label,\
            alpha_place_holder:alpha})
    ptr += real_size

#use feature to initialize the weight
if args.better_init:
    training_embedding, training_normed_embedding,\
            training_logits = get_feature(img_data, num_training_category, alpha)
    category_mean = utils.get_category_mean(training_embedding, class_id, num_training_category)
    classifier_init_val = np.zeros((embedding_dim,num_training_category), dtype = np.float32)
    for i in range(num_training_category):
        classifier_init_val[:,i] = np.transpose(category_mean[i,:])
    #classifier_init_val = np.random.randn(embedding_dim,num_training_category)
    #classifier_init_val = classifier_init_val.astype(np.float32)
    #for i in range(classifier_init_val.shape[1]):
    #    classifier_init_val[:,i] = classifier_init_val[:,i]/np.sqrt(np.sum(classifier_init_val[:,i] ** 2)+1e-4)

    assign_op = tf.assign(retrieval_layer.out, classifier_init_val)
    sess.run(assign_op)

change_flag = False

for epoch in range(args.nb_epoch):
    ptr = 0
    random_ind = np.random.permutation(num_training_sample)

    for step in tqdm(range(int(num_training_sample / training_batch_size)+1)):
        real_size = min(training_batch_size, num_training_sample-ptr)
        if real_size <= 0:
            break
        img = np.zeros((real_size, default_image_size,\
                default_image_size, 3), dtype = np.float32)
        label =  np.zeros((real_size, num_training_category), dtype = np.float32)

        #data augmentation
        if args.data_augment:
            flip_flag = np.random.randint(2, size=real_size)
            x_offset = np.random.randint(offset*2, size=real_size)
            y_offset = np.random.randint(offset*2, size=real_size)
            for i in range(real_size):
                img[i,:,:,:] = img_data[random_ind[ptr+i],\
                        y_offset[i]:(y_offset[i]+default_image_size),\
                        x_offset[i]:(x_offset[i]+default_image_size),:]
                if flip_flag[i]:
                    img[i,:,:,:] = img[i, :, ::-1, :]
                label[i, int(class_id[random_ind[ptr+i]])] = 1.0
        else:
            img[:,:,:,:] = img_data[ptr:ptr+real_size,\
                offset:(offset+default_image_size),\
                offset:(offset+default_image_size),:] 
            for i in range(real_size):
                label[i, int(class_id[ptr + i])] = 1.0

        img = (np.float32(img)/255.0-0.5)*2.0
        if args.learn_norm:
            _, cost, pred, accu, weights, nf, nw = sess.run([train_op, loss_op, correct_pred, accuracy,\
                    retrieval_layer.out,\
                    retrieval_layer.norm_feature,  retrieval_layer.norm_weight],\
                    feed_dict={img_place_holder: img, label_place_holder: label, \
                    alpha_place_holder:alpha, lr_place_holder: learning_rate})
        else:
            _, cost, pred, accu, weights = sess.run([train_op, loss_op, correct_pred, accuracy,\
                retrieval_layer.out],\
                feed_dict={img_place_holder: img, label_place_holder: label,\
                alpha_place_holder:alpha, lr_place_holder: learning_rate})

        ptr += real_size

    logger.log_value('loss', cost, step = epoch)
    logger.log_value('acc', accu, step = epoch)
    if args.learn_norm:
        logger.log_value('nf', nf, step = epoch)
        logger.log_value('nw', nw, step = epoch)

    if args.alpha_decay:
        if (epoch+1)%10 == 0:
            learning_rate = learning_rate*0.562
            alpha = alpha/1.41
        print('Current alpha: {}'.format(alpha))

    #heat up
    if args.heat_up:
        if epoch == args.nb_epoch - args.nb_hu_epoch:
            learning_rate = learning_rate*0.1
            alpha = alpha*0.25
            
    training_embedding, training_normed_embedding, training_logits = get_feature(img_data, num_training_category, alpha)
    if args.dataset != 'ebay':
        training_nmi = utils.eval_nmi(training_embedding, class_id, num_training_category, fast_kmeans = args.fast_kmeans)

    training_all_acc = utils.eval_all_acc(training_logits, class_id)

    if args.dataset != 'ebay':
        training_recall_at_1 = utils.eval_recall(training_embedding, class_id)
        logger.log_value('Training Recall At 1', training_recall_at_1, step = epoch)
    
    training_norm_recall_at_1 = utils.eval_recall(training_normed_embedding, class_id)
    logger.log_value('Training Norm Recall At 1', training_norm_recall_at_1, step = epoch)
    if args.dataset != 'ebay':
        logger.log_value('Training NMI', training_nmi, step = epoch)
        print('Training NMI: {}'.format(training_nmi))
    print('Training ACC: {}'.format(training_all_acc))
    logger.log_value('All Training acc', training_all_acc, step = epoch)
    
    #calculate and log all information
    model_weights = retrieval_layer.out.eval(sess)
    weight_norm_mean, weight_norm_var = utils.eval_feature_norm_var(model_weights.T)
    category_mean = utils.get_category_mean(training_normed_embedding, class_id, num_training_category)
    for i in range(model_weights.shape[1]):
        model_weights[:,i] = model_weights[:,i]/np.sqrt(np.sum(model_weights[:,i] ** 2)+1e-4)
    mean_weight_diff =np.sum((category_mean-model_weights.T)**2)
    training_histogram, bad_wrong_histogram, bad_correct_histogram, bad_num_list, num_list =\
            utils.get_histogram(training_embedding, training_normed_embedding, model_weights, class_id)
    norm_list, cos_dis_list = utils.get_norm_and_number(training_embedding, training_normed_embedding, model_weights, class_id)
    _, cos_dis_to_mean_list = utils.get_norm_and_number(training_embedding, training_normed_embedding, category_mean.T, class_id)
    weights_dis_mat = np.matmul(model_weights.T,model_weights)
    mean_dis_mat = np.matmul(category_mean, category_mean.T)
    mean_norm = np.mean(norm_list)
    var_norm = np.var(norm_list)
    mean_cos_val = np.mean(cos_dis_to_mean_list)
    var_cos_val = np.var(cos_dis_to_mean_list)

    logger.log_value('Train Norm Mean', mean_norm, step = epoch)
    logger.log_value('Train Norm Var', var_norm, step = epoch)
    logger.log_value('Train Cos Mean', mean_cos_val, step = epoch)
    logger.log_value('Train Cos Var', var_cos_val, step = epoch)
    logger.log_value('Weights Norm Mean', weight_norm_mean, step = epoch)
    logger.log_value('Mean Weights Diff',  mean_weight_diff, step = epoch)
    logger.log_histogram('Norm Histogram',  norm_list, step = epoch)
    logger.log_histogram('Cos Distance to Weight Histogram',  cos_dis_list, step = epoch)
    logger.log_histogram('Cos Distance to Mean Histogram',  cos_dis_to_mean_list, step = epoch)
    logger.log_histogram('Training Seperateness',  weights_dis_mat, step = epoch)
    logger.log_histogram('Training Mean Seperateness',  mean_dis_mat, step = epoch)
    
    #validation
    valid_embedding, valid_normed_embedding, _ = get_feature(valid_img_data, num_training_category, alpha)
    if args.dataset != 'ebay':
        validation_nmi = utils.eval_nmi(valid_embedding, valid_class_id, num_valid_category, fast_kmeans = args.fast_kmeans)
        valid_recall_at_1 = utils.eval_recall(valid_embedding, valid_class_id)

    valid_norm_recall_at_1 = utils.eval_recall(valid_normed_embedding, valid_class_id)

    valid_category_mean = utils.get_category_mean(valid_normed_embedding, valid_class_id, num_valid_category)
    valid_norm_list, valid_cos_dis_list = utils.get_norm_and_number(valid_embedding, valid_normed_embedding, valid_category_mean.T, valid_class_id)
    valid_weights_dis_mat = np.matmul(valid_category_mean,valid_category_mean.T)

    if args.dataset != 'ebay':
        logger.log_value('Test NMI', validation_nmi, step = epoch)
        logger.log_value('Test Recall At 1', valid_recall_at_1, step = epoch)
    logger.log_value('Test Norm Recall At 1', valid_norm_recall_at_1, step = epoch)

    logger.log_histogram('Test Norm Histogram',  valid_norm_list, step = epoch)
    logger.log_histogram('Test Cos Distance To Mean Histogram',  valid_cos_dis_list, step = epoch)
    logger.log_histogram('Test Mean Seperateness',  valid_weights_dis_mat, step = epoch)
    print('Test Norm Recall At 1: {}'.format(valid_norm_recall_at_1))

    if args.normed_test:
        if args.dataset != 'ebay':
            training_normed_nmi = utils.eval_nmi(training_normed_embedding, class_id,\
                    num_training_category, True, fast_kmeans = args.fast_kmeans)
            logger.log_value('Training Normed NMI', training_normed_nmi, step = epoch)
            validation_normed_nmi = utils.eval_nmi(valid_normed_embedding, valid_class_id,\
                    num_valid_category, True, fast_kmeans = args.fast_kmeans)
            logger.log_value('Test Normed NMI', validation_normed_nmi, step = epoch)
        else:
            if (epoch+1)%5 == 0:
                training_normed_nmi = utils.eval_nmi(training_normed_embedding, class_id,\
                        num_training_category, True, fast_kmeans = args.fast_kmeans)
                logger.log_value('Training Normed NMI', training_normed_nmi, step = epoch)
                validation_normed_nmi = utils.eval_nmi(valid_normed_embedding, valid_class_id,\
                        num_valid_category, True, fast_kmeans = args.fast_kmeans)
                logger.log_value('Test Normed NMI', validation_normed_nmi, step = epoch)

    if (epoch+1)%10 == 0:
        saver.save(sess, '{}/{}/{}/check_point'.format(args.model_dir, args.dataset, suffix),\
                global_step=epoch)
