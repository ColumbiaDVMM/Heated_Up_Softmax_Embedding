#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: metric_learning_lenet.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-08-2018
#  Last Modified: Sat Sep  8 23:22:44 2018
#
#  Usage: python metric_learning_lenet.py -h
#  Description:
#
#  Copyright (C) 2018 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================

from __future__ import print_function

# Import MNIST data
import mnist as mnist_dataset
from sklearn.cluster import KMeans
import sklearn
import tensorflow as tf
import os
import numpy as np
from nets import lenet
import layers
import utils
from Loggers import Logger, FileLogger
from tqdm import tqdm

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import plotly.plotly as py
from mpl_toolkits.mplot3d import axes3d, Axes3D

import argparse

font = {'size' : 15}

mpl.rc('font', **font)

slim = tf.contrib.slim

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--log_dir', default='./mnist_logs/', help='folder to output log')
parser.add_argument('--model_dir', default='./mnist_model/', help='folder to save the model')
parser.add_argument('--save_image_dir', default='./mnist_distribution_map/',
        help='folder to save the feature distribution result')
parser.add_argument('--alpha' , type=float, default=1.0, help='beta hyperparameter value')
parser.add_argument('--beta' , type=float, default=0.0, help='beta hyperparameter value')
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--display_step', default=10, type=int, help='show step')
parser.add_argument('--embedding_dim', default=2, type=int, help='embedding dim')
parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate')
parser.add_argument('--nb_epoch', type=int, default=20, help='Number of epochs')
parser.add_argument('--nb_hu_epoch', default=20, type=int, help='number of epoch for training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument("--draw", action="store_true",
                    help="Draw scatter map")
parser.add_argument("--normed_test", action="store_true",
                    help="Draw scatter map")
parser.add_argument("--norm_weights", action="store_true",
                    help="test with normed feature or not")
parser.add_argument("--learn_norm", action="store_true",
                    help="learning feature norm (alpha) or not")
parser.add_argument("--l2_norm", action="store_true",
                    help="L2 Norm or not")
parser.add_argument("--bn", action="store_true",
                    help="use batch norm to the bottleneck feature or not.")
parser.add_argument("--label_smoothing", action="store_true",
                    help="use label smoothing or not")
parser.add_argument("--heat_up", action="store_true",
                    help="use heat up or not")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

tf.set_random_seed(args.seed)
np.random.seed(args.seed)

mnist = mnist_dataset.read_data_sets("./data/", one_hot=True, reshape=False)

def draw_feature_and_weights(save_file_name, feature, weights, test_class_id):
    embedding_dim = feature.shape[1]
    if embedding_dim == 3:
        fig = plt.figure(figsize=(4, 4), dpi=160)
        ax = Axes3D(fig)
    elif embedding_dim==2:
        fig = plt.figure(figsize=(4, 4), dpi=160)
        ax = plt.subplot(111)
    else:
        print('Only support feature with dim 3 or 2.')
        return

    n_classes = np.amax(test_class_id) + 1
    for i in range(n_classes):
        point_by_number = feature[test_class_id==i,:]
        mean = np.mean(point_by_number,axis = 0)
        if embedding_dim==2:
            ax.scatter(point_by_number[:,0],point_by_number[:,1],\
                color='C{}'.format(i), s = 30,  alpha=0.05)
        elif embedding_dim == 3:
            ax.scatter(point_by_number[:,0],point_by_number[:,1],point_by_number[:,2],\
                color='C{}'.format(i), s = 30, alpha=0.05)

    for i in range(n_classes):
        center = weights[:,i]

        if embedding_dim==2:
            ax.scatter([center[0]], [center[1]],\
                color='C{}'.format(i), edgecolor='black', s = 50, linewidth='1', label='{}', marker='D')
        elif embedding_dim == 3:
            ax.scatter(center[0], center[1], center[2],\
                color='C{}'.format(i), edgecolor='black', linewidth='1', marker='D')

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig(save_file_name, bbox_inches='tight')
    plt.close()

def get_feature(dataset, num_category):
    embedding = np.zeros((dataset.num_examples,args.embedding_dim))
    logits = np.zeros((dataset.num_examples,num_category))
    labels = np.zeros((dataset.num_examples,num_category))
    offset = 0
    test_total_batch = int(dataset.num_examples/batch_size)
    for i in range(test_total_batch):
        real_size = min(batch_size, dataset.num_examples-offset)
        batch_x, batch_y = dataset.next_batch_test(real_size)
        feature, tmp_logits = sess.run([test_net_output, test_out_layer],\
                feed_dict={X: batch_x, Y: batch_y, alpha_place_holder:args.alpha})
        embedding[offset:offset + real_size, :] = feature
        logits[offset:offset + real_size, :] = tmp_logits
        labels[offset:offset + real_size, :] = batch_y
        offset = offset+real_size    

    norm_embedding = np.copy(embedding)
    for i in range(norm_embedding.shape[0]):
        norm_embedding[i,:] = norm_embedding[i,:]/np.sqrt(np.sum(norm_embedding[i,:] ** 2)+1e-4)
    return embedding, norm_embedding, logits, labels

suffix = 'mnist_lr_{:1.1e}'.format(args.learning_rate)
suffix = suffix + '_alpha_{:1.1f}'.format(args.alpha)
if args.l2_norm:
    suffix = suffix + '_l2n'
if args.learn_norm:
    suffix = suffix + '_ln'
if args.label_smoothing:
    suffix = suffix + '_ls'
if args.norm_weights:
    suffix = suffix + '_nw'
if args.bn:
    suffix = suffix + '_bn'
    args.alpha = args.alpha/np.sqrt(args.embedding_dim)
if args.heat_up:
    suffix = suffix + '_heat'

LOG_DIR = args.log_dir
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_DIR = args.log_dir + suffix
logger = Logger(LOG_DIR)

# Parameters
learning_rate =args.learning_rate 
training_epochs = args.nb_epoch
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# tf Graph input
X = tf.placeholder("float", [None, n_input, n_input, 1])
#X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, n_classes])
alpha_place_holder = tf.placeholder(tf.float32, shape=())
lr_place_holder = tf.placeholder(tf.float32, shape=())

with slim.arg_scope(lenet.lenet_arg_scope()):
    #training and test graph
    net_output, _ = lenet.lenet(X, embedding_dim = args.embedding_dim, use_bn = args.bn)#
    test_net_output, _ = lenet.lenet(X,embedding_dim = args.embedding_dim,\
            reuse = True, is_training = False, use_bn = args.bn)#

#retrieval layer
with tf.variable_scope('retrieval'):
    retrieval_layer = layers.retrieval_layer_2(args.embedding_dim, n_classes)

out_layer, bottleneck = retrieval_layer.get_output(net_output,\
        alpha = alpha_place_holder, l2_norm = args.l2_norm,\
        learn_norm = args.learn_norm, norm_weights = args.norm_weights)
test_out_layer, test_bottleneck = retrieval_layer.get_output(test_net_output,\
        alpha = alpha_place_holder, l2_norm = args.l2_norm,\
        learn_norm = args.learn_norm, norm_weights = args.norm_weights)

pred = tf.nn.softmax(out_layer)  # Apply softmax to logits

if args.label_smoothing:
    loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(
        logits=out_layer, onehot_labels=Y, label_smoothing = 0.1))
else:
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	logits=out_layer, labels=Y))

global_step = tf.train.get_or_create_global_step()

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
optimizer = tf.train.AdamOptimizer(learning_rate=lr_place_holder)
train_op = slim.learning.create_train_op(loss_op, optimizer)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

if args.draw:
    try: 
        os.stat('{}/{}/'.format(args.save_image_dir, suffix))
    except:
        os.makedirs('{}/{}/'.format(args.save_image_dir, suffix))

try:
    os.stat('{}/{}/'.format(args.model_dir, suffix))
except:
    os.makedirs('{}/{}/'.format(args.model_dir, suffix))

# Training cycle
for epoch in tqdm(range(training_epochs)):
    total_batch = int(mnist.train.num_examples/batch_size)
    avg_cost = 0
    avg_acc = 0
    # Loop over all batches
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l, a = sess.run([train_op, loss_op, accuracy],\
                feed_dict={X: batch_x, Y: batch_y,
                    alpha_place_holder:args.alpha, lr_place_holder: learning_rate})
        # Compute average loss
        avg_cost += l / total_batch
        avg_acc += a / total_batch
    
    if args.heat_up:
        if epoch == args.nb_epoch - args.nb_hu_epoch:
            learning_rate = learning_rate*0.1
            args.alpha = args.alpha*0.25

    ## Display logs per epoch step
    if epoch % display_step == 0:

        test_embedding, test_normed_embedding, test_logits, test_labels\
                = get_feature(mnist.test, n_classes)
        test_class_id = np.argmax(test_labels, axis=1)
        model_weights = retrieval_layer.out.eval(sess)
        normed_model_weights = model_weights.copy()
        for i in range(model_weights.shape[1]):
            normed_model_weights[:,i] = model_weights[:,i]/np.sqrt(np.sum(model_weights[:,i] ** 2)+1e-4)

        test_nmi = utils.eval_nmi(test_embedding, test_class_id , n_classes)
        normed_test_nmi = utils.eval_nmi(test_normed_embedding, test_class_id , n_classes)
        test_acc = utils.eval_all_acc(test_logits, test_class_id)
        print('Test Accuracy: {}'.format(test_acc))
        print('Test NMI: {}'.format(test_nmi))
        logger.log_value('Test Accuracy', test_acc, step = epoch)
        logger.log_value('Test NMI', test_nmi, step = epoch)
        logger.log_value('Normed Test NMI', normed_test_nmi, step = epoch)
        logger.log_value('loss', avg_cost, step = epoch)
        logger.log_value('avg acc', avg_acc, step = epoch)

        saver.save(sess, '{}/{}/check_point'.format(args.model_dir, suffix),\
                global_step=epoch)

        if args.draw:
            save_png_name = '{}/{}/test_epoch{:03d}.png'.format(args.save_image_dir, suffix, epoch)
            draw_feature_and_weights(save_png_name, test_normed_embedding, \
                    normed_model_weights*0.9, test_class_id)
            if not args.l2_norm:
                save_png_name = '{}/{}/test_no_l2n_epoch{:03d}.png'.format(args.save_image_dir, suffix, epoch)
                draw_feature_and_weights(save_png_name, test_embedding, \
                    model_weights, test_class_id)
               
