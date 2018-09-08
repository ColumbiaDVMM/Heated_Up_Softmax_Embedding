#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: layers.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-07-2018
#  Last Modified: Fri Sep  7 21:07:05 2018
#
#  Usage: 
#  Description:
#
#  Copyright (C) 2018 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================

from __future__ import print_function

import tensorflow as tf
import numpy as np

class retrieval_layer_2:
    def __init__(self, bn_node, n_classes):
        self.bn_node = bn_node
        self.n_classes = n_classes
        #weights
        self.out = tf.get_variable("retrieval_out", shape=[bn_node, n_classes],
            initializer=tf.variance_scaling_initializer())

        #bias
        self.bout = tf.Variable(tf.zeros([n_classes]))

        self.norm_feature = tf.Variable(1.0, tf.float32)
        self.norm_weight = tf.Variable(1.0, tf.float32)

        self.var_dict = {
                'classifier_w': self.out,
                'classifier_b': self.bout,
                }

    def get_output(self, pre_layer, alpha, l2_norm = False, norm_weights = False, learn_norm = False):
        if l2_norm:
            layer_3 = tf.nn.l2_normalize(pre_layer, dim = 1)
            norm_out = tf.nn.l2_normalize(self.out,dim = 0)
            out_layer = tf.matmul(layer_3, norm_out)
            if learn_norm:
                out_layer = self.norm_feature*out_layer + self.bout
            else:
                out_layer = alpha*out_layer + self.bout
        else:
            layer_3 = pre_layer
            if norm_weights:
                norm_out = tf.nn.l2_normalize(self.out, dim = 0)
                if learn_norm:
                    out_layer = self.norm_feature*tf.matmul(pre_layer, norm_out) + self.bout
                else:
                    out_layer = alpha*(tf.matmul(pre_layer, norm_out)) + self.bout
            else:
                if learn_norm:
                    out_layer = self.norm_feature*tf.matmul(pre_layer, norm_out) + self.bout
                else:
                    out_layer = alpha*tf.matmul(pre_layer, self.out) + self.bout
        return out_layer, layer_3

def nca_loss(distance_matrix, one_hot_label):
    #pos_dis = tf.reduce_sum(tf.exp(distance_matrix)*one_hot_label, axis = 1)
    pos_dis = tf.reduce_sum(distance_matrix*one_hot_label, axis = 1)
    #neg_dis = tf.reduce_sum(tf.exp(distance_matrix)*(1.0-one_hot_label), axis = 1)
    neg_dis = tf.reduce_max(distance_matrix*(1.0-one_hot_label), axis = 1)
    #loss = -1.0*tf.reduce_mean(tf.log(pos_dis/neg_dis))
    loss = tf.reduce_mean(neg_dis-pos_dis)
    return loss
