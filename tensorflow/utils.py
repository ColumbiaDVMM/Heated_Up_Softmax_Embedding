#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: utils.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-07-2018
#  Last Modified: Fri Sep  7 14:50:19 2018
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
from Loggers import Logger, FileLogger
from sklearn.cluster import KMeans
import sklearn


def get_histogram(embedding, norm_embedding, normed_weights, label):
    cos_dis = np.matmul(norm_embedding,normed_weights)
    norm_val = np.sum(embedding*embedding, axis = 1)
    histogram = np.zeros(11)
    num_list = np.zeros(11)
    bad_wrong_histogram = np.zeros(11)
    bad_num_list = np.zeros(11)
    bad_correct_histogram = np.zeros(11)
    bad_num_list = np.zeros(11)

    for i in range(cos_dis.shape[0]):
        predict_label = np.argmax(cos_dis[i,:])
        ind = 0
        ind = int(norm_val[i]/10)
        if ind>10:
            ind = 10
        if ind<0:
            ind = 0
        histogram[ind] = histogram[ind]+cos_dis[i,label[i]]
        num_list[ind] = num_list[ind]+1
        if predict_label != label[i]:
            bad_wrong_histogram[ind] = bad_wrong_histogram[ind]+cos_dis[i,predict_label]
            bad_correct_histogram[ind] = bad_correct_histogram[ind]+cos_dis[i,label[i]]
            bad_num_list[ind] = bad_num_list[ind]+1
    for i in range(11):
        if num_list[i]==0:
            continue
        histogram[i] = histogram[i]/num_list[i]
        if bad_num_list[i]==0:
            continue
        bad_wrong_histogram[i] = bad_wrong_histogram[i]/bad_num_list[i]
        bad_correct_histogram[i] = bad_correct_histogram[i]/bad_num_list[i]
    return histogram, bad_wrong_histogram, bad_correct_histogram, bad_num_list, num_list

def get_norm_and_number(embedding, norm_embedding, normed_weights, label):
    cos_dis_mat = np.matmul(norm_embedding,normed_weights)
    norm_val =np.sqrt(np.sum(embedding*embedding, axis = 1))
    cos_dis = np.zeros(embedding.shape[0])
    for i in range(embedding.shape[0]):
        cos_dis[i] = cos_dis_mat[i,label[i]]
    return norm_val, cos_dis

def get_category_mean(embedding, label, num_category):
    embedding_dim = embedding.shape[1]
    category_mean = np.zeros((num_category,embedding_dim), dtype = np.float32)
    for i in range(num_category):
        feature_by_category = embedding[label==i,:]
        category_mean_t = np.mean(feature_by_category,axis = 0)
        category_mean_t = category_mean_t/np.sqrt(np.sum(category_mean_t ** 2)+1e-4)
        category_mean[i,:] = category_mean_t
    return category_mean

def eval_all_acc(logits, label):
    num_sample = logits.shape[0]
    pred_label = np.argmax(logits, axis = 1)
    acc = np.sum(np.equal(pred_label,label))/float(num_sample)
    return acc

def eval_seperateness(model_weights):
    seperateness = np.sum(np.cov(model_weights.T)) - np.trace(np.cov(model_weights.T))
    return seperateness

def bad_count(model_weights):
    bad_num = np.sum(np.matmul(model_weights.T,model_weights)>0.5)-model_weights.shape[1]
    return bad_num

def eval_compactness(embedding, label, n_classes, normed_flag = False):
    compactness = 0
    norm_embedding = np.copy(embedding)
    if normed_flag:
        for i in range(norm_embedding.shape[0]):
            norm_embedding[i,:] = norm_embedding[i,:]/np.sqrt(np.sum(norm_embedding[i,:] ** 2)+1e-4)
    for i in range(n_classes):
        embedding_by_class = norm_embedding[label==i,:]
        compactness += np.trace(np.cov(embedding_by_class.T))/n_classes
    return compactness

def eval_feature_norm_var(embedding):
    embedding_norm = np.sqrt(np.sum(embedding**2, axis = 1))
    mean = np.mean(embedding_norm)
    var = np.var(embedding_norm)
    return mean,var

def eval_nmi(embedding, label, num_category, normed_flag = False, fast_kmeans = False):
    if normed_flag:
        for i in range(embedding.shape[0]):
            embedding[i,:] = embedding[i,:]/np.sqrt(np.sum(embedding[i,:] ** 2)+1e-4)
    if fast_kmeans:
        kmeans = KMeans(n_clusters=num_category, n_init = 1, n_jobs=8)
    else:
        kmeans = KMeans(n_clusters=num_category)
    kmeans.fit(embedding)
    y_kmeans_pred = kmeans.predict(embedding)
    nmi = sklearn.metrics.normalized_mutual_info_score(label, y_kmeans_pred)
    return nmi

def eval_recall_old(embedding, label):
    norm = np.sum(embedding*embedding,axis = 1)
    norm = np.reshape(norm, (norm.shape[0],1))
    norm_mat = np.repeat(norm,embedding.shape[0],axis = 1)
    dis = norm_mat+norm_mat.T-2*np.matmul(embedding,embedding.T)
    dis = dis+np.diag(1e10*np.ones(embedding.shape[0]))
    pred = np.argmin(dis,axis = 1)
    right_num = 0
    for i in range(embedding.shape[0]):
        if label[i]==label[pred[i]]:
            right_num = right_num+1
    recall = float(right_num)/float(embedding.shape[0])
    return recall

def eval_recall(embedding, label):
    norm = np.sum(embedding*embedding,axis = 1)
    #norm = np.reshape(norm, (norm.shape[0],1))
    #dis = norm_mat+norm_mat.T-2*np.matmul(embedding,embedding.T)
    #dis = dis+np.diag(1e10*np.ones(embedding.shape[0]))
    #pred = np.argmin(dis,axis = 1)
    right_num = 0
    for i in range(embedding.shape[0]):
        dis = norm[i] + norm - 2*np.squeeze(np.matmul(embedding[i],embedding.T))
        dis[i] = 1e10
        pred = np.argmin(dis)
        if label[i]==label[pred]:
            right_num = right_num+1
    recall = float(right_num)/float(embedding.shape[0])
    return recall

def eval_recall_K(embedding, label, K_list):
    norm = np.sum(embedding*embedding,axis = 1)
    right_num = 0
    recall_list = K_list.copy()
    recall_list = K_list*0
    for i in range(embedding.shape[0]):
        dis = norm[i] + norm - 2*np.squeeze(np.matmul(embedding[i],embedding.T))
        dis[i] = 1e10
        index = np.argsort(dis)
        list_index = 0
        for k in range(np.max(K_list)):
            if label[i]==label[index[k]]:
                recall_list[list_index] = recall_list[list_index]+1
                break
            if k>=K_list[list_index]-1:
                list_index = list_index + 1
    recall_list = recall_list/float(embedding.shape[0])
    for i in range(recall_list.shape[0]):
        if i == 0:
            continue
        recall_list[i] = recall_list[i]+recall_list[i-1]
    return recall_list

def eval_get_list(embedding, label, K):
    norm = np.sum(embedding*embedding,axis = 1)
    right_num = 0
    return_list = np.zeros((embedding.shape[0], K), dtype = np.int32)
    good_list = np.zeros((embedding.shape[0], K), dtype = np.int32)
    for i in range(embedding.shape[0]):
        dis = norm[i] + norm - 2*np.squeeze(np.matmul(embedding[i],embedding.T))
        dis[i] = 1e10
        index = np.argsort(dis)
        list_index = 0
        for k in range(K):
            return_list[i,k] = index[k]
            if label[i]==label[index[k]]:
                good_list[i,k] = 1
    return return_list, good_list

def eval_recall_fashion(gallery_embedding, gallery_label, query_embedding, query_label):
    #print(len(gallery_label))
    #print(len(query_label))
    gallery_norm = np.sum(gallery_embedding*gallery_embedding,axis = 1)
    query_norm = np.sum(query_embedding*query_embedding,axis = 1)
    right_num = 0
    for i in range(query_embedding.shape[0]):
        dis = query_norm[i] + gallery_norm - 2*np.squeeze(np.matmul(query_embedding[i],gallery_embedding.T))
        #print(dis.shape)
        pred = np.argmin(dis)
        if query_label[i]==gallery_label[pred]:
            right_num = right_num+1
    recall = float(right_num)/float(query_embedding.shape[0])
    return recall

def eval_recall_K_fashion(gallery_embedding, gallery_label, query_embedding, query_label, K_list):
    gallery_norm = np.sum(gallery_embedding*gallery_embedding,axis = 1)
    query_norm = np.sum(query_embedding*query_embedding,axis = 1)
    right_num = 0
    recall_list = K_list.copy()
    recall_list = K_list*0
    for i in range(query_embedding.shape[0]):
        dis = query_norm[i] + gallery_norm - 2*np.squeeze(np.matmul(query_embedding[i],gallery_embedding.T))
        index = np.argsort(dis)
        list_index = 0
        for k in range(np.max(K_list)):
            if query_label[i]==gallery_label[index[k]]:
                recall_list[list_index] = recall_list[list_index]+1
                break
            if k>=K_list[list_index]-1:
                list_index = list_index + 1
    recall_list = recall_list/float(query_embedding.shape[0])
    for i in range(recall_list.shape[0]):
        if i == 0:
            continue
        recall_list[i] = recall_list[i]+recall_list[i-1]
    return recall_list

def generate_group_index(num_samples, num_classes, group_dict):
    index = np.zeros((num_samples,),dtype=np.int)
    group_index = np.random.permutation(num_classes)
    cur_ind = 0
    for i in group_index:
        for j in range(len(group_dict[i])):
            index[cur_ind] = group_dict[i][j]
            cur_ind += 1
    return index

def generate_pair_index(num_pairs, num_classes, group_dict):
    index = np.zeros((num_pairs,),dtype=np.int)
    index_p = np.zeros((num_pairs,),dtype=np.int)
    cur_ind = 0
    group_ind = np.random.randint(0, num_classes, size=(num_pairs))
    for i in group_ind:
        ind_a = np.random.randint(0,len(group_dict[i]))
        ind_b = ind_a
        while ind_b != ind_a:
            ind_b = np.random.randint(0,len(group_dict[i]))
        index[cur_ind] = group_dict[i][ind_a]
        index[cur_ind] = group_dict[i][ind_b]
        cur_ind += 1
    return index, index_p
