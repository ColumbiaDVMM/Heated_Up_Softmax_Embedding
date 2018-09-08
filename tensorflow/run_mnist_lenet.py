#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: run_mnist_lenet.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-08-2018
#  Last Modified: Sat Sep  8 15:51:45 2018
#
#  Usage: python run_mnist_lenet.py -h
#  Description:
#
#  Copyright (C) 2018 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================
import subprocess
import shlex
import time

gpu_set = ['0']

parameter_set = [
            ' --normed_test --norm_weights --alpha=16.0 --l2_norm --learning_rate=0.001 --draw --heat_up',
            ' --normed_test --norm_weights --alpha=16.0 --l2_norm --learning_rate=0.001 --draw',
            ' --normed_test --norm_weights --alpha=0.25 --l2_norm --learning_rate=0.001 --draw',
            ' --normed_test --norm_weights --alpha=4.0 --l2_norm --learning_rate=0.001 --draw',
            ' --normed_test --norm_weights --alpha=32.0 --l2_norm --learning_rate=0.001 --draw',
            ' --normed_test --norm_weights --alpha=64.0 --l2_norm --learning_rate=0.001 --draw',
        ]

number_gpu = len(gpu_set)
process_set = []

for idx, parameter in enumerate(parameter_set):
    print('Test Parameter: {}'.format(parameter))
    
    command = 'python ./tensorflow/metric_learning_lenet.py --nb_epoch 100  {} \
            --gpu-id {}  '.format(parameter, gpu_set[idx%number_gpu])

    print(command)
    p = subprocess.Popen(shlex.split(command))
    process_set.append(p)
    
    if (idx+1)%number_gpu == 0:
        print('Wait for process end')
        for sub_process in process_set:
            sub_process.wait()
    
        process_set = []

    time.sleep(60)

for sub_process in process_set:
    sub_process.wait()
