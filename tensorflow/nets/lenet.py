# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a variant of the LeNet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def lenet(images,
          embedding_dim=3,
          is_training=True,
          reuse=False,
          use_bn = False,
          scope='LeNet'):
    """Creates a variant of the LeNet model.
    
    Note that since the output is a set of 'logits', the values fall in the
    interval of (-infinity, infinity). Consequently, to convert the outputs to a
    probability distribution over the characters, one will need to convert them
    using the softmax function:
    
          logits = lenet.lenet(images, is_training=False)
          probabilities = tf.nn.softmax(logits)
          predictions = tf.argmax(logits, 1)
    
    Args:
      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset. If 0 or None, the logits
        layer is omitted and the input features to the logits layer are returned
        instead.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.
    
    Returns:
       net: a 2D Tensor with the logits (pre-softmax activations) if num_classes
        is a non-zero integer, or the inon-dropped-out nput to the logits layer
        if num_classes is 0 or None.
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    end_points = {}
    with tf.variable_scope(scope, 'LeNet', [images], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):

            net = end_points['conv1'] = slim.conv2d(images, 32, [5, 5], normalizer_fn=None, scope='conv1')
            net = end_points['pool1'] = slim.max_pool2d(net, [2, 2], 2, scope='pool1')

            net = end_points['conv2'] = slim.conv2d(net, 64, [5, 5], normalizer_fn=None, scope='conv2')
            net = end_points['pool2'] = slim.max_pool2d(net, [2, 2], 2, scope='pool2')

            net = slim.flatten(net)
            net = slim.fully_connected(net, 500, normalizer_fn=None, scope='fc1')

            if use_bn:
              logits = slim.fully_connected(net, embedding_dim, activation_fn=None, scope='BottleNeck')
            else:
              logits = slim.fully_connected(net, embedding_dim, activation_fn=None,
                        normalizer_fn=None, scope='BottleNeck')

    return logits, end_points

lenet.default_image_size = 28

def lenet_bn_arg_scope(weight_decay=0.00004,
                    use_batch_norm=True,
                    batch_norm_decay=0.95,
                    batch_norm_epsilon=0.001,
                    activation_fn=tf.nn.relu):
  """Defines the default lenet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      # use fused batch norm if possible.
      'fused': None,
  }

  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,
      normalizer_params=normalizer_params) as sc:
    return sc

lenet_arg_scope = lenet_bn_arg_scope
