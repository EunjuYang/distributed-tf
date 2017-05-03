from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutil as util
import tensorflow as tf
import layer



def node1(images, FLAGS):

    conv1 = layer._conv(images, 3, 64, 5, 5, 1, 1, 'SAME')
    pool1 = layer._mpool(conv1, 3, 3, 2, 2)
    norm1 = layer._norm(pool1, 4., 1.0, 0.001 / 9.0, 0.75)
    conv2 = layer._conv(norm1, 64, 64, 5, 5, 1, 1, 'SAME')
    norm2 = layer._norm(conv2, 4., 1.0, 0.001 / 9.0, 0.75)
    pool2 = layer._mpool(norm2, 3, 3, 2, 2)

    return pool2


def node2(pool2, FLAGS):

  # local3
  reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
  dim = reshape.get_shape()[1].value
  local3 = layer._affine(reshape, dim, 384)

  # local4
  local4 = layer._affine(local3, 384, 192)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = util._variable_with_weight_decay('weights', [192, 10],
                                          stddev=1/192.0, wd=0.0)
    biases = util._variable_on_cpu('biases', [10],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    util.print_activations(softmax_linear)

  return softmax_linear
