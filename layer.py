import tensorflow as tf
import disttf_util as util

conv_counter = 0
affine_counter = 0
pool_counter = 0
norm_counter = 0
parameters = []


def _conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType):
    global conv_counter
    global parameters
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        strides = [1, dH, dW, 1]
        conv = tf.nn.conv2d(inpOp, kernel, strides, padding=padType)
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.reshape(tf.nn.bias_add(conv, biases),
                          conv.get_shape())
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        util.print_activations(conv1)
        return conv1


def _affine(inpOp, nIn, nOut):
    global affine_counter
    global parameters
    name = 'affine' + str(affine_counter)
    affine_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        affine1 = tf.nn.relu_layer(inpOp, kernel, biases, name=name)
        parameters += [kernel, biases]
        return affine1


def _mpool(inpOp, kH, kW, dH, dW):
    global pool_counter
    global parameters
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    ksize = [1, kH, kW, 1]
    strides = [1, dH, dW, 1]
    return tf.nn.max_pool(inpOp,
                          ksize=ksize,
                          strides=strides,
                          padding='VALID',
                          name=name)

def _norm(inpOp, depth, bias, alpha, beta):
    global norm_counter
    global parameters
    name = 'norm' + str(norm_counter)
    return tf.nn.lrn(inpOp, depth_radius=depth, bias= bias, alpha=alpha, beta=beta, name=name)


def loss(logits, labels):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.stack([batch_size, 1000]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=onehot_labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss



