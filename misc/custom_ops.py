"""
Some codes from
https://github.com/openai/InfoGAN/blob/master/infogan/misc/custom_ops.py
"""
from __future__ import division
from __future__ import print_function

# import prettytensor as pt
# from prettytensor.pretty_tensor_class import Phase
from tensorflow.python.training import moving_averages
import tensorflow as tf
import numpy as np


def fc(inputs, num_out, name, activation_fn=None, reuse=None):
    shape = inputs.get_shape()
    if len(shape) == 4:
        inputs = tf.reshape(inputs, tf.stack([tf.shape(inputs)[0], np.prod(shape[1:])]))
        inputs.set_shape([None, np.prod(shape[1:])])

    w_init = tf.random_normal_initializer(stddev=0.02)  # tf.keras.layers.Dense
    # return tf.layers.dense(inputs=inputs, units=num_out, activation=activation_fn, kernel_initializer=w_init,
    #                        use_bias=biased, name=name)
    return tf.contrib.layers.fully_connected(inputs, num_out, activation_fn=activation_fn, weights_initializer=w_init,
                                             reuse=reuse, scope=name)


def concat(inputs, axis):
    return tf.concat(values=inputs, axis=axis)


def conv_batch_normalization(inputs, name, epsilon=1e-5, in_dim=None, is_training=True, activation_fn=None, reuse=None):
    return tf.contrib.layers.batch_norm(inputs, decay=0.9, center=True, scale=True, epsilon=epsilon,
                                        activation_fn=activation_fn,
                                        param_initializers={'beta': tf.constant_initializer(0.),
                                                            'gamma': tf.random_normal_initializer(1., 0.02)},
                                        reuse=reuse, is_training=is_training, scope=name)

    # ema = tf.train.ExponentialMovingAverage(decay=0.9)
    # shape = inputs.get_shape()
    # shp = in_dim or shape[-1]
    # mean, variance = tf.nn.moments(inputs, [0, 1, 2])
    # mean.set_shape((shp,))
    # variance.set_shape((shp,))
    # ema_apply_op = ema.apply([mean, variance])
    #
    # with tf.variable_scope(name, reuse=reuse):
    #     gamma = tf.get_variable(name='gamma', shape=[shp], initializer=tf.random_normal_initializer(1., 0.02))
    #     beta = tf.get_variable(name='beta', shape=[shp], initializer=tf.constant_initializer(0.))
    #
    #     if is_training:
    #         with tf.control_dependencies([ema_apply_op]):
    #             normalized_x = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
    #     else:
    #         normalized_x = tf.nn.batch_normalization(inputs, ema.average(mean), ema.average(variance),
    #                                                  beta, gamma, epsilon)
    #
    # if activation_fn is not None:
    #     normalized_x = activation_fn(normalized_x)
    #
    # return normalized_x


def fc_batch_normalization(inputs, name, epsilon=1e-5, in_dim=None, is_training=True, activation_fn=None, reuse=None):
    ori_shape = inputs.get_shape()
    if ori_shape[0] is None:
        ori_shape = -1
    new_shape = [ori_shape[0], 1, 1, ori_shape[1]]
    x = tf.reshape(inputs, new_shape)
    normalized_x = conv_batch_normalization(x, name, epsilon=epsilon, in_dim=in_dim, is_training=is_training,
                                            activation_fn=activation_fn, reuse=reuse)
    return tf.reshape(normalized_x, ori_shape)


def reshape(inputs, shape, name):
    return tf.reshape(inputs, shape, name)


def Conv2d(inputs, k_h, k_w, c_o, s_h, s_w, name, activation_fn=None, reuse=None, padding='SAME', biased=False):
    c_i = inputs.get_shape()[-1]
    w_init = tf.random_normal_initializer(stddev=0.02)

    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name, reuse=reuse) as scope:
        kernel = tf.get_variable(name='weights', shape=[k_h, k_w, c_i, c_o], initializer=w_init)
        output = convolve(inputs, kernel)

        if biased:
            biases = tf.get_variable(name='biases', shape=[c_o])
            output = tf.nn.bias_add(output, biases)
        if activation_fn is not None:
            output = activation_fn(output, name=scope.name)

        return output


def Deconv2d(inputs, output_shape, name, k_h, k_w, s_h=2, s_w=2, reuse=None, activation_fn=None, biased=False):
    output_shape[0] = inputs.get_shape()[0]
    ts_output_shape = tf.stack(output_shape)
    w_init = tf.random_normal_initializer(stddev=0.02)

    deconvolve = lambda i, k: tf.nn.conv2d_transpose(i, k, output_shape=ts_output_shape, strides=[1, s_h, s_w, 1])
    with tf.variable_scope(name, reuse=reuse) as scope:
        kernel = tf.get_variable(name='weights', shape=[k_h, k_w, output_shape[-1], inputs.get_shape()[-1]],
                                 initializer=w_init)
        output = deconvolve(inputs, kernel)

        if biased:
            biases = tf.get_variable(name='biases', shape=[output_shape[-1]])
            output = tf.nn.bias_add(output, biases)
        if activation_fn is not None:
            output = activation_fn(output, name=scope.name)

        deconv = tf.reshape(output, [-1] + output_shape[1:])

        return deconv


def add(inputs, name):
    return tf.add_n(inputs, name=name)


def UpSample(inputs, size, method, align_corners, name):
    return tf.image.resize_images(inputs, size, method, align_corners)


def flatten(inputs, name):
    input_shape = inputs.get_shape()
    dim = 1
    for d in input_shape[1:].as_list():
        dim *= d
        inputs = tf.reshape(inputs, [-1, dim])

    return inputs


# class conv_batch_norm(pt.VarStoreMethod):
#     """Code modification of:
#      http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
#      and
#      https://github.com/tensorflow/models/blob/master/inception/inception/slim/ops.py"""
#
#     def __call__(self, input_layer, epsilon=1e-5, decay=0.9, name="batch_norm", in_dim=None, phase=Phase.train):
#         shape = input_layer.shape
#         shp = in_dim or shape[-1]
#
#         self.gamma = self.variable("gamma", [shp], init=tf.random_normal_initializer(1., 0.02))
#         self.beta = self.variable("beta", [shp], init=tf.constant_initializer(0.))
#
#         self.mean, self.variance = tf.nn.moments(input_layer, [0, 1, 2])
#         # sigh...tf's shape system is so..
#         self.mean.set_shape((shp,))
#         self.variance.set_shape((shp,))
#
#         normalized_x = tf.nn.batch_normalization(input_layer, self.mean, self.variance, None, None, epsilon)
#         return input_layer.with_tensor(normalized_x, parameters=self.vars)
#
#     # def __call__(self, input_layer, epsilon=1e-5, momentum=0.1, name="batch_norm",
#     #              in_dim=None, phase=Phase.train):
#     #     self.ema = tf.train.ExponentialMovingAverage(decay=0.9)
#     #
#     #     shape = input_layer.shape
#     #     shp = in_dim or shape[-1]
#     #     with tf.variable_scope (name) as scope:
#     #         self.gamma = self.variable ("gamma", [shp], init=tf.random_normal_initializer (1., 0.02))
#     #         self.beta = self.variable ("beta", [shp], init=tf.constant_initializer (0.))
#     #
#     #         self.mean, self.variance = tf.nn.moments(input_layer.tensor, [0, 1, 2])
#     #         # sigh...tf's shape system is so..
#     #         self.mean.set_shape ((shp,))
#     #         self.variance.set_shape ((shp,))
#     #         self.ema_apply_op = self.ema.apply ([self.mean, self.variance])
#     #
#     #         if phase == Phase.train:
#     #             with tf.control_dependencies ([self.ema_apply_op]):
#     #                 normalized_x = tf.nn.batch_norm_with_global_normalization (
#     #                     input_layer.tensor, self.mean, self.variance, self.beta, self.gamma, epsilon,
#     #                     scale_after_normalization=True)
#     #         else:
#     #             normalized_x = tf.nn.batch_norm_with_global_normalization (
#     #                 x, self.ema.average (self.mean), self.ema.average (self.variance), self.beta,
#     #                 self.gamma, epsilon,
#     #                 scale_after_normalization=True)
#     #
#     #         return input_layer.with_tensor (normalized_x, parameters=self.vars)
#
# #
# pt.Register(assign_defaults=('phase'))(conv_batch_norm)
#
#
# @pt.Register(assign_defaults=('phase'))
# class fc_batch_norm(conv_batch_norm):
#     def __call__(self, input_layer, *args, **kwargs):
#         ori_shape = input_layer.shape
#         if ori_shape[0] is None:
#             ori_shape[0] = -1
#         new_shape = [ori_shape[0], 1, 1, ori_shape[1]]
#         x = tf.reshape(input_layer.tensor, new_shape)
#         normalized_x = super(self.__class__, self).__call__(input_layer.with_tensor(x), *args, **kwargs)  # input_layer)
#         return normalized_x.reshape(ori_shape)
#
#
# def leaky_rectify(x, leakiness=0.01):
#     assert leakiness <= 1
#     ret = tf.maximum(x, leakiness * x)
#     # import ipdb; ipdb.set_trace()
#     return ret
#
#
# @pt.Register
# class custom_conv2d(pt.VarStoreMethod):
# # class custom_conv2d(object):
#     def __call__(self, input_layer, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, in_dim=None, padding='SAME',
#                  name="conv2d"):
#         with tf.variable_scope(name):
#             w = self.variable('w', [k_h, k_w, in_dim or input_layer.shape[-1], output_dim],
#                                init=tf.truncated_normal_initializer(stddev=stddev))
#             # w = tf.get_variable('w', shape=[k_h, k_w, in_dim or input_layer.shape[-1], output_dim], dtype=tf.float32,
#             #                   initializer=tf.truncated_normal_initializer(stddev=stddev))
#             conv = tf.nn.conv2d(input_layer.tensor, w, strides=[1, d_h, d_w, 1], padding=padding)
#
#             # biases = self.variable('biases', [output_dim], init=tf.constant_initializer(0.0))
#             # import ipdb; ipdb.set_trace()
#             # return input_layer.with_tensor(tf.nn.bias_add(conv, biases), parameters=self.vars)
#             return input_layer.with_tensor(conv, parameters=self.vars)
#
#
# @pt.Register
# class custom_deconv2d(pt.VarStoreMethod):
# # class custom_deconv2d(object):
#     def __call__(self, input_layer, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d"):
#         output_shape[0] = input_layer.shape[0]
#         ts_output_shape = tf.stack(output_shape)
#         with tf.variable_scope(name):
#             # filter : [height, width, output_channels, in_channels]
#             w = self.variable('w', [k_h, k_w, output_shape[-1], input_layer.shape[-1]],
#                                init=tf.random_normal_initializer(stddev=stddev))
#
#             # w = tf.get_variable('w', shape=[k_h, k_w, output_shape[-1], input_layer.shape[-1]], dtype=tf.float32,
#             #                   initializer=tf.random_normal_initializer(stddev=stddev))
#
#             try:
#                 deconv = tf.nn.conv2d_transpose(input_layer, w, output_shape=ts_output_shape, strides=[1, d_h, d_w, 1])
#
#             # Support for versions of TensorFlow before 0.7.0
#             except AttributeError:
#                 deconv = tf.nn.deconv2d(input_layer, w, output_shape=ts_output_shape,
#                                         strides=[1, d_h, d_w, 1])
#
#             # biases = self.variable('biases', [output_shape[-1]], init=tf.constant_initializer(0.0))
#             # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), [-1] + output_shape[1:])
#             deconv = tf.reshape(deconv, [-1] + output_shape[1:])
#
#             return deconv
#
#
# @pt.Register
# class custom_fully_connected(pt.VarStoreMethod):
# # class custom_fully_connected(object):
#     def __call__(self, input_layer, output_size, scope=None, in_dim=None, stddev=0.02, bias_start=0.0):
#         shape = input_layer.shape
#         input_ = input_layer.tensor
#         try:
#             if len(shape) == 4:
#                 input_ = tf.reshape(input_, tf.stack([tf.shape(input_)[0], np.prod(shape[1:])]))
#                 input_.set_shape([None, np.prod(shape[1:])])
#                 shape = input_.get_shape().as_list()
#
#             with tf.variable_scope(scope or "Linear"):
#                 matrix = self.variable("Matrix", [in_dim or shape[1], output_size], dt=tf.float32,
#                                        init=tf.random_normal_initializer(stddev=stddev))
#                 bias = self.variable("bias", [output_size], init=tf.constant_initializer(bias_start))
#                 return input_layer.with_tensor(tf.matmul(input_, matrix) + bias, parameters=self.vars)
#         except Exception:
#             import ipdb; ipdb.set_trace()
