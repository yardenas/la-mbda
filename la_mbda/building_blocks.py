import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd


def decoder(type_, shape, layers, units):
    if type_ == 'rgb_image':
        return ConvDecoder(shape, dist='normal')
    elif type_ == 'binary_image':
        return ConvDecoder(shape, dist='bernoulli')
    elif type_ == 'dense':
        return DenseDecoder(shape, layers, units)


def encoder(type_, shape, layers, units):
    if type_ == 'rgb_image' or type_ == 'binary_image':
        return ConvEncoder(shape)
    elif type_ == 'dense':
        return DenseEncoder(layers, units)


class ConvDecoder(tf.Module):
    def __init__(self, shape, depth=32, activation=tf.nn.relu, dist='normal'):
        super(ConvDecoder, self).__init__()
        self._shape = shape
        self._dist = dist
        self._depth = depth
        self._dense = tf.keras.layers.Dense(32 * depth)
        self._layers = tf.keras.Sequential(
            [tf.keras.layers.Conv2DTranspose(4 * depth, 5, activation=activation, strides=2),
             tf.keras.layers.Conv2DTranspose(2 * depth, 5, activation=activation, strides=2),
             tf.keras.layers.Conv2DTranspose(depth, 6, activation=activation, strides=2),
             tf.keras.layers.Conv2DTranspose(self._shape[-1], 6, strides=2)])

    def __call__(self, inputs):
        x = self._dense(inputs)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
        x = self._layers(x)
        if tf.reduce_any(tf.shape(x)[-3:] != self._shape):
            x = tf.keras.layers.UpSampling2D()(x)
        x = tf.reshape(x, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
        if self._dist == 'normal':
            return tfd.Independent(tfd.Normal(x, 1.0), len(self._shape))
        elif self._dist == 'bernoulli':
            return tfd.Independent(tfd.Bernoulli(x, dtype=tf.float32), len(self._shape))


class ConvEncoder(tf.Module):
    def __init__(self, shape, depth=32, activation=tf.nn.relu):
        super(ConvEncoder, self).__init__()
        self._shape = shape
        self._depth = depth
        self._layers = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(depth, 4, activation=activation, strides=2),
             tf.keras.layers.Conv2D(2 * depth, 4, activation=activation, strides=2),
             tf.keras.layers.Conv2D(4 * depth, 4, activation=activation, strides=2),
             tf.keras.layers.Conv2D(8 * depth, 4, activation=activation, strides=2)])

    def __call__(self, inputs):
        x = tf.reshape(inputs, (-1,) + tuple(inputs.shape[-3:]))
        x = self._layers(x)
        x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
        shape = tf.concat([tf.shape(inputs)[:-3], [x.shape[-1]]], 0)
        return tf.reshape(x, shape)


class DenseDecoder(tf.Module):
    def __init__(self, shape, layers, units, activation=tf.nn.relu, dist='normal'):
        super(DenseDecoder, self).__init__()
        self._layers = tf.keras.Sequential(
            [tf.keras.layers.Dense(units, activation) for _ in range(layers)] +
            [tf.keras.layers.Dense(np.prod(shape))])
        self._dist = dist
        self._shape = shape

    def __call__(self, inputs):
        x = self._layers(inputs)
        x = tf.reshape(x, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
        if self._dist == 'normal':
            return tfd.Independent(tfd.Normal(x, 1.0), len(self._shape))
        elif self._dist == 'bernoulli':
            return tfd.Independent(tfd.Bernoulli(x, dtype=tf.float32), len(self._shape))


class DenseEncoder(tf.Module):
    def __init__(self, layers, units, activation=tf.nn.relu):
        super().__init__()
        self._layers = tf.keras.Sequential(
            [tf.keras.layers.Dense(units, activation) for _ in range(layers)])

    def __call__(self, inputs):
        return self._layers(inputs)
