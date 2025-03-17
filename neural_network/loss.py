# -*- coding: utf-8 -*-

"""
This module defines custom loss functions.

"""

import tensorflow as tf


tf.keras.utils.get_custom_objects().clear()


@tf.keras.utils.register_keras_serializable()
def loss_function(y_true, y_pred):

    squared_diff = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_diff)
