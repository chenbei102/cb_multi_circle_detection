# -*- coding: utf-8 -*-

"""
This module provides functions for building deep neural networks using TensorFlow.

"""

from tensorflow import keras


def create_deep_cnn(input_shape, filter_lst, kernel_lst):
    """
    Construct a deep convolutional neural network (CNN) using the specified
    parameters.

    Args:
        input_shape (tuple): The shape of the input images, given as
            (height, width, channels).
        filter_lst (list of int): A list specifying the number of filters for
            each convolutional layer.
        kernel_lst (list of int or tuple): A list specifying the kernel size
            for each convolutional layer.

    Returns:
        model (tf.keras.Model): A deep CNN model.
    """

    assert len(filter_lst) == len(kernel_lst), \
        "'filter_lst' and 'kernel_lst' have different lengths."

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))

    for ff, kk in zip(filter_lst, kernel_lst):
        model.add(keras.layers.Conv2D(ff, kk, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Conv2D(1, kernel_lst[-1], padding='same', activation='relu'))

    return model
