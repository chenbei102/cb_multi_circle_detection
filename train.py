# -*- coding: utf-8 -*-

"""
Train the deep neural network model

"""

from neural_network.model import create_deep_cnn

import tensorflow as tf
import yaml
import os
import json



if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # parse YAML file to extract configuration parameters

    with open("config.yml", 'r') as file:
        config = yaml.safe_load(file)

    common_config = config['common']

    work_dir = common_config['work_dir']

    input_shape = (common_config['image_shape']['height'], common_config['image_shape']['width'], 1)

    model_config = config['cnn_model']

    # -------------------------------------------------------------------------
    # load or construct a deep convolutional neural neural

    if model_config['resume']:

        if model_config['from_checkpoint']:
            model_fname = model_config['checkpoint_filename']
        else:
            model_fname = model_config['model_filename']

        model = tf.keras.models.load_model(os.path.join(work_dir, model_fname))

    else:

        filter_lst = model_config['filters']
        kernel_lst = model_config['kernel_sizes']
        lr = model_config['learning_rate']

        model = create_deep_cnn(input_shape, filter_lst, kernel_lst, lr)

    model.summary()

    # -------------------------------------------------------------------------
    # establish the checkpointing strategy

    checkpoint_filepath = os.path.join(work_dir, model_config['checkpoint_filename'])

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # -------------------------------------------------------------------------
    # load datasets for training and validation

    dataset_train = tf.data.Dataset.load(os.path.join(work_dir, "dataset_train"))
    dataset_valid = tf.data.Dataset.load(os.path.join(work_dir, "dataset_valid"))

    # -------------------------------------------------------------------------
    # train the deep neural network model 

    num_epochs = model_config['num_epochs']
 
    hist = model.fit(dataset_train, epochs=num_epochs,
                     validation_data=dataset_valid,
                     callbacks=[model_checkpoint_callback])

    # -------------------------------------------------------------------------
    # store the model and its training history

    model.save(os.path.join(work_dir, model_config['model_filename']))

    with open(os.path.join(work_dir, "history.json"), 'w') as fh:
        json.dump(hist.history, fh)
