# -*- coding: utf-8 -*-

"""
Generate TensorFlow datasets for training, validation, and testing.

"""

from utils.dataset_utils import generate_sample
from utils.noise_scheduler import LinearNoiseScheduler

import numpy as np
import tensorflow as tf
import yaml
import os
import json



def create_dataset(img_width: int = 128,
                   img_height: int = 128,
                   dataset_size: int = 1024,
                   batch_size: int = 32,
                   shuffle: bool = True,
                   noise=None,
                   noise_steps=0):
    """
    Create a TensorFlow dataset containing input and output images, along with
    a list of center coordinates for circular regions within the images.

    Args:
        img_width (int, optional): The width of each image (default: 128).
        img_height (int, optional): The height of each image (default: 128).
        dataset_size (int, optional): The total number of image pairs in the
            dataset (default: 1024).
        batch_size (int, optional): The number of image pairs per batch (default: 32).
        shuffle (bool, optional): Whether to shuffle the dataset (default: True).
        noise (optional): A noise scheduler used for adding noise to the images.
        noise_steps (int, optional): The corresponding timestep in the diffusion model
            that determines the noise level (default: 0).

    Returns:
        tuple: A tuple containing:
        - dataset (tf.data.Dataset): A TensorFlow dataset object.
        - c_list (list): A list of `dataset_size` elements, where each element represents
              the coordinates of the center of a circular region in an image.
    """

    x_list = []
    y_list = []
    c_list = []
    for i in range(dataset_size):
        x, y, c  = generate_sample(np.random.randint(1, high=6),
                                   img_height, img_width)
        x_list.append(x)
        y_list.append(y)
        c_list.append(c.tolist())
    
    x_arr = np.array(x_list)
    y_arr = np.array(y_list)

    if noise is not None:
        x_arr = noise.add_noise(x_arr, noise_steps)
    
    dataset = tf.data.Dataset.from_tensor_slices((x_arr, y_arr))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=dataset_size)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset, c_list


def save_list2json(data_list, filename):
    """
    Save a Python list to a JSON file.

    Args:
        data_list (list): The list to be saved.
        filename (str): The name of the JSON file to create.
    """
    try:
        with open(filename, 'w') as json_file:
            json.dump(data_list, json_file)
    except Exception as e:
        print(f"An error occurred while saving to {filename}: {e}")

        

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # parse YAML file to extract configuration parameters

    with open("config.yml", 'r') as file:
        config = yaml.safe_load(file)

    common_config = config['common']

    work_dir = common_config['work_dir']

    if 'noise_scheduler' in config:
        noise_config = config['noise_scheduler']
        noise = LinearNoiseScheduler(noise_config['num_timesteps'],
                                     noise_config['beta_start'],
                                     noise_config['beta_end'])
        noise_steps = noise_config['noise_steps']
    else:
        noise = None
        noise_steps = 0

    # -------------------------------------------------------------------------
    # create datasets for training, validation, and testing

    dataset_train, _ = create_dataset(noise=noise, noise_steps=noise_steps)
    dataset_valid, _ = create_dataset(dataset_size=512, noise=noise, noise_steps=noise_steps)
    dataset_test, centers_test = create_dataset(shuffle=False, noise=noise, noise_steps=noise_steps)

    # save datasets 

    dataset_train.save(os.path.join(work_dir, "dataset_train"))
    dataset_valid.save(os.path.join(work_dir, "dataset_valid"))
    dataset_test.save(os.path.join(work_dir, "dataset_test"))

    # save the center coordinates of circular areas in the testing dataset for
    # model evaluation

    save_list2json(centers_test, os.path.join(work_dir, "centers_test.json"))
    
