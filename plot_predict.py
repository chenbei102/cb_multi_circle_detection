# -*- coding: utf-8 -*-

"""
Visualize the results of circular area detection

"""

from utils.dataset_utils import generate_sample
from utils.noise_scheduler import LinearNoiseScheduler
from utils.evaluation_utils import find_circle_centers

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
import os
import cv2



if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # parse YAML file to extract configuration parameters

    with open("config.yml", 'r') as file:
        config = yaml.safe_load(file)

    common_config = config['common']

    work_dir = common_config['work_dir']

    img_w = common_config['image_shape']['width']
    img_h = common_config['image_shape']['height']
    
    if 'noise_scheduler' in config:
        noise_config = config['noise_scheduler']
        noise = LinearNoiseScheduler(noise_config['num_timesteps'],
                                     noise_config['beta_start'],
                                     noise_config['beta_end'])
        noise_steps = noise_config['noise_steps']
    else:
        noise = None
        noise_steps = 0

    model_config = config['cnn_model']

    # -------------------------------------------------------------------------
    # load the deep convolutional neural neural

    if model_config['from_checkpoint']:
        model_fname = model_config['checkpoint_filename']
    else:
        model_fname = model_config['model_filename']

    model = tf.keras.models.load_model(os.path.join(work_dir, model_fname),
                                       compile=False)
    model.summary()

    # -------------------------------------------------------------------------
    # create a sample image and perform circular area detection

    img_in, _, _ = generate_sample(np.random.randint(3, 6), img_w, img_h)
    
    pred_img = model.predict(np.expand_dims(img_in, axis=0))

    pred_centers = find_circle_centers(pred_img)

    # -------------------------------------------------------------------------
    # visualize the detected circular areas 

    im = np.array(img_in * 255.0 / img_in.max(), dtype = np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    R1 = int(np.ceil(0.08 * img_w))

    for x, y in pred_centers[0]:
        cv2.circle(im, (x, y), 1, (0,255,0), 1)
        cv2.circle(im, (x, y), R1, (0,255,0), 1)

    figure, ax = plt.subplots()

    ax.imshow(im)

    plt.show()
