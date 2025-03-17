# -*- coding: utf-8 -*-

"""
Generate a plot of the training and validation loss to visualize the training
progress of a deep neural network model.

"""

import numpy as np
import matplotlib.pyplot as plt
import json
import yaml
import os



def plot_hist(history, title_str=None):
    
    fig, ax = plt.subplots()

    ax.plot(history['loss'])
    ax.plot(history['val_loss'])
    
    ax.set_yscale('log')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if title_str is not None:
        ax.set_title(title_str)
    else:
        ax.set_title('Training progress')

    ax.grid(linewidth=0.5)
    ax.legend(['Training', 'Validation'])

    

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # parse YAML file to extract configuration parameters

    with open("config.yml", 'r') as file:
        config = yaml.safe_load(file)

    common_config = config['common']

    work_dir = common_config['work_dir']
    
    # -------------------------------------------------------------------------
    # plot the training loss over epochs

    fname = os.path.join(work_dir, "history.json")

    with open(fname, 'r') as fh:
        hist = json.load(fh)

    plot_hist(hist)

    plt.show()
