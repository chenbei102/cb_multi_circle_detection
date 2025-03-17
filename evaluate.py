# -*- coding: utf-8 -*-

"""
Evaluate the performance of the hybrid deep learning and feature detection model.

"""

from utils.evaluation_utils import find_circle_centers, count_nearby_coords

import numpy as np
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
    # detect circular areas in each image of the testing dataset

    dataset_test = tf.data.Dataset.load(os.path.join(work_dir, "dataset_test"))
    
    pred_img = model.predict(dataset_test)

    pred_centers = find_circle_centers(pred_img)

    # -------------------------------------------------------------------------
    # compare the detected center coordinates with target ones

    fname = os.path.join(work_dir, "centers_test.json")

    with open(fname, 'r') as fh:
        center_lst = json.load(fh)

    assert len(pred_centers) == len(center_lst), "Dataset length mismatch."

    num_circles = 0
    cnt_detected = 0
    false_positives = 0

    for i in range(len(pred_centers)):

        num = len(center_lst[i])
        cc, fp = count_nearby_coords(pred_centers[i], np.array(center_lst[i], dtype=int))

        num_circles += num 
        cnt_detected += cc
        false_positives += fp

    # calculate detection statistics for circular area detection.         

    detection_rate = (cnt_detected / num_circles) * 100.0

    precision = cnt_detected / (cnt_detected + false_positives) 
    recall = cnt_detected / num_circles 
    f1_score = (2 * precision * recall) / (precision + recall) 

    print(f"{cnt_detected} out of {num_circles} circular areas were detected.")
    print(f"Missed: {num_circles - cnt_detected}")
    print(f"False Positives: {false_positives}")
    print(f"The detected rate is {detection_rate:.2f} %.")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
