# -*- coding: utf-8 -*-

"""
This module contains utility functions for evaluating the deep learning model.

"""

import numpy as np
import cv2


def find_circle_centers(images):
    """
    Find the center coordinates of circles in the given images.

    Args:
        images (numpy.ndarray): A NumPy array representing the input images.

    Returns:
        list of numpy.ndarray: A list where each element is a NumPy array
            containing the (x, y) coordinates of detected circle centers in
            the corresponding image. If no circles are found, return a
            single-element array `[[-100, -100]]`
    """

    L_min = np.min(images.shape[1:3])
    L_max = np.max(images.shape[1:3])
    
    R1 = int(np.ceil(0.01 * L_min))
    R2 = int(np.ceil(0.10 * L_max))
          
    center_coords = []

    for i in range(len(images)):
    
        img = images[i, :, :, :]
        im = np.array(img * 255.0 / img.max(), dtype = np.uint8)
        im = cv2.GaussianBlur(im, (5, 5), 2)

        # detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, 1, R1,
                                   param1=50, param2=30, minRadius=R1, maxRadius=R2)

        if circles is not None:
            coords = np.array(circles[0, :, :2], dtype=int)
            coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
            center_coords.append(coords)
        else:
            center_coords.append(np.array([[-100, -100]], dtype=int))

    return center_coords
