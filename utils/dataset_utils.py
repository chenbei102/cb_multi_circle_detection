# -*- coding: utf-8 -*-

"""
This module contains utility functions for creating datasets used in deep learning model training. 

"""

import numpy as np
import cv2



def random_gray():
    """
    Generate a random grayscale value.
    """
    return np.ones(3) * (np.random.randint(25, high=256) / 255)


def generate_coordinates(N, R, height, width):
    """
    Generates N random 2D coordinates with minimum distance R.

    Args:
        N: Number of coordinates to generate.
        R: Minimum distance between any two coordinates.
        height: Height of the image.
        width: Width of the image.

    Returns:
        A NumPy array of shape (N, 2) containing the generated coordinates,
        or None if generation fails.
    """

    if N <= 0 or R < 0:
        return None

    if (0 >= height) or (0 >= width):
        return None

    cnt = 0
    attempts = 0
    max_attempts = 100

    coord = np.zeros((N, 2), dtype=int)

    x = np.zeros(2*N, dtype=int)
    y = np.zeros(2*N, dtype=int)

    R2 = R*R

    while cnt < N:
        if attempts > max_attempts:
            return None  # Generation failed

        x = np.random.randint(2*R, high=width-2*R, size=2*N)
        y = np.random.randint(2*R, high=height-2*R, size=2*N)
        
        for xx, yy in zip(x, y):
            valid = True
            for i in range(cnt):
                dx = coord[i, 0] - xx
                dy = coord[i, 1] - yy
                if dx*dx + dy*dy < R2:
                    valid = False
                    break
                
            if valid:
                coord[cnt, 0] = xx
                coord[cnt, 1] = yy
                cnt += 1
                if cnt == N:
                    break
                
        attempts += 1

    return coord


def generate_sample(num_circles, height, width):
    """
    Generates a sample consisting of an input image, an output image,
    and a numpy array containing the coordinates of the centers of all circular
    areas.

    Args:
        num_circles (int): The number of circles to generate in the input image.
        height (int): The height of the generated input and output images.
        width (int): The width of the generated input and output images.

    Returns:
        tuple: A tuple containing:
        - img_in (numpy.ndarray): A 2D array representing the input image.
        - img_out (numpy.ndarray): A 2D array representing the output image. 
        - coords (numpy.ndarray): A 2D array of shape (num_circles, 2), where
          each row contains the (x, y) coordinates of the centers of a circular
          areas.
    """

    img_in = np.zeros((height, width, 1))
    img_out = np.zeros((height, width, 1))

    R1 = int(np.ceil(0.08 * width))
    
    coords = generate_coordinates(num_circles, int(0.5*R1), height, width)
    for x, y in coords:
        cv2.circle(img_in, (x, y), R1, random_gray(), cv2.FILLED)
        cv2.circle(img_out, (x, y), R1, (1,1,1), 2)

    coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]

    return img_in, img_out, coords
