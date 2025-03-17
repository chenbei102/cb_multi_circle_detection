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


def compare_coordinates(coord1, coord2, tolerance):
    """
    Compare two coordinates (x, y) based on their distance and relative
    ordering.

    Args:
        coord1 (numpy.ndarray)): The first coordinate as (x1, y1).
        coord2 (numpy.ndarray): The second coordinate as (x2, y2).
        tolerance (int): The maximum allowable distance for the coordinates to
            be considered equal.

    Returns:
        int: 
        - 0 if the Euclidean distance between the coordinates is within the
            given tolerance.
        - 1 if coord1 is greater than coord2 in lexicographic order (x1 > x2 or,
            if x1 == x2, then y1 > y2).
        - -1 if coord1 is smaller than coord2 in lexicographic order (x1 < x2 or,
            if x1 == x2, then y1 < y2).
    """

    dx = coord1[0] - coord2[0]
    dy = coord1[1] - coord2[1]
    d2 = dx*dx + dy*dy
    R2 = tolerance * tolerance

    if (d2 <= R2):
        flag = 0
    elif (dx > 0):
        flag = 1
    elif (dx < 0):
        flag = -1
    elif (dy > 1):
        flag = 1
    elif (dy < 1):
        flag = -1

    return flag


def count_nearby_coords(coord1, coord2, tolerance=3):
    """
    count the number of detected coordinates that match the target coordinates
    within a given tolerance.

    This function compares a list of detected coordinates (`coord1`) against a
    list of target coordinates (`coord2`). It determines how many detected
    coordinates are within the specified `tolerance` of any target coordinate.
    Also, it counts the number of false positives—coordinates in detected
    coordinates that do not match any target coordinate.

    Args:
        coord1 (numpy.ndarray): A 2D array where each row represents (x, y)
            coordinates of detected points.
        coord2 (numpy.ndarray): A 2D array where each row represents (x, y)
            coordinates of target points.
        tolerance (int, optional): The maximum allowed distance between a
            detected coordinate and a target coordinate for them to be
            considered a match (default: 3).

    Returns:
        tuple: A tuple containing:
        - cnt_detected (int): The number of detected coordinates that match
            target coordinates (true positives).
        - false_positives (int): The number of false positives (detected
            coordinates that do not match any target).
    """

    N1 = len(coord1)
    if (1 == N1) and (0 > coord1[0, 0]):
        return 0, 0

    N2 = len(coord2)

    flag_coord1 = np.ones(N1, dtype=int)
    
    cnt_detected = 0
    false_positives = 0

    i = 0
    j = 0

    while j < N2:

        # Implement a variant of the binary search algorithm to efficiently
        # locate matching coordinates

        is_found = False
        l = i
        r = N1-1
        while l <= r:
            m = (l + r) // 2
            flag = compare_coordinates(coord1[m, :], coord2[j, :], tolerance)
            if 0 == flag:
                is_found = True
                flag_coord1[m] = 0
                i = m
                break
            elif 1 == flag:
                r = m-1
            else:
                l = m+1
                
        if is_found:
            cnt_detected += 1
        else:
            for m in range(N1):
                if 0 == compare_coordinates(coord1[m, :], coord2[j, :], tolerance):
                    is_found = True
                    flag_coord1[m] = 0
                    cnt_detected += 1
                    break

        j += 1

    for i in np.asarray(0 != flag_coord1).nonzero()[0]:
        for j in range(N2):
            if 0 == compare_coordinates(coord1[i, :], coord2[j, :], tolerance):
                flag_coord1[i] = 0
                break

    false_positives = np.sum(flag_coord1)

    return cnt_detected, false_positives
