# Hybrid Deep Learning and Feature Detection for Locating Multiple Circular Areas

This repository explores an approach that combines deep learning with conventional feature detection techniques to locate multiple circular areas in an image.  

## Problem Formulation  

Given a grayscale image containing $N_c$ circular bright spot areas, where:  
- $N_c$ is a random number greater than 1.  
- The circular areas vary in brightness.  
- Their positions are randomly distributed within the image domain.  
- The image may contain background noise.

The objective is to develop a model that, when given an input image, accurately returns a list of coordinates corresponding to all detected circular areas.  

## Model Construction Approach  

The process of locating circular areas consists of two key steps:  

1. **Image Denoising and Brightness Normalization**  
   A deep convolutional neural network (CNN) is utilized to denoise the image and transform circular regions of varying brightness into multiple circles with nearly uniform brightness.  

2. **Center Coordinate Extraction**  
   The Circle Hough Transform is applied to extract the coordinates of the centers of these multi-circles.  

![model_illustration](fig/model_illustration.png)

<p align="center"><strong>Illustration of the model for detecting multiple circular areas</strong></p>
