'''
This file contains the imports, the declaration of variables, the kernels and the settings utilities for the project.
'''

import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
from scipy import stats
import json
import emoji
import time
import os

#Paths
video_path = 'data/video/Bundes clip.mp4'
bboxes_path = 'data/bounding-boxes/frame.json' 
HOG_path = 'data/HOG-detected/frame.json'
results_video_path = 'results/video/player-detection.avi' 

#Initialization of variables and tresholds
kalman_filters = {}
hog_tracking_list = []
kalman_tracking_list = []
frame_counter = 0
actual_player_id = 0
area_treshold = 6000
centroid_threshold = 10  
iou_threshold = 0.95 

#Parameters for HOG
hog_winstride = (4,4)
hog_padding = (4,4)
hog_scale = 1.001

# Parameters of Daimler HOG
winSizeD = (48, 96)  
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9

# Kernels for court operations
erosion_kernel = np.ones((7, 7), np.uint8) #kernel for erosion
dilatation_kernel = np.ones((13, 13), np.uint8) #Kernel for dilatation
blur_kernel = (19, 19) #Kernel for blur denoising

#RGB Values for yellow and red
yellow_standard = np.array([255, 255, 0])
red_standard = np.array([255, 0, 0])

# Green bounds for court removing - HSV values
lower_green = np.array([35, 30, 30]) 
upper_green = np.array([70, 255, 255]) 

# first interval of hsv red
lower_red1 = np.array([0, 60, 40])
upper_red1 = np.array([25, 255, 255])

# Second interval for hsv red
lower_red2 = np.array([150, 100, 70])
upper_red2 = np.array([180, 255, 255])

# Yellow interval for hsv mask
lower_yellow = np.array([20, 80, 80])
upper_yellow = np.array([30, 255, 255])

