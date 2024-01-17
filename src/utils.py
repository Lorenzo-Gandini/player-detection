import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
from scipy import stats
import json
import time


original = cv2.imread('data/img/frame.png')
image = original.copy()

video_path = 'data/video/Bundes short.mp4'
json_path = 'results/json/players_detected.json'

## --- START VARIABLES BLOCK ---
# Green bounds for court removing - HSV values
lower_green = np.array([30, 30, 30]) 
upper_green = np.array([70, 255, 255]) 

# Parameters of Daimler HOG
winSize = (48, 96)  
blockSize = (16, 16)  
blockStride = (8, 8)  
cellSize = (8, 8)  
nbins = 9  

# Kernels for court operations
erosion_kernel = np.ones((7, 7), np.uint8) #kernel da applicare per l'erosione
dilatation_kernel = np.ones((13, 13), np.uint8) #Kernel da applciare per la dilatazione
blur_kernel = (15, 15) #Kernel da applicare per il denoising

# Definisci i colori standard RGB per giallo e rosso
yellow_standard = np.array([255, 255, 0])
red_standard = np.array([255, 0, 0])

# ++++ Parameters for CAMShift
# first interval of red (0-10)
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([20, 255, 255])

# Second interval for red (160-180)
lower_red2 = np.array([160, 70, 50])
upper_red2 = np.array([180, 255, 255])

#Yellow interval for mask
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])