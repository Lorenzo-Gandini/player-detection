import numpy as np
import cv2
import imutils
import argparse
from imutils.object_detection import non_max_suppression
from scipy import stats
import time

original = cv2.imread('data/img/frame.png')
image = original.copy()

## --- START VARIABLES BLOCK ---
# Green bounds for court removing
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
