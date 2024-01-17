'''
TO DO : Finish the improvement
'''

import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
from scipy import stats
import json
import time

class Config:
    """
    Configuration class for image processing parameters.
    """
    # HSV values for green detection
    lower_green = np.array([30, 30, 30]) 
    upper_green = np.array([70, 255, 255]) 

    # Parameters for Histogram of Oriented Gradients (HOG)
    winSize = (48, 96)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9

    # Kernels for image processing
    erosion_kernel = np.ones((7, 7), np.uint8)
    dilatation_kernel = np.ones((13, 13), np.uint8)
    blur_kernel = (15, 15)

    # Standard RGB colors for yellow and red
    yellow_standard = np.array([255, 255, 0])
    red_standard = np.array([255, 0, 0])

# Utility functions
def read_image(path):
    """
    Reads an image from a specified path.

    Parameters:
    path (str): Path to the image file.

    Returns:
    numpy.ndarray: The image read from the file.
    """
    return cv2.imread(path)

def save_json(data, path):
    """
    Saves data in JSON format to a specified path.

    Parameters:
    data (dict): Data to be saved.
    path (str): Path to save the JSON file.
    """
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

# Other utility functions could include image processing, color analysis, etc.
# ...


original = cv2.imread('data/img/frame.png')
image = original.copy()

video_path = 'data/video/Bundes short.mp4'
json_path = 'results/json/players_detected.json'

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