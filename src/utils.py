import numpy as np
import cv2
import imutils
import argparse
from imutils.object_detection import non_max_suppression

# Parametri dei kernel per il pre-processing dell'immagine in court detection
lower_green = np.array([30, 30, 30])            # Valori minimi di H, S, V per il verde
upper_green = np.array([70, 255, 255])          # Valori massimi di H, S, V per il verde
blur_kernel = (15, 15)                          #Kernel da applicare per il denoising con blur
erosion_kernel = np.ones((7, 7), np.uint8)      #kernel da applicare per l'erosione
dilatation_kernel = np.ones((13, 13), np.uint8) #Kernel da applciare per la dilatazione


#--- PARAMETRI INIZIALI PER HOG CON DAIMLER DATASET ---
winSize = (48, 96)  # Esempio, da adattare in base alle specifiche di Daimler
blockSize = (16, 16)  # Esempio
blockStride = (8, 8)  # Esempio
cellSize = (8, 8)  # Esempio
nbins = 9  # Esempio
