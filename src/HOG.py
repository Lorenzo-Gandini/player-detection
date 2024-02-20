'''
Functions used for identify new players with HOG.
'''

from utilities import *
from functions import extract_court, apply_green_mask, save_new_coordinates

def preprocess_image(image):
    '''
    Apply to the image some operations in order to facilitate the recognition of players with HOG. 
    Court identification -> Mask of the court -> dilates the mask to catch also players near lines -> Blur on Roi -> HSV -> Apply mask to roi -> 
    Saturate roi -> Enhance Roi -> Image ready to be analyzed with HOG.

    Args:
    image : The frame to enhance

    Returns
    contrast_enhanced : The enhanced image of field where look for players 
    '''

    max_contour = extract_court(image)

    if max_contour is not None:  
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [max_contour], 255)

        # Dilatate the mask, in roder to increase the area since some players can be near the line of the field.
        kernel = np.ones((10, 10), np.uint8) 
        dilated_mask = cv2.dilate(mask, kernel)

        # Since we dilatate the mask, the dimension of the image must be the same
        mask_resized = cv2.resize(dilated_mask, (image.shape[1], image.shape[0]))
        if mask_resized.dtype != np.uint8:
            mask_resized = mask_resized.astype(np.uint8)

        # Extract the roi from the mask
        roi = cv2.bitwise_and(image, image, mask=mask_resized)

    denoised_roi = cv2.GaussianBlur(roi, (7,7), 0)
    hsv_roi = cv2.cvtColor(denoised_roi, cv2.COLOR_BGR2HSV)
    roi_green_mask = apply_green_mask(denoised_roi)
    
    #Luminance and saturation will be reduce in order to saturate the field and facilitate the detection with HOG.
    hsv_roi[:, :, 2] = np.where(roi_green_mask, hsv_roi[:, :, 2] * 0.5, hsv_roi[:, :, 2])  
    hsv_roi[:, :, 1] = np.where(roi_green_mask, hsv_roi[:, :, 1] * 0.5, hsv_roi[:, :, 1])  

    court_dimmed = cv2.cvtColor(hsv_roi, cv2.COLOR_HSV2BGR)

    # Enhance contrast of the saturated image. Image now is ready to be analyzed
    contrast_enhanced = cv2.convertScaleAbs(court_dimmed, alpha=1.3, beta=0)  

    return contrast_enhanced

def detect_people_with_HOG(image):
    '''
    Apply the detection of players in the field, after a conversion in gray scale.
    A NMS will be applied in order to select only some boxes that can be overlapped. 

    Args:
    image : the image to analyze

    Returns:
    pick : the coordinates of all the boxes detected 
    '''

    # Pre-processing of the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # HOG definition with the parameters setted in utilities.py
    hog = cv2.HOGDescriptor(winSizeD, blockSize, blockStride, cellSize, nbins)
    daimler_detector = cv2.HOGDescriptor_getDaimlerPeopleDetector()
    hog.setSVMDetector(daimler_detector)
    players, _ = hog.detectMultiScale(gray_image, winStride = hog_winstride, padding = hog_padding, scale = hog_scale)     

    # apply non-maxima suppression to the bounding boxes using a fairly large overlap threshold to try to maintain overlapping boxes that are still people
    players = np.array([[x, y, x + w, y + h] for (x, y, w, h) in players])
    
    pick = non_max_suppression(players, probs=None, overlapThresh=0.5) 
    return pick
   
def detect_players(image, counter, hog_tracking_list):
    ''' This manage the functions for HOG detection '''
    preprocessed_image = preprocess_image(image)
    players_coordinates = detect_people_with_HOG(preprocessed_image)

    save_new_coordinates(players_coordinates, image, counter, hog_tracking_list)
    