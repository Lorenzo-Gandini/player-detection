print("\n + RUNNING : Detection of players from a frame.")
from utils  import *

players_detected = []

#---- COURT ------#
def get_roi(image):
    '''
    Defines the Region Of Interest in the frame. This is defined by the court area, defined as the biggest green area detected.
    This operation is usefull in order to reduce the time of HOG 
    '''

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply denoising and erosion in order to remove some artifacts.  
    blurred_image = cv2.GaussianBlur(green_mask, blur_kernel, 0) 
    erosion = cv2.erode(blurred_image, erosion_kernel, iterations=1)
    
    # This operation enlarge the white part, in order to remove lines and enphatize the court
    dilatation = cv2.dilate(erosion, dilatation_kernel, iterations=1)

    # Define contours of the area
    contours, _ = cv2.findContours(dilatation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Go for the biggest area finded in the given image
    max_area = 0
    max_contour = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_contour = cnt

    if max_contour is not None:  
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [max_contour], 255)

        # Dilatate the mask, in roder to increase the area since some players can be near the line of the field.
        kernel = np.ones((100, 100), np.uint8) 
        dilated_mask = cv2.dilate(mask, kernel)

        # Since we dilatate the mask, the dimension of the image must be the same
        mask_resized = cv2.resize(dilated_mask, (image.shape[1], image.shape[0]))
        if mask_resized.dtype != np.uint8:
            mask_resized = mask_resized.astype(np.uint8)

        # Extract the roi from the mask
        roi = cv2.bitwise_and(image, image, mask=mask_resized)
        return roi

def remove_court(image):
    '''
    Remove the court in the image, in order to simplify the player detection.
    '''

    denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(green_mask)
    res = cv2.bitwise_and(image, image, mask=mask_inv)

    return res

#---- PLAYERS ------#
def detect_people(image):
    '''
    Pedestrian detection with HOG. Return coordinates of squares where should be located people in the court.
    '''

    # Pre-processing of the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Define the HOG Descriptor for pedestrian detection
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    daimler_detector = cv2.HOGDescriptor_getDaimlerPeopleDetector()
    hog.setSVMDetector(daimler_detector)
    players, _ = hog.detectMultiScale(gray_image, winStride=(1, 1), padding=(1, 1), scale=1.0001) #winstride dispendioso ma più efficace.

    # apply non-maxima suppression to the bounding boxes using a fairly large overlap threshold to try to maintain overlapping boxes that are still people
    players = np.array([[x, y, x + w, y + h] for (x, y, w, h) in players])
    pick = non_max_suppression(players, probs=None, overlapThresh=0.4)
    
    return pick

def save_players(coords, image):
    '''
    Given a sequence of coordinates, draw the bounding box with the team color that is inside the bounding box.
    '''

    for (xA, yA, xB, yB) in coords:
        roi = image[yA:yB, xA:xB]
        bbox_color = analyze_color_statistics(roi)

        players_detected.append({
            'x': xA,
            'y': yA, 
            'w': xB-xA, 
            'h': yB-yA,
            'color': bbox_color
            })
        
        for player in players_detected:
            player['x'] = int(player['x'])
            player['y'] = int(player['y'])
            player['h'] = int(player['h'])
            player['w'] = int(player['w'])

            if isinstance(player['color'], np.ndarray):
                player['color'] = player['color'].tolist()

    with open('results/json/players_detected.json', 'w') as f:
        json.dump(players_detected, f)

#--- COLORS -------#
def classify_color(color):
    '''
    Classify the color given the hsv values. Need to be changed with other colors if we change the video.
    '''

    print(f"Classify color : {color}")
    h, s, v = color

    if 5 <= h <= 70 and s > 20 and v > 20:  # Range molto ampliato per Giallo
        return 'giallo', (0, 255, 255)
    elif (0 <= h <= 30 or 130 <= h <= 190) and s > 20 and v > 20:  # Range molto ampliato per Rosso
        return 'rosso', (0, 0, 255)
    else:
        return 'non specificato', (255, 255, 255)

def classify_color_euclidean(rgb_color):
    '''
    Classify the color of the bounding box with the comparison between the color inside the bounding box and the standard ones
    '''

    distance_to_yellow = np.sqrt(np.sum((rgb_color - yellow_standard) ** 2))
    distance_to_red = np.sqrt(np.sum((rgb_color -  red_standard) ** 2))

    # Classification of the color based on the distance with the standard colours
    if distance_to_yellow < distance_to_red:
        return 'giallo', (0, 255, 255)
    elif distance_to_red < distance_to_yellow:
        return 'rosso', (0, 0, 255)
    else:
        return 'non specificato', (255, 255, 255)

def find_prevalent_color(avg_color, med_color, mod_color):
    '''
    Define the prevalent color of bounding box. Given average colour, median colour and moda.
    If 2 of this 3 variables has the same colour, so it's defined as the prevalent colour. 
    '''

    colors = [classify_color_euclidean(avg_color), classify_color_euclidean(med_color), classify_color_euclidean(mod_color)]
    color_count = {}

    for color_name, _ in colors:
        if color_name in color_count:
            color_count[color_name] += 1
        else:
            color_count[color_name] = 1

    for color_name, count in color_count.items():
        if count >= 2:
            for color in colors:
                if color[0] == color_name:
                    return color[1]
        else:
            return (255, 255, 255) 

def analyze_color_statistics(roi, value_threshold=30):
    '''
    Filtering the darker pixels and extract the colour stats usefull for the analysis.
    Using average, median and moda, we can avoid errors in defining the color of the box.
    '''

    # For each RGB channel, filter the darker pixels so they don't affect the computing of colors
    non_black_pixels_mask = (roi[:, :, 0] > value_threshold) & \
                            (roi[:, :, 1] > value_threshold) & \
                            (roi[:, :, 2] > value_threshold)
    filtered_roi = roi[non_black_pixels_mask]

    # If there are not enough pixels, return standard values
    if filtered_roi.size == 0:
        return [0, 0, 0], [0, 0, 0], [0, 0, 0]

    # Define average, median and moda for each RGB channel
    average_color = np.mean(filtered_roi.reshape(-1, 3), axis=0)
    median_color = np.median(filtered_roi.reshape(-1, 3), axis=0)
    moda_color = stats.mode(filtered_roi.reshape(-1, 3), axis=0).mode[0]

    bbox_color = find_prevalent_color(average_color, median_color, moda_color)
    return bbox_color

def analyze_frame(image):
    roi = get_roi(image) 
    court_removed = remove_court(roi)
    coordinates = detect_people(court_removed)
    save_players(coordinates, image)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(" + COMPLETE : Detection of players from a frame.\n")