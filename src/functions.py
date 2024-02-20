from utilities import *

'''
This file contains most of the functions used along the project.
'''


def load_video(video_path):
    ''' 
    Return the video and all the info about it. 

    Args: 
    video_path : the path where find the video

    Returns:
    cap : The object under observations
    video_length : the total length in frames of the video
    frame_width :the width of the frame
    frame_height : the height of the frame
    video_fps : number of fps in the video
    '''
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, video_length, frame_height, frame_width, video_fps

def load_json_data(path):
    ''' 
    Load the data from the given json

    Args: 
    path : the path where find the json file

    Returns:
    Or the file or an empty list in case of error.
    '''
    try:
        with open(path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def save_json_data(path, tracking_list):
    ''' 
    Save the data from the given list in a json file founded in the path passes as argument.

    Args: 
    path : the path where create the json file
    tracking_list : list where are all the data that will be stored
    '''
    formatted_data = []
    
    for value in tracking_list:
        formatted_data.append(value)

    with open(path, 'w') as file:
        json.dump(formatted_data, file, indent=4)

def append_box(tracking_list, box_id, frame_counter, coords, bbox_color):
    ''' 
    Append to the passed list, the bounding box passed as a parameter
    
    Args:
    tracking_list : The list where append the formatted items
    box_id : The id of the bbox
    frame_counter : The actual frame under observations
    coords : coordinate sof the bbox
    bbox_color : The color extracted from the analysis.
    '''
    tracking_list.append({
        'id': box_id, 
        'frame': frame_counter, 
        'coords': coords,  
        'color': bbox_color
    })

def load_bboxes_by_frame(frame_number, tracking_list):
    ''' 
    Load the list with only the bounding boxes with a specific frame number
    
    Args:
    frame_counter : The actual frame under observations
    tracking_list : The list to filter

    Return:
    The filtered list
    '''
    return [bbox for bbox in tracking_list if bbox['frame'] == frame_number]

def extract_court(image):
    '''
    Given an image, extract the court in order to define the area where find players and reduce time in execution (instead looking also in the crowd).
    To clean from artifacts that can appears during the blurring and during the masking process, are used the operation of erosion and dilatation.
    After that will be found the bigger area that will represent the court.

    Args: 
    image : image to analyze

    Returns:   
    max_contour : the biggest area representing the court points.    
    '''

    green_mask = apply_green_mask(image)
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
    
    return max_contour

def apply_green_mask(image):

    '''
    Given an image, convert it in hsv values from rgb and then extract the mask with only the pixels that are green in the image

    Args:
    image: the image to analyze

    Return:
    The binary mask, where is white everything green, and black all the other pixels.
    '''
        
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    return green_mask

def apply_red_mask(image):
    '''
    Given an image, extract the histogram with a mask for red values

    Args:
    image: the image to analyze

    Return:
    The histogram of the image with the red mask
    '''
        
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)
    roi_hist = cv2.calcHist([hsv], [0], red_mask, [180], [0, 180]) 
    return roi_hist

def apply_yellow_mask(image):
    '''
    Given an image, extract the histogram with a mask for yellow values

    Args:
    image: the image to analyze

    Return:
    The histogram of the image with the yellow mask
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow) 
    roi_hist = cv2.calcHist([hsv], [0], yellow_mask, [180], [0, 180]) 
    return roi_hist

def get_roi_hist(roi, color):
    ''' 
    Get the histogram of color of the image based on the mask for that specific color.

    Args: 
    roi : image to run on the histogram
    color: defines which mask to use

    Retunrs:
    The histogram of the given roi
    '''
    if color == [0, 0, 255] or [255, 255, 255]:
        roi_hist = apply_red_mask(roi)   
    else: 
        roi_hist = apply_yellow_mask(roi)
    return roi_hist

def classify_color_euclidean(rgb_color):
    '''
    Define the nearest "standard" color between red and yellow.

    Args:
    rgb_color : the rgb to check

    Return:
    The rgb value of the closest standard color.
    '''
    distance_to_yellow = np.linalg.norm(rgb_color - yellow_standard)
    distance_to_red = np.linalg.norm(rgb_color - red_standard)

    if distance_to_yellow < distance_to_red:
        return 'yellow', (0, 255, 255)
    elif distance_to_red < distance_to_yellow:
        return 'red', (0, 0, 255)
    else:
        return 'non specificato', (255, 255, 255)

def classify_and_find_prevalent_color(average_color, median_color, moda_color):
    '''
    Given the 3 statistics from analyze_color_statistics(), count the occurrency that appears most.
    If at least 2 statistic tell us that a specific color is the most frequent, this will be the main color.

    Args :
    average_color : the average value for each channel color inside the given roi
    median_color : the median value for each channel color inside the given roi
    moda_color : colors that appears most frequently

    Return:
    rgb : The rgb value of the main color 
    '''
    colors = [classify_color_euclidean(average_color),
              classify_color_euclidean(median_color),
              classify_color_euclidean(moda_color)]

    # Using a dictionary to count occurrences
    color_count = {color: sum(c[0] == color for c in colors) for color, _ in colors}

    # Determine the prevalent color
    _, rgb = max(colors, key=lambda c: color_count[c[0]])
    return rgb

def analyze_color_statistics(roi):
    '''
    Given an image, this function extracts the main color inside the picture. Used in order to identify the color of the jersey of the player detected.
    Firstly remove the darker pixels. Then extract the average color, the median color and the moda of color inside the image.
    Based on these value, we are able to identify the dominant color.

    Args: 
    roi : the image to analyze

    Return:
    bbox_color : An RGB value that represent the color of the jersey.
    '''
    # Filter darker pixels that can bring biases
    value_threshold = 30
    non_black_pixels_mask = np.all(roi > value_threshold, axis=-1)
    filtered_roi = roi[non_black_pixels_mask]

    # in case there's an error in the bbox, make it white in order to be visible during analysis.
    if filtered_roi.size == 0:
        return (255, 255, 255)

    # average, median, and moda of colors inside roi
    average_color = np.mean(filtered_roi, axis=0)
    median_color = np.median(filtered_roi, axis=0)
    moda_color = stats.mode(filtered_roi, axis=0).mode[0]

    # find prevalent color
    bbox_color = classify_and_find_prevalent_color(average_color, median_color, moda_color)
    return bbox_color

def save_new_coordinates(coords, image, counter, hog_tracking_list):
    '''
    Used by HOG in order to save new players detected.
    
    Args:
    coords : coordinates to save
    image : needed for the color analysis
    counter : the actual frame under analysis
    hog_tracking_list : the list of all the tracked bboxes. The place where append the coordinates.
    '''

    id = 0
    for single_coord in coords:
        xA, yA, xB, yB = single_coord
        roi = image[yA:yB, xA:xB]
        bbox_color = analyze_color_statistics(roi)
        id += 1

        append_box(hog_tracking_list, id, counter, 
                   {'x': int(xA), 'y': int(yA), 'w': int(xB - xA), 'h': int(yB - yA)}, bbox_color)

def check_false_positive(frame_counter, kalman_tracking_list):
    '''
    Clean a list from false positive, defined as boxes that keep the same coords for 4 consecutive frames.
    Basically the bboxes that doesn't contain something to track with the filter.

    Args:
    frame_counter : Value of the actual frame under observation
    kalman_tracking_list : list of all the elements tracked with kalman.
    '''
    ids_to_remove = []
    frame_min3 =  [entry for entry in kalman_tracking_list if entry['frame'] == frame_counter - 3]
    frame_min2 =  [entry for entry in kalman_tracking_list if entry['frame'] == frame_counter - 2]
    frame_min1 =  [entry for entry in kalman_tracking_list if entry['frame'] == frame_counter - 1]
    frame_act =  [entry for entry in kalman_tracking_list if entry['frame'] == frame_counter]

    for index in range(len(frame_act)):
        if frame_act[index]['coords'] == frame_min1[index]['coords'] and frame_act[index]['coords'] == frame_min2[index]['coords'] and frame_act[index]['coords'] == frame_min3[index]['coords'] :
            ids_to_remove.append(frame_act[index]['id'])

    for id in ids_to_remove:
        if id in kalman_filters:
            del kalman_filters[id]
    kalman_tracking_list = [entry for entry in kalman_tracking_list if entry['id'] not in ids_to_remove]

def distance_centroids(bbox_1, bbox_2, treshold):
    ''' 
    Defines how much two bboxes are far away from each other, checking the distance of centroids. 
    Is used in order to understand if they represent the same player.

    Args:
    bbox1 : the first bounding box to check
    bbox1 : the second bounding box to check
    treshold : the treshold that sets the distance.

    Returns: 
    Boolean with the evaluation of the distance (ca. if the bboxes represents the same player). 
    '''
    x1, y1, w1, h1 = bbox_1
    x2, y2, w2, h2 = bbox_2
    centroid_1 = (x1 + w1 / 2, y1 + h1 / 2)
    centroid_2 = (x2 + w2 / 2, y2 + h2 / 2)

    if np.linalg.norm(np.array(centroid_1) - np.array(centroid_2)) < treshold:
        return True
    else:
        return False

def calculate_iou(bbox1, bbox2, threshold):
    ''' 
    Define if two bounding boxes respects the treshold of Interesection over Union. 
    Is used in order to understand if they represent the same player.

    Args:
    bbox1 : the first bounding box to check
    bbox1 : the second bounding box to check
    treshold : the treshold that sets the rule.

    Returns:
    Boolean with the evaluation of IoU (ca. if the represent the same player). 
    '''
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    union_area = w1 * h1 + w2 * h2 - inter_area
    iou = inter_area / union_area

    if iou > threshold:
        return True
    else:
        return False

def check_area_along_video(video_length):
    ''' 
    Check along the video passed in input if some bounding boxes has constantly the same area. 
    In that case, it removes them from the list of b-boxes that will be drawn.

    Args:
    video_length : the length in frames of the video

    Returns:
    tracking_data : Cleaned list of the bounding boxes
    '''

    # Dictionary to keep track of areas for each ID
    area_history = {}
    ids_to_remove = set()  

    tracking_data = load_json_data(bboxes_path)

    for frame in range(video_length):

        # load boxes with the current frame
        current_frame_boxes = [box for box in tracking_data if box['frame'] == frame]

        for box in current_frame_boxes:
            box_id = box['id']
            box_area = box['coords']['w'] * box['coords']['h']

            if box_id not in area_history:
                # initialize the area history for this specific ID
                area_history[box_id] = [box_area]
            else:
                # Append the current area to the area history 
                area_history[box_id].append(box_area)

                # if the area has not changed for more than 5 frames remove it
                if len(area_history[box_id]) >= 5 and len(set(area_history[box_id][-5:])) == 1:
                    ids_to_remove.add(box_id)

    tracking_data = [entry for entry in tracking_data if entry['id'] not in ids_to_remove] #Filtered list
    save_json_data(bboxes_path, tracking_data)

    return tracking_data

def remove_boxes_outside_court(list, video_length):
    '''
    Remove all the boxes that go outside the court.

    Args: 
    list : the list with all the bounding box 
    video_length : the total length of the video, expressed in frames

    Return:
    filtered_boxes : Cleaned list of the bounding boxes
    '''

    ids_to_remove = set()
    cap = cv2.VideoCapture(video_path)
    kernel = np.ones((10,10), np.uint8) 

    for frame in range(video_length):
        _, frame_to_check = cap.read()
        court_contour = extract_court(frame_to_check)

        if court_contour is not None:
            # analog operations for court detection during HOG
            mask = np.zeros(frame_to_check.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [court_contour], 255)
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)

            for item in load_bboxes_by_frame(frame, list):
                #extract the coordinates to check
                x, y, w, h = item['coords'].values()
                corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]

                # check if the coords are outside the dilatated mask
                for corner in corners:
                    x_coord = max(0, min(corner[0], dilated_mask.shape[1] - 1))
                    y_coord = max(0, min(corner[1], dilated_mask.shape[0] - 1))
                    if dilated_mask[y_coord, x_coord] == 0:  # <- Outside the corners
                        ids_to_remove.add(item['id'])
                        break

    filtered_bboxes = [bbox for bbox in list if bbox['id'] not in ids_to_remove]

    return filtered_bboxes