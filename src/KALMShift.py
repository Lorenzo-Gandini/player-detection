'''
KALMShift (Kalman filter + CAMShift) 's operations.
'''

from utilities import *
from functions import load_bboxes_by_frame, append_box, get_roi_hist

def initialize_kalman_filter():
    '''
    Initialize a new kalman filter that will be linked with a box's id.

    Returns:
    kalman : the initialized kalman filter 
    '''

    kalman = cv2.KalmanFilter(4, 2)  # 4 state parameters (x, y, dx, dy), 2 measurement parameters (x, y)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01  # Tune this parameter
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.0001  #Low because i want to  
    return kalman

def initialize_tracking_objects(detected_obj, kalman):
    '''
    Given the coords to track and an initializated kalman filter, defines the initial states of the process of tracking.
    
    Args:
    detected_obj : the object where extract the coords and start the tracking.
    kalman : the initialized kalman filter that will track the object passed

    Returns:
    coordinates of the object
    ''' 
    init_x, init_y, init_w, init_h = detected_obj['x'], detected_obj['y'], detected_obj['w'], detected_obj['h']

    #Initial position and velocity of the object when it starts to track the player
    kalman.statePre = np.array([[init_x + init_w / 2], [init_y + init_h / 2], [0], [0]], np.float32)

    #Will be the updated position. At this moment is the copy of the initial state
    kalman.statePost = kalman.statePre.copy()

    # CONTROLLARE SE SI PUò RIMUUO
    return (init_x, init_y, init_w, init_h)

def process_frame(frame, roi_hist, track_window, kalman):
    ''' 
    Processing of the image in order to better apply kalman
       
    Args:
    frame : the image to analyze
    roi_hist : the histogram of colors of the roi
    track_window : the actual coordinates of the window
    kalman : the initialized kalman filter that will track the object passed

    Returns:
    new_track_window : The updated position of the bounding box
    ''' 

    def is_tracking_reliable(track_window):
        '''
        Check if the tracking made by CAMShift is reliable. The criteria in this case is the area, because the tracking tends to reduce the bounding box.
        If it's to big, it means that is tracking more than one player or something else. During the video the player area is near 2500-3000.
        A treshold of 4000 leave some space for correction but is also enough to cut off the errors.
        '''
        return (track_window[2] * track_window[3]) <= area_treshold

    #Pre-processing to facilitate the tracking
    frame = cv2.GaussianBlur(frame, (9, 9), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # The prediction with kalman
    predicted_state = kalman.predict()
    predicted_x, predicted_y = predicted_state[0, 0], predicted_state[1, 0]

    # The window is the same dimension of hog's
    predicted_w, predicted_h = 48, 96 
    track_window = (int(predicted_x - predicted_w / 2), int(predicted_y - predicted_h / 2), predicted_w, predicted_h)

    # Apply CAMShift
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1)
    ret, new_track_window = cv2.CamShift(dst, track_window, term_crit)

    if is_tracking_reliable(new_track_window):
        ''' Check the reliability of CAMShift's result before updating Kalman filter'''
        # case CAMShit is realiable
        new_x, new_y, new_w, new_h = new_track_window
        center_x = new_x + new_w / 2
        center_y = new_y + new_h / 2
        actual_measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]], np.float32)
        kalman.correct(actual_measurement)

        # Update the tracked window
        new_track_window = (int(max(center_x - new_w / 2, 0)), int(max(center_y - new_h / 2, 0)), new_w, new_h)

    else:
        #Case CAMShift is not reliable: keep the coords detected by kalman

        estimated_x = predicted_state[0, 0]
        estimated_y = predicted_state[1, 0]
        top_left_x = int(max(estimated_x - predicted_w / 2, 0))
        top_left_y = int(max(estimated_y - predicted_h / 2, 0))

        new_track_window = (top_left_x, top_left_y, predicted_w, predicted_h)

    return new_track_window

def create_kalman(frame_number, actual_player_id, kalman_tracking_list, hog_tracking_list):
    '''
    Function that manage all the function in order to create a new kalman filter for a given bounding box.    
    
    Args:
    frame_number : The number of the frame under observation
    actual_player_id : the last id
    kalman_tracking_list : The list that works as memory with all the bounding boxes tracked in every frame.
    hog_tracking_list : the list with all the last detection with HOG.

    Returns:
    actual_player_id : the updated last id inserted
    ''' 

    bboxes = load_bboxes_by_frame(frame_number, hog_tracking_list)

    for bbox in bboxes:
        bbox_color = bbox["color"]
        
        kalman = initialize_kalman_filter()
        initialize_tracking_objects(bbox['coords'], kalman)
        kalman_filters[actual_player_id] = kalman

        append_box(kalman_tracking_list, actual_player_id, frame_number, bbox['coords'], bbox_color)

        actual_player_id = actual_player_id + 1
        
    return actual_player_id

def update_kalman(frame, frame_counter, kalman_tracking_list):
    '''
    Function that manage all the function in order to update the position with the tracker.    
    
    Args:
    frame : the image of the frame
    frame_counter : The number of the frame under observation
    kalman_tracking_list : The list that works as memory with all the bounding boxes tracked in every frame that will be updated.
    ''' 
    previous_bboxes = load_bboxes_by_frame(int(frame_counter-1), kalman_tracking_list)

    for bbox in previous_bboxes:
        bbox_id = bbox["id"]
        bbox_coords = bbox["coords"]
        bbox_color = bbox["color"]

        if bbox_id in kalman_filters:
            kalman = kalman_filters[bbox_id]

            init_x, init_y, init_w, init_h = bbox_coords['x'], bbox_coords['y'], bbox_coords['w'], bbox_coords['h']

            roi = frame[init_y:init_y + init_h, init_x:init_x + init_w]
            roi_hist = get_roi_hist(roi, bbox_color)
            track_window = (init_x, init_y, init_w, init_h)

            new_track_window = process_frame(frame, roi_hist, track_window, kalman)
            append_box(kalman_tracking_list, bbox_id, frame_counter, 
                    {'x': new_track_window[0], 
                    'y': new_track_window[1], 
                    'w': new_track_window[2], 
                    'h': new_track_window[3]}, bbox_color)
            