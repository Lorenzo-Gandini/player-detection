'''
Import the coords from json and implement Kalman
'''
from utils import *

print("\n +++++++++++ RUNNING : Kalman ")
# Initialize the start time
start_time = time.time()
json_path = "results/json/players_detected.json"
AREA_TRESHOLD = 5000
counter_player = 0

# Initialize Kalman filter
kalman = cv2.KalmanFilter(4, 2)  # 4 state parameters (x, y, dx, dy), 2 measurement parameters (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.5  # Tune this parameter
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.001  # Tune this parameter

#Extract the first box
def extract_file():
    '''Load the json and extract the values of the bboxes'''
    try:
        with open(json_path, 'r') as f:
            detected_objects = json.load(f)       
            return detected_objects

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Errore nella lettura dei dati: {e}")


def is_camshift_result_reliable(track):
    area = track[2]*track[3]
    if area > AREA_TRESHOLD:
        return False
    else:
        return True
    
detected_objects = extract_file()

for obj in detected_objects:
    counter_player += 1
    print("# I'm running player : " + str(counter_player))
    video_output = "results/video/Kalman-050001/tracked-player"
    video_output = video_output + "_" + str(counter_player) + ".avi"

    # Get the video
    cap = cv2.VideoCapture(video_path)
    ret,frame = cap.read()
    height, width, _ = frame.shape #Dimension of the frame for write in new video
    result = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'XVID'), 25, (width, height)) 

    init_x, init_y, init_w, init_h = obj['x'], obj['y'], obj['w'], obj['h'] 

    kalman.statePre = np.array([[init_x + init_w / 2], [init_y + init_h / 2], [0], [0]], np.float32)
    kalman.statePost = np.array([[init_x + init_w / 2], [init_y + init_h / 2], [0], [0]], np.float32)

    track_window = (init_x, init_y, init_w, init_h)

    roi = frame[obj['y'] : obj['y'] + obj['h'], obj['x'] : obj['x'] + obj['w']]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    if obj['color'] == [0, 0, 255] or [255, 255, 255]:
        mask_red1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask_red1, mask_red2)    
    else: 
        mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)

    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])

    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1)

    start_time_2 = time.time()
    while(True):
        elapsed_time = time.time() - start_time_2
        print(f"Elapsed Time: {elapsed_time:.2f} seconds", end="\r")

        ret,frame = cap.read()
        
        if ret == True:
            height, width, _ = frame.shape #Dimension of the frame for write in new video

            workon = cv2.GaussianBlur(frame.copy(), (9,9), 0)
            hsv = cv2.cvtColor(workon, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            # Kalman Prediction
            predicted_state = kalman.predict()
            predicted_x, predicted_y = predicted_state[0, 0], predicted_state[1, 0]

            # Adjust the window size as needed
            predicted_w, predicted_h = 48, 96  # Example sizes, adjust based on your scenario

            track_window = (int(predicted_x - predicted_w / 2),
                            int(predicted_y - predicted_h / 2),
                            predicted_w, predicted_h)

            # Apply CAMShift
            ret, new_track_window = cv2.CamShift(dst, track_window, term_crit)

            # Check the reliability of CAMShift's result before updating Kalman filter
            if is_camshift_result_reliable(new_track_window):
                new_x, new_y, new_w, new_h = new_track_window
                center_x = new_x + new_w / 2
                center_y = new_y + new_h / 2
                actual_measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]], np.float32)
                estimated_state = kalman.correct(actual_measurement)
                
                # Use CAMShift result for drawing
                rect_points = np.array([[new_x, new_y], [new_x + new_w, new_y],
                                        [new_x + new_w, new_y + new_h], [new_x, new_y + new_h]], np.int32)
                reliability_text = "Reliable Tracking"
            else:
                estimated_x = predicted_state[0, 0]  # Access the first element
                estimated_y = predicted_state[1, 0]  # Access the first element

                # Convert center back to top-left corner
                top_left_x = int(estimated_x - predicted_w / 2)
                top_left_y = int(estimated_y - predicted_h / 2)

                rect_points = np.array([[top_left_x, top_left_y],
                                        [top_left_x + predicted_w, top_left_y],
                                        [top_left_x + predicted_w, top_left_y + predicted_h],
                                        [top_left_x, top_left_y + predicted_h]], np.int32)
                reliability_text = "Unreliable Tracking"
               
            # Add reliability text
            color = obj['color'] 
            frame = cv2.putText(frame, reliability_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (255, 255, 255), 2, cv2.LINE_AA)
            img = cv2.polylines(frame, [rect_points], True, color, 2)
            result.write(img)
        else:
            break
    print(f"\n# Completed player : {counter_player} in {time.time() - start_time_2} seconds \n")        
    cap.release()
    result.release()

print (f" + COMPLETE : Kalman in {time.time() - start_time} seconds")