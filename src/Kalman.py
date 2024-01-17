'''
Import the coords from json and implement Kalman
'''

# Import required libraries
from utils import *

# Constants
AREA_THRESHOLD = 5000
COLOR_RED = [0, 0, 255]
COLOR_WHITE = [255, 255, 255]
JSON_PATH = "results/json/players_detected.json"

# Function Declarations
def initialize_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)  # 4 state parameters (x, y, dx, dy), 2 measurement parameters (x, y)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.001  # Tune this parameter
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.0001  # Tune this parameter
    return kalman

def load_json_data():
    # Load data from JSON file
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error in reading data: {e}")
        return None
    
def get_roi_hist(roi, color):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if color == [0, 0, 255] or [255, 255, 255]:
        mask_red1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask_red1, mask_red2)    
    else: 
        mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    return roi_hist

def is_tracking_reliable(track_window):
    # Work on this
    # Determine if tracking is reliable based on AREA_THRESHOLD
    area = track_window[2] * track_window[3]
    return area <= AREA_THRESHOLD

def prepare_video_writer(frame, counter_player):
    # Prepare the video writer object
    height, width, _ = frame.shape
    video_output = f"results/video/Kalman-000100001/tracked-player_{counter_player}.avi"
    return cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'XVID'), 25, (width, height))

def initialize_tracking_objects(detected_obj, kalman):
    # Initialize tracking objects and Kalman filter
    init_x, init_y, init_w, init_h = detected_obj['x'], detected_obj['y'], detected_obj['w'], detected_obj['h']
    kalman.statePre = np.array([[init_x + init_w / 2], [init_y + init_h / 2], [0], [0]], np.float32)
    kalman.statePost = kalman.statePre.copy()
    return (init_x, init_y, init_w, init_h)

def process_frame(frame, roi_hist, track_window, kalman, obj):
    # Process each frame for tracking
    workon = cv2.GaussianBlur(frame.copy(), (9, 9), 0)
    hsv = cv2.cvtColor(workon, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Kalman Prediction
    predicted_state = kalman.predict()
    predicted_x, predicted_y = predicted_state[0, 0], predicted_state[1, 0]

    # Adjust the window size as needed
    predicted_w, predicted_h = 48, 96  # Example sizes, adjust based on your scenario
    track_window = (int(predicted_x - predicted_w / 2), int(predicted_y - predicted_h / 2), predicted_w, predicted_h)

    # Apply CAMShift
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1)
    ret, new_track_window = cv2.CamShift(dst, track_window, term_crit)

    # Check the reliability of CAMShift's result before updating Kalman filter
    if is_tracking_reliable(new_track_window):
        new_x, new_y, new_w, new_h = new_track_window
        center_x = new_x + new_w / 2
        center_y = new_y + new_h / 2
        actual_measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]], np.float32)
        estimated_state = kalman.correct(actual_measurement)

        # Use CAMShift result for drawing
        rect_points = cv2.boxPoints(ret)
        rect_points = np.intp(rect_points)
        reliability_text = "Reliable Tracking"
    else:
        estimated_x = predicted_state[0, 0]
        estimated_y = predicted_state[1, 0]
        top_left_x = int(estimated_x - predicted_w / 2)
        top_left_y = int(estimated_y - predicted_h / 2)
        rect_points = np.array([[top_left_x, top_left_y], [top_left_x + predicted_w, top_left_y],
                                [top_left_x + predicted_w, top_left_y + predicted_h], [top_left_x, top_left_y + predicted_h]], np.int32)
        reliability_text = "Unreliable Tracking"

    # Draw the tracking result
    color = obj['color']
    frame = cv2.polylines(frame, [rect_points], True, color, 2)
    frame = cv2.putText(frame, reliability_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame, new_track_window

# Main Execution
def main():
    print("\n+++++++++++ RUNNING : Kalman")
    start_time = time.time()

    detected_objects = load_json_data()

    if detected_objects is None:
        print("Error: Unable to read file")
        return

    kalman = initialize_kalman_filter()
    counter_player = 0

    for obj in detected_objects:
        start_time_player = time.time()
        counter_player += 1
        print("# I'm running player : " + str(counter_player))
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read video")
            continue

        result = prepare_video_writer(frame, counter_player)
        init_x, init_y, init_w, init_h = initialize_tracking_objects(obj, kalman)
        track_window = (init_x, init_y, init_w, init_h)

        # Extract ROI and calculate histogram
        roi = frame[init_y:init_y + init_h, init_x:init_x + init_w]
        roi_hist = get_roi_hist(roi, obj['color'])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, track_window = process_frame(frame, roi_hist, track_window, kalman, obj)
            
            elapsed_time_player = time.time() - start_time_player
            print(f"Elapsed Time for Player {counter_player}: {elapsed_time_player:.2f} seconds", end="\r")
            result.write(frame)

        cap.release()
        result.release()
        print(f"# Completed player : {counter_player} \n")

    print(f" + COMPLETE : Kalman in {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()
