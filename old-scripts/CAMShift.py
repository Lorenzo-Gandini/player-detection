'''
Given a json with the initial coordinates of the bounding boxes, this code run the Camshift for these bbox.
'''

from utils import *

# Initialize the start time
start_time = time.time()
counter_player = 0

MIN_AREA = 40 * 80
MAX_AREA = 75 * 150

def extract_file():
    '''Load the json and extract the values of the bboxes'''
    try:
        with open(json_path, 'r') as f:
            detected_objects = json.load(f)       
            return detected_objects

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Errore nella lettura dei dati: {e}")

detected_objects = extract_file()


for obj in detected_objects:
    video_output = "results/video/tracked-tr50-eps1-blur9/tracked-player"
    counter_player += 1
    
    print("\n + RUNNING :\nPlayer "+str(counter_player))
    
    cap = cv2.VideoCapture(video_path)
    ret,frame = cap.read()
    
    track_window = (obj['x'], obj['y'], obj['w'], obj['h'])
    color = obj['color']
    roi = frame[obj['y'] : obj['y'] + obj['h'], obj['x'] : obj['x'] + obj['w']]

    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #extract the color for the bbox
    if color == [0, 255, 255]:
        print('Color : yellow')
        mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
    elif color == [0, 0, 255] or [255, 255, 255]:
        print('Color : red')
        # Mask for red backprojection
        mask_red1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask_red1, mask_red2)
    else :
        print('Color : white')


    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1)

    video_output = video_output + "_" + str(counter_player) + ".avi" 

    height, width, _ = frame.shape #Dimension of the frame for write in new video
    result = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'XVID'), 25, (width, height)) 

    while(1):
        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Print elapsed time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds", end="\r")

        ret, frame = cap.read()
        if ret == True:
            workon = cv2.GaussianBlur(frame.copy(), (9,9), 0)

            hsv = cv2.cvtColor(workon, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            # apply camshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            _ , _ , wA, hA = track_window

            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.intp(pts)
            img = cv2.polylines(frame,[pts],True, color, 2)

            result.write(img)
        else:
            break
    print("\n + COMPLETED.")
    cap.release()
    result.release()

