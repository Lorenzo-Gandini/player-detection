'''
This works for multiple players and with a loop taked from the json
'''
print("\n + RUNNING : CAMshift.")

from utils  import *
from frameAnalysis import analyze_frame

cap = cv2.VideoCapture(video_path)

# take first frame of the video
ret,frame = cap.read()

# REMOVE IN ANALYSIS 
# analyze_frame(frame)

#Open the json with the coordinates and the color of each bounding box.
try:
    with open('results/json/players_detected.json', 'r') as f:
        detected_objects = json.load(f)       

except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Errore nella lettura dei dati: {e}")

#Initialize the lists of tracking windows and histograms
tracking_windows = []
histograms = []
colors = []

for obj in detected_objects:

    track_window = (obj['x'], obj['y'], obj['w'], obj['h'])
    tracking_windows.append(track_window)

    roi = frame[obj['y'] : obj['y'] + obj['h'], obj['x'] : obj['x'] + obj['w']]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    colors.append(obj['color'])
    
    #yellow
    if obj['color'] == '[0, 255, 255]':
        mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)

    #red    
    elif obj['color'] == '[0, 0, 255]':
        mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    
    else:
        mask = cv2.inRange(hsv_roi, np.array([0, 0, 0]), np.array([255, 255, 255 ]))

    
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])   
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    histograms.append(roi_hist)


term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1)

while(1):
    ret, frame = cap.read()
    if not ret:
        break
   
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for i, window in enumerate(tracking_windows):
        dst = cv2.calcBackProject([hsv],[0],histograms[i],[0,180],1)
        ret, window = cv2.CamShift(dst, window, term_crit)
        
        # Aggiorna la finestra di tracciamento
        tracking_windows[i] = window
        
        # Disegna la finestra di tracciamento sul frame
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        frame = cv2.polylines(frame, [pts], True, colors[i], 2)

    cv2.imshow('Tracciamento Multi-Giocatore', frame)
    if cv2.waitKey(30) & 0xff == 27:
        break

print(" + COMPLETE : CAMShift.")


