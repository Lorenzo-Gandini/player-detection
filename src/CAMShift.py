print("\n + RUNNING : CAMshift.")

from utils  import *


cap = cv2.VideoCapture(video_path)

# take first frame of the video
ret,frame = cap.read()


# setup initial location of window
x, y, w, h = 1290, 430, 50, 100 # simply hardcoded the values
x2, y2, w2, h2 = 710, 450, 50, 100 # simply hardcoded the values

track_window = (x, y, w, h)
track_window2 = (x2, y2, w2, h2)

# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


if color == 'rosso':
# Creazione delle maschere per catturare il rosso
mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)

# Combinazione delle maschere
mask = cv2.bitwise_or(mask1, mask2)

roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by at least 1 pt

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
while(1):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply camshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)

        img2 = cv2.polylines(frame,[pts],True, 255, 2)

        ret, track_window2 = cv2.CamShift(dst, track_window2, term_crit)

        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)

        img2 = cv2.polylines(frame,[pts],True, 120, 2)

        cv2.imshow('img2',img2)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

print(" + COMPLETE : CAMShift.")
