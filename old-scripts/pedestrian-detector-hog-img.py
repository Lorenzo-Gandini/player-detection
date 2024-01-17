from utils  import *
from courtDetectorImg import contours 

image = cv2.imread('data/img/frame.png')
threshold_area = 1/3*(image.shape[0]*image.shape[1])


#--- DEFINIZIONE DI HOG ----#
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
daimler_detector = cv2.HOGDescriptor_getDaimlerPeopleDetector()
hog.setSVMDetector(daimler_detector)

for cnt in contours:
    # Filtra contorni troppo piccoli
    if cv2.contourArea(cnt) > threshold_area:
        # Crea una maschera bianca con la forma della zona d'interesse
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [cnt], 255)

        # Applica la dilatazione alla maschera
        kernel = np.ones((100, 100), np.uint8)  # Dimensioni del kernel per la dilatazione
        dilated_mask = cv2.dilate(mask, kernel)

        # Assicurati che la maschera abbia le stesse dimensioni e sia dello stesso tipo dell'immagine di input
        mask_resized = cv2.resize(dilated_mask, (image.shape[1], image.shape[0]))
        if mask_resized.dtype != np.uint8:
            mask_resized = mask_resized.astype(np.uint8)

        # Estrai la ROI dall'immagine originale usando la maschera dilatata
        roi = cv2.bitwise_and(image, image, mask=mask_resized)


#PRE-PROCESSING DELL'IMMAGINE
# Converti l'immagine in scala di grigi
gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(gray_image)

# Apply Non-Local Means Image Denoising
dst = cv2.fastNlMeansDenoising(clahe_image, h=10, templateWindowSize=3, searchWindowSize=23)


# PLAYER DETECTION
players, _ = hog.detectMultiScale(dst, winStride=(2, 2), padding=(1, 1), scale=1.001)

# apply non-maxima suppression to the bounding boxes using a fairly large overlap threshold to try to maintain overlapping boxes that are still people
players = np.array([[x, y, x + w, y + h] for (x, y, w, h) in players])
pick = non_max_suppression(players, probs=None, overlapThresh=0.4)

new_image = image.copy()
# draw the final bounding boxes
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(new_image, (xA, yA), (xB, yB), (0, 255, 0), 2)

#Salva l'immagine
cv2.imwrite("results/img/pedestrian-detection.jpg", new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()