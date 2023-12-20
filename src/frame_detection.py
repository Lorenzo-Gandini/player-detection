from utils  import *

def get_roi(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply denoising and erosion in order to remove some artifacts.  
    blurred_image = cv2.GaussianBlur(green_mask, blur_kernel, 0) 
    erosion = cv2.erode(blurred_image, erosion_kernel, iterations=1)
    
    # This operation enlarge the white part, in order to remove lines and enphatize the court
    dilatation = cv2.dilate(erosion, dilatation_kernel, iterations=1)

    # Define contours of the area
    contours, _ = cv2.findContours(dilatation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #The treshold is 1/3 in order to remove most of the croud area.
    threshold_area = 1/3*(image.shape[0]*image.shape[1])

    for cnt in contours:
        if cv2.contourArea(cnt) > threshold_area:
           
            # If the area is big enough, create the mask of the image
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [cnt], 255)

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
    denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(green_mask)
    res = cv2.bitwise_and(image, image, mask=mask_inv)

    return res

def detect_people(image, orignal):
    print("I'm in detect_people \n")
    start_time = time.time()

    # Pre-processing of the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    daimler_detector = cv2.HOGDescriptor_getDaimlerPeopleDetector()
    hog.setSVMDetector(daimler_detector)
    players, _ = hog.detectMultiScale(gray_image, winStride=(1, 1), padding=(1, 1), scale=1.001) #winstride dispendioso ma più efficace.

    # apply non-maxima suppression to the bounding boxes using a fairly large overlap threshold to try to maintain overlapping boxes that are still people
    players = np.array([[x, y, x + w, y + h] for (x, y, w, h) in players])
    pick = non_max_suppression(players, probs=None, overlapThresh=0.4)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        print("I'm in the for")
        roi = image[yA:yB, xA:xB]
        player_color, bbox_color = analyze_color_statistics(roi)
        cv2.rectangle(original, (xA, yA), (xB, yB), bbox_color, 2)
        cv2.imshow("result", original)

    #tell me the time of HOG
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tempo impiegato: {elapsed_time} secondi")

    cv2.imshow("Detected", image)

def classify_color(color):
    print(f"Classify color : {color}")
    h, s, v = color

    if 10 <= h <= 60 and s > 30 and v > 30:  # Range molto ampliato per Giallo
        return 'giallo', (0, 255, 255)
    elif (0 <= h <= 25 or 140 <= h <= 180) and s > 30 and v > 30:  # Range molto ampliato per Rosso
        return 'rosso', (0, 0, 255)
    else:
        return 'non specificato', (255, 255, 255)

def euclidean_distance(color1, color2):
    return np.sqrt(np.sum((color1 - color2) ** 2))

def classify_color_euclidean(rgb_color):
    # Definisci i colori standard RGB per giallo e rosso
    yellow_standard = np.array([255, 255, 0])
    red_standard = np.array([255, 0, 0])

    # Calcola la distanza dal giallo e dal rosso
    distance_to_yellow = euclidean_distance(rgb_color, yellow_standard)
    distance_to_red = euclidean_distance(rgb_color, red_standard)

    # Classifica in base alla distanza minima
    if distance_to_yellow < distance_to_red:
        return 'giallo', (0, 255, 255)
    elif distance_to_red < distance_to_yellow:
        return 'rosso', (0, 0, 255)
    else:
        return 'non specificato', (255, 255, 255)

def find_prevalent_color(avg_color, med_color, mod_color):
    # Non è più necessario estrarre il primo elemento; passa l'intero array RGB
    colors = [classify_color_euclidean(avg_color), classify_color_euclidean(med_color), classify_color_euclidean(mod_color)]
    color_count = {}

    for color_name, _ in colors:
        if color_name in color_count:
            color_count[color_name] += 1
        else:
            color_count[color_name] = 1

    for color_name, count in color_count.items():
        if count >= 2:
            # Restituisce il nome e il colore BGR del primo colore che appare almeno due volte
            for color in colors:
                if color[0] == color_name:
                    print(f"color is : {color_name}")
                    return color
        else:
            print(f"color is : white")
            return 'non specificato', (255, 255, 255) 

def analyze_color_statistics(roi, value_threshold=30):

    # Filtra i pixel troppo scuri basandosi sul valore nel canale RGB
    non_black_pixels_mask = (roi[:, :, 0] > value_threshold) & \
                            (roi[:, :, 1] > value_threshold) & \
                            (roi[:, :, 2] > value_threshold)
    filtered_roi = roi[non_black_pixels_mask]

    # Se non ci sono abbastanza pixel, restituisci valori predefiniti
    if filtered_roi.size == 0:
        return [0, 0, 0], [0, 0, 0], [0, 0, 0]

    # Calcola media, mediana e moda per ciascun canale RGB
    average_color = np.mean(filtered_roi.reshape(-1, 3), axis=0)
    median_color = np.median(filtered_roi.reshape(-1, 3), axis=0)
    moda_color = stats.mode(filtered_roi.reshape(-1, 3), axis=0).mode[0]

    print(f"Media RGB: {average_color}, Mediana RGB: {median_color}, Moda RGB: {moda_color}")

    color, bbox_color = find_prevalent_color(average_color, median_color, moda_color)

    return color, bbox_color

roi = get_roi(image)
black_court = remove_court(roi)
detect_people(black_court, image)

cv2.waitKey(0)
cv2.destroyAllWindows()