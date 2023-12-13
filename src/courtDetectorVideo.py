import cv2
import numpy as np

# Leggi il video
video_path = 'data/video/Bundes short.mp4'
output_path = 'results/video/court-detected.avi' # Path del video da scrivere
cap = cv2.VideoCapture(video_path)

# Verifica se il video è stato aperto correttamente
if not cap.isOpened():
    print("Errore nell'apertura del video.")
    exit()

# Leggi il primo frame per ottenere le dimensioni dell'immagine
ret, frame = cap.read()
if not ret:
    print("Errore nella lettura del primo frame.")
    exit()

height, width, _ = frame.shape # Estrai le dimensioni del frame
roi_top = int(height / 5) # Definisci la ROI (parte superiore dell'immagine), così da escludere buona parte dei tifosi

# Parametri del codec e il VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))

#Definizione dei kernel
lower_green = np.array([30, 30, 30]) # Valori minimi di H, S, V per il verde
upper_green = np.array([70, 255, 255]) # Valori massimi di H, S, V per il verde
blur_kernel = (15, 15) #Kernel da applicare per il denoising
erosion_kernel = np.ones((7, 7), np.uint8) #kernel da applicare per l'erosione
dilatation_kernel = np.ones((13, 13), np.uint8) #Kernel da applciare per la dilatazione

while True:
    # Leggi il frame successivo
    ret, frame = cap.read()

    # Esci dal ciclo se il video è terminato
    if not ret:
        break

    # Converte il frame in formato HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Crea una maschera binaria per il colore verde
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    blurred_image = cv2.GaussianBlur(green_mask, blur_kernel, 0) #Applicare denoising tramite blur
    erosion = cv2.erode(blurred_image, erosion_kernel, iterations=1) #Erosione per eliminare piccoli artifatti e ridurre le linee
    dilatation = cv2.dilate(erosion, dilatation_kernel, iterations=1) #Dilatazione per enfatizzare la zona verde
    contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Identificazione dei contorni della maschera

    #Definisco come treshold una zona molto ampia, per filtrare tutte le aree ridotte. Il campo è sicuramente l'area maggiore identificata
    threshold_area = 1/3*(frame.shape[0]*frame.shape[1])
    
    # Disegna i contorni sull'immagine originale
    for cnt in contours:
        if cv2.contourArea(cnt) > threshold_area:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 5)

    # Scrivi il frame risultante nel nuovo video
    out.write(frame)

# Rilascia la cattura del video e chiudi il VideoWriter
cap.release()
out.release()
cv2.destroyAllWindows()

