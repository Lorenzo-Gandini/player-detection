import cv2
import numpy as np

# Leggi l'immagine
image = cv2.imread('data/img/frame.png')
original = image.copy()

# Parametri
lower_green = np.array([30, 30, 30]) # Valori minimi di H, S, V per il verde
upper_green = np.array([70, 255, 255]) # Valori massimi di H, S, V per il verde
blur_kernel = (15, 15) #Kernel da applicare per il denoising
erosion_kernel = np.ones((7, 7), np.uint8) #kernel da applicare per l'erosione
dilatation_kernel = np.ones((13, 13), np.uint8) #Kernel da applciare per la dilatazione

# Converte l'immagine in formato HSV così da poter analizzare il canale h
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Crea una maschera binaria per il colore verde, così da andare a indetificare solo il campo da calcio
green_mask = cv2.inRange(hsv, lower_green, upper_green)

# Effettua un denoising alla maschera, con un filtro, l'erosione dei residui e la dilatazione dell'area bianca
# Così da ridurre le linee del campo e minimizzare gli ostacoli ai bordi dell'area. 
blurred_image = cv2.GaussianBlur(green_mask, blur_kernel, 0) 
erosion = cv2.erode(blurred_image, erosion_kernel, iterations=1)
dilatation = cv2.dilate(erosion, dilatation_kernel, iterations=1)

# Trova i contorni nell'immagine binaria filtrata
contours, _ = cv2.findContours(dilatation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

''' 
#Threshold per definire l'area. Siccome un campo da calcio occupa l'area maggiore dell'imamgine,
la soglia sarà un'terzo dell'area totale.
Previene la selezione di eventuali banner, o cartelloni pubblicitari verdi.
'''
threshold_area = 1/3*(image.shape[0]*image.shape[1])

# Disegna i contorni sull'immagine originale
for cnt in contours:
    # Filtra contorni troppo piccoli
    if cv2.contourArea(cnt) > threshold_area:
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

# Visualizza l'immagine risultante
cv2.imwrite('results/img/court-in-frame.png', image)
cv2.imshow("Campo da calcio", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
