
from utils import *

def process_frame(analyzed_image):
    '''
    This function takes as input an image ad returns the same image with the draw of the court detected
    '''
    # Converte l'immagine in formato HSV così da poter analizzare il canale h
    hsv = cv2.cvtColor(analyzed_image, cv2.COLOR_BGR2HSV)

    # Crea una maschera binaria per il colore verde, così da andare a indetificare solo il campo da calcio
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Effettua un denoising alla maschera, con un filtro, l'erosione dei residui e la dilatazione dell'area bianca
    # Così da ridurre le linee del campo e minimizzare gli ostacoli ai bordi dell'area. 
    blurred_image = cv2.GaussianBlur(green_mask, blur_kernel, 0) 
    erosion = cv2.erode(blurred_image, erosion_kernel, iterations=1)
    dilatation = cv2.dilate(erosion, dilatation_kernel, iterations=1)

    # Trova i contorni nell'immagine binaria filtrata
    contours, _ = cv2.findContours(dilatation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Disegna i contorni sull'immagine originale
    # Definition of the Area
    max_area = 0
    max_contour = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_contour = cnt

    if max_contour is not None:
        epsilon = 0.005 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        cv2.drawContours(analyzed_image, [approx], -1, (0, 255, 0), 3)        

    return analyzed_image


# Leggi l'immagine
image = cv2.imread('data/img/frame.png')

my_image = process_frame(image)

# Visualizza l'immagine risultante
cv2.imwrite('results/img/court-in-frame.png', my_image)
cv2.waitKey(0)
cv2.destroyAllWindows()