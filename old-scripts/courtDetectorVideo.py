import cv2
import numpy as np
from courtDetectorImg import process_frame

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
#roi_top = int(height / 5) # Definisci la ROI (parte superiore dell'immagine), così da escludere buona parte dei tifosi


# Parametri del codec e il VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    processed_frame = process_frame(frame)
    out.write(processed_frame)

# Rilascia la cattura del video e chiudi il VideoWriter
cap.release()
out.release()
cv2.destroyAllWindows()

