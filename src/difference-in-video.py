import cv2
import numpy as np
from utilities import *  # Assicurati che utilities contenga tutto ciò che è necessario

# # Apri i video
# video1 = cv2.VideoCapture('results/video/version_of_video/31-12.avi')
# video2 = cv2.VideoCapture('results/video/version_of_video/31-13.avi')

# # Proprietà per il video di output
# frame_width = int(video1.get(3))
# frame_height = int(video1.get(4))
# output_video = cv2.VideoWriter('results/video/diff_video_31-1213.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (frame_width, frame_height))

# # Funzione per confrontare i frame e evidenziare le differenze
# def evidenzia_differenze(frame1, frame2):
#     # Calcola la differenza assoluta tra i frame
#     diff = cv2.absdiff(frame1, frame2)
    
#     # Converte la differenza in scala di grigi
#     gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

#     # Applica una soglia per isolare le differenze
#     _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # La soglia è stata abbassata a 10

#     # Evidenzia le differenze in rosso (o qualsiasi altro colore)
#     frame1[thresh == 255] = [0, 0, 255]

#     return frame1

# # Leggi e confronta i frame
# while True:
#     ret1, frame1 = video1.read()
#     ret2, frame2 = video2.read()

#     if not ret1 or not ret2:
#         break

#     # Evidenzia le differenze
#     output_frame = evidenzia_differenze(frame1, frame2)

#     # Scrivi il frame di output nel video
#     output_video.write(output_frame)

# # Rilascia tutte le risorse
# video1.release()
# video2.release()
# output_video.release()
# cv2.destroyAllWindows()


# Carica il video
cap = cv2.VideoCapture("results/video/no-finals-check.avi")

# Calcola il numero di frame totali
frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Imposta il punto di inizio e di fine del taglio
start_frame = 100
end_frame = 267

# Calcola la durata del video tagliato
duration = end_frame - start_frame

# Crea un nuovo video writer
video_writer = cv2.VideoWriter('results/video/video_tagliato_nochecks.avi', cv2.VideoWriter_fourcc(*'XVID'), 25,(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


# Ciclo per estrarre i frame e scriverli nel video tagliato
frame_count = 0
while frame_count < frame_length:

    # Leggi il frame
    ret, frame = cap.read()

    # Controlla se il frame è stato letto correttamente
    if not ret:
      print("no buono")
      break

    # Se il frame è all'interno del punto di inizio e di fine, scrivilo nel video tagliato
    if start_frame <= frame_count <= end_frame:
        video_writer.write(frame)
        print(frame_count)

    # Incrementa il contatore dei frame
    frame_count += 1

# Rilascia la cattura del video
cap.release()

# Rilascia il video writer
video_writer.release()

# Stampa un messaggio di conferma
print("Il video tagliato è stato salvato.")
