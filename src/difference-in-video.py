from utils import *

# Apri i video
video1 = cv2.VideoCapture('results/video/tracked-tr10-eps1/tracked-player_1.avi')
video2 = cv2.VideoCapture('results/video/Kalman-050001/tracked-player_1.avi')

# Proprietà per il video di output
frame_width = int(video1.get(3))
frame_height = int(video1.get(4))
output_video = cv2.VideoWriter('results/video/differenze_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (frame_width, frame_height))

# Funzione per confrontare i frame e evidenziare le differenze
def evidenzia_differenze(frame1, frame2):
    # Calcola la differenza assoluta tra i frame
    diff = cv2.absdiff(frame1, frame2)
    
    # Converte la differenza in scala di grigi
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Applica una soglia per isolare le differenze
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Conta i pixel modificati
    pixel_modificati = np.count_nonzero(thresh)

    # Evidenzia le differenze in nero (o qualsiasi altro colore)
    frame1[thresh == 255] = [0, 0, 0]

    return frame1, pixel_modificati

# Leggi e confronta i frame
total_modified_pixels = 0
while True:
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()

    if not ret1 or not ret2:
        break

    # Evidenzia le differenze e conta i pixel modificati
    output_frame, pixel_modificati = evidenzia_differenze(frame1, frame2)
    total_modified_pixels += pixel_modificati

    # Scrivi il frame di output nel video
    output_video.write(output_frame)

# Rilascia tutte le risorse
video1.release()
video2.release()
output_video.release()
cv2.destroyAllWindows()

# Stampa il totale dei pixel modificati
print("Pixel modificati totali:", total_modified_pixels)
