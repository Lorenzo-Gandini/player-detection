# Import required libraries
from utilities import *
import HOG as HOG
import drawVideo as dv
import KALMShift as klm
import matchBoxes as mtch
from functions import load_video, save_json_data, check_false_positive

total_time = time.time()

print(emoji.emojize ("\n:T-Rex: Running"))

#Load the video. The path is defined in utilities
cap, video_length, frame_height, frame_width, video_fps = load_video(video_path)  

while frame_counter <= video_length:
    start_time = time.time()
    ret, frame = cap.read()

    print(emoji.emojize (f":window:  Analyzing frame n:{frame_counter} "))
                                                                        
    if not ret:
        # No more frame to analyze
        break
    
    #First frame
    if frame_counter == 0:
        HOG.detect_players(frame, frame_counter, hog_tracking_list)
        actual_player_id = klm.create_kalman(frame, actual_player_id, frame_counter, kalman_tracking_list, hog_tracking_list)
        
        print(emoji.emojize (f":magnifying_glass_tilted_right: First detection in {round(time.time() - start_time, 2)} sec."))
        frame_counter += 1
        actual_player_id += 1
        hog_tracking_list = []
        continue

    #Run this for new detections    
    if frame_counter % 25 == 0:
        HOG.detect_players(frame, frame_counter, hog_tracking_list)
        klm.update_kalman(frame, frame_counter, kalman_tracking_list)
        actual_player_id = mtch.select_best_bounding_boxes(frame, frame_counter, actual_player_id, hog_tracking_list, kalman_tracking_list)
        print(emoji.emojize (f":magnifying_glass_tilted_right: Detecting new players in {round(time.time() - start_time, 2)} sec."))
        hog_tracking_list = []

    #Update the position of players with trackers
    else:
        klm.update_kalman(frame, frame_counter, kalman_tracking_list)
        if (frame_counter - 5 ) % 25 == 0:  
            check_false_positive(frame_counter, kalman_tracking_list)
        print(emoji.emojize (f":soccer_ball: Tracking execute in {round(time.time() - start_time, 2)} sec."))


    frame_counter += 1   

#Save the bounding boxes detected in a json file.
save_json_data(bboxes_path, kalman_tracking_list)

cap.release()
cap = cv2.VideoCapture(video_path)

# Last function that draws the bounding box
dv.final_video(cap, video_length, frame_height, frame_width, video_fps, results_video_path)

cap.release()
print(emoji.emojize("\n:victory_hand:  Execution completed !"))
print(emoji.emojize(f"\n:nine-thirty: Total time : {round(time.time() - total_time, 2)} seconds"))
