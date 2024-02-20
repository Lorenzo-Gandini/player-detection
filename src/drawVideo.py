'''
Functions used for drawn the squares / bounding boxes around the players, based on the coordinates in the list passed in input.
'''

from utilities import *
from functions import check_area_along_video, load_json_data, remove_boxes_outside_court

def draw_bounding_box(frame, player_data):
    '''
    Draw the bounding box of the gven player
    
    Args:
    frame : the image where put the bounding boxes
    player_data : the information about the bounding box
    '''
    coords = player_data['coords']
    color = player_data['color']
    x, y, w, h = coords['x'], coords['y'], coords['w'], coords['h']

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

def final_video(cap, video_length, frame_height, frame_width, video_fps, results_video_path):
    '''
    This function is the main part of the drawing part. 
    Before drawing the boxes, clean some false positive. After that, 

    Args: 
    cap : the video to analyze
    video_length : the length of the video, in frame
    frame_height : the height of the frame, in order to write a new video with the same dimension of the original
    frame_width : the width of the frame, in order to write a new video with the same dimension of the original
    video_fps : the fps of the video, in order to write a new video with the same fps of the original
    results_video_path : the path where write the final results.
    '''

    if not cap.isOpened():
        print("Errore during video opening")
        return
    out = cv2.VideoWriter(results_video_path, cv2.VideoWriter_fourcc(*'XVID'), video_fps, (frame_width, frame_height))
    current_frame = 0
    check_area_along_video(video_length)

    tracking_data = load_json_data(bboxes_path)
    tracking_data = remove_boxes_outside_court(tracking_data, video_length)
    
    while current_frame < video_length :
        print(emoji.emojize (f":pencil:  Drawing frame {current_frame} / {video_length}"), end="\r")
        ret, frame = cap.read()
        
        if not ret:
            print(f"End of video or error in reading frame {current_frame}")
            break
        
        # Find the players of that specific frame
        players_in_frame = [p for p in tracking_data if p['frame'] == current_frame]

        # For each of the players founded, draw the bbox.
        for player_data in players_in_frame:
            draw_bounding_box(frame, player_data)
        
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
