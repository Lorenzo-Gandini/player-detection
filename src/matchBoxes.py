'''
Functions used for matching if HOG and Klamshift boxes are looking for the same player or not
'''


from utilities import *
import KALMShift as klm
from functions import calculate_iou, distance_centroids, analyze_color_statistics, append_box


def load_bounding_boxes_from_list(tracking_list, frame_number):
    '''Loads bounding box from the given track list, for a specific frame.
    
    Args:
    tracking_list : the list where to look for the bboxes.
    frame_number : the specific frame that we want to look for

    Returns:
    bounding_boxes : The bounding boxes in the list that appear in that specific frame

    '''
    bounding_boxes = []
    for item in tracking_list:
        if item['frame'] == frame_number:
            bbox = item['coords']
            bbox_id = item['id']
            bounding_boxes.append((bbox_id, bbox['x'], bbox['y'], bbox['w'], bbox['h']))
    return bounding_boxes


def select_best_bounding_boxes(frame, frame_counter, actual_player_id, hog_tracking_list, kalman_tracking_list):
    '''
    That's the main point where happens the comparison between the bboxes. 
    Extract the bboxes from hog and kalman, and calculate the distance of centroids and how much the bboxes are overlapped.
    If they are really near and overlapped for most of the portion (treshold defined in utilities.py) they are marked as the same player.
    If not, initialize a new box to track.

    Args:
    frame : the image to analyze in order to extract the color of the player
    frame_counter : the frame under observation
    actual_player_id : the last id of the boxes tracked
    hog_tracking_list : the list with the boxes detected by hog in the previous step
    kalman_tracking_list : the list with all the boxes tracked since this moment

    Returns:
    actual_player_id : the updated id of the boxes.
    '''
    hog_boxes = load_bounding_boxes_from_list(hog_tracking_list, frame_counter)
    kalman_boxes = load_bounding_boxes_from_list(kalman_tracking_list, frame_counter)
    ids_to_remove = []

    for hog_box_id, *hog_box_coords in hog_boxes:
        for kalman_box_id, *kalman_box_coords in kalman_boxes:
            # if they represent the same object, remove the id from the list of hog boxes
            if distance_centroids(hog_box_coords, kalman_box_coords, centroid_threshold) and calculate_iou(hog_box_coords, kalman_box_coords, iou_threshold):
                ids_to_remove.append(hog_box_id)
                break

    #Filtered list of hog without the id that represent an already tracked object
    hogs_to_add = [entry for entry in hog_boxes if entry[0] not in ids_to_remove]

    # Loop on every item in hog_list
    for hog_box_id, *hog_box_coords in hogs_to_add:
        x, y, w, h = hog_box_coords
        roi = frame[y:y + h, x:x + w]
        bbox_color = analyze_color_statistics(roi)

        kalman = klm.initialize_kalman_filter()
        bbox_coords = klm.initialize_tracking_objects({'x': x, 'y': y, 'w': w, 'h': h}, kalman)
        kalman_filters[actual_player_id] = kalman
        append_box(kalman_tracking_list, actual_player_id, frame_counter, 
                    {'x': bbox_coords[0], 'y': bbox_coords[1], 'w': bbox_coords[2], 'h': bbox_coords[3]}, bbox_color)
        actual_player_id = actual_player_id + 1

    return actual_player_id


