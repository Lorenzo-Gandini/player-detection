
print("\n + RUNNING : Detection of the court in a single frame.")
from utils import *

def process_frame(analyzed_image):
    '''
    This function takes as input an image ad returns the same image with the draw of the court detected, with a green line.
    '''

    hsv = cv2.cvtColor(analyzed_image, cv2.COLOR_BGR2HSV)

    # Creation of a green mask, in order to identify the court.
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    ''' Denoise the mask with a filter, apply an erosion of the residues and dilates the remaining area. 
    In this way, the lines will be reduced and minimize the obstacles during the area detection.
    --- Are used big kernels, since is necessary to enphatize very much '''
    blurred_image = cv2.GaussianBlur(green_mask, blur_kernel, 0) 
    erosion = cv2.erode(blurred_image, erosion_kernel, iterations=1)
    dilatation = cv2.dilate(erosion, dilatation_kernel, iterations=1)

    # Find all the countours in the filtered image
    contours, _ = cv2.findContours(dilatation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the area
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


#Read the image
image = cv2.imread('data/img/frame.png')
my_image = process_frame(image)

#Write the image with the court detected
cv2.imwrite('results/img/court-in-frame.png', my_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(" + COMPLETE : Detection of the court in a single frame.\n")
print(" - You can find the result image in results/img/court-in-frame.png \n")