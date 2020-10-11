# import cv2
# import time

# capture = cv2.VideoCapture(0)
# width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# writer = cv2.VideoWriter('../DATA/My_Capture.mp4', cv2.VideoWriter_fourcc(*'DIVX'),25, (width, height), 0)

# font = cv2.FONT_HERSHEY_SIMPLEX
# start = time.time()
# time.clock()
# elapsed=0
# count = 0

# while True:
#     ret, frame = capture.read()
#     elapsed = time.time() - start
#     count += 1
#     fps = count / elapsed

#     new_frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
#     cv2.putText(img=new_frame, text=str(round(fps,2)), org=(10,50), fontFace=font, fontScale=2, color=(255,255,255), thickness=2, lineType=cv2.LINE_4)
#     writer.write(new_frame)    
#     cv2.imshow('Video',new_frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# writer.release()
# capture.release()
# cv2.destroyAllWindows()
#Libs

import cv2
import time
import numpy as np
import itertools


# Functions
def gray_scale(frame):
    """This a function that converts 1 frame from
    the camera feed to grayscale.

    This function uses cv2.
    """

    gray_frame = cv2.cvtColor(src=frame, 
                              code=cv2.COLOR_RGB2GRAY)
    
    return gray_frame


def as_canny(frame):
    """This a function that converts 1 frame from
    the camera feed to the canny function of cv2.

    This function uses cv2.
    """

    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_red = np.array([30,150,50])
    # upper_red = np.array([255,255,180])
    # mask = cv2.inRange(hsv, lower_red, upper_red)
    # res = cv2.bitwise_and(frame,frame, mask= mask)
    # CALCULATE MEDIAN VALUE IMAGE FOR THRESHOLD
    med_val_frame = np.median(frame)
    lower = int(max(0, 0.7*med_val_frame)) #THRESHOLD IS 0 OR 70% OF MEDIAN VALUE
    upper = int(max(255, 1.3*med_val_frame)) # UPPER IS EITHER MAX OF MEDIAN OR 255
    blur_img = cv2.blur(frame, ksize=(5,5)) # POSSIBLE TO USE BLUR_IMG IN CANNY FUNCTION
    frame = cv2.Canny(frame,threshold1=lower, threshold2=upper)

    return frame


def show_corners(frame, maxcorners=250, radius=5, color=(0,0,255), thickness=1):
    """This a function that discovers corners and draws
    circles on the frame from the camera feed.

    This function uses cv2.
    This function uses numpy.
    """

    gray_frame = gray_scale(frame)
    corners = cv2.goodFeaturesToTrack(image=gray_frame, 
                                      maxCorners=maxcorners, 
                                      qualityLevel=0.01, 
                                      minDistance=10)
    corners = np.int0(corners)
    
    for i in corners:
        x, y = i.ravel()
        cv2.circle(frame, 
                   center=(x, y), 
                   radius=radius, color=color, 
                   thickness=thickness)

    return frame


def show_normal(frame):
    """This a function that shows the normal camera feed.

    This function uses cv2.
    """

    return frame


def show_feed_options(filter=False, option=None):
    """This a function that shows the camera feed with the selectect
    option.

    @ Option param: - as_canny
                    - show_corners

    This function uses cv2.
    """
    while True:
        _, frame = capture.read()

        if filter:
            result = option(frame)

        else:
            result = frame
        
        cv2.imshow('Video', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    capture.release()
    cv2.destroyAllWindows()


def loop_show_feed():
    """This a function that shows the camera feed whil loopoing
    through options.

    This function uses cv2.
    """

    start_time = time.time()
    time_threshold = 3
    funct_list = ['show_normal', 'gray_scale',  'as_canny', 'show_corners']
    func = funct_list[0]
    i = 0

    while True:

        time_elapsed = time.time() - start_time
        
        if time_elapsed > time_threshold:

            start_time = time.time()
            i += 1

            if i >= len(funct_list):
                i = 0
            
            func = funct_list[i]         
        
        _, frame = capture.read()

        comm = str(func)
        print(time_elapsed)
        comm = eval(comm)
        result = comm(frame)

        cv2.imshow('Video', result)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
            
            

        


# Variables

capture = cv2.VideoCapture(0)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Runtime

# show_feed_options()
# show_feed_options(True, show_corners)
show_feed_options(True, as_canny)
# loop_show_feed()