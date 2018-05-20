import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math

def matches(image1 , image2):
    image2 = cv2.cvtColor(image2 , cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints1 , descriptors1 = orb.detectAndCompute(image1 , None)
    keypoints2 , descriptors2 = orb.detectAndCompute(image2 , None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck = True)
    matches = bf.match(descriptors1 , descriptors2)
    return len(matches)

image_to_be_found = cv2.imread('./images/box_in_scene.png')
#cv2.imshow('image_to_be_found' , image_to_be_found)
#cv2.waitKey()
#cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)

while(True):
    ret , frame = cap.read()

    #cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
    length = frame.shape[0]
    width = frame.shape[1]
    cv2.rectangle(frame , (math.floor(length / 8) , math.floor(width / 8)) , (math.floor(length / 2) , math.floor(width / 2)) , (255 , 0 , 0) , 4)
    #cv2.imshow('me' , frame)

    cropped_image = frame[math.floor(width / 8) : math.floor(width / 2) , math.floor(length / 8) : math.floor(length / 2)]
    #cv2.imshow('cropped image' , cropped_image)

    matches_found = matches(image_to_be_found , cropped_image)

    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

    matches_string = 'Matches:'
    matches_string += str(matches_found)

    cv2.putText(frame, matches_string, (math.floor(length / 2), math.floor(width / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    if matches_found < 200:
        cv2.putText(frame , 'Object not found' , (math.floor(length / 4) , math.floor(width / 4)) , cv2.FONT_HERSHEY_SIMPLEX , 2 , (0 , 0 , 255) , 2 , cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Object found', (math.floor(length / 4), math.floor(width / 4)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('image' , frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()