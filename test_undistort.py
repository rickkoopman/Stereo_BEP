import cv2 
import numpy as np
import pickle as pkl

from gstreamer_pipeline import gstreamer_pipeline

with open('./calibrationL.pkl', 'rb') as f:
    cameraL = pkl.load(f)

with open('./calibrationR.pkl', 'rb') as f:
    cameraR = pkl.load(f)

capL = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
capR = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)

while True:
    _, frameL = capL.read()
    _, frameR = capR.read()

    h, w = frameL.shape[:2]
    mapxL, mapyL = cv2.initUndistortRectifyMap(cameraL['mtx'], cameraL['dist'], None, cameraL['mtx_new'], (w,h), 5)
    
    h, w = frameR.shape[:2]
    mapxR, mapyR = cv2.initUndistortRectifyMap(cameraR['mtx'], cameraR['dist'], None, cameraR['mtx_new'], (w,h), 5)

    # x, y, w, h = cameraL['roi']
    # niceL = cv2.remap(frameL, mapxL, mapyL, cv2.INTER_LINEAR)[y:y+h, x:x+w]

    # x, y, w, h = cameraR['roi']
    # niceR = cv2.remap(frameR, mapxR, mapyR, cv2.INTER_LINEAR)[y:y+h, x:x+w]

    niceL = cv2.remap(frameL, mapxL, mapyL, cv2.INTER_LINEAR)
    niceR = cv2.remap(frameR, mapxR, mapyR, cv2.INTER_LINEAR)

    cv2.imshow('Left', niceL)
    cv2.imshow('Right', niceR)

    keycode = cv2.waitKey(10) & 0xff
    if keycode in [ord('q'), 27]:
        break

cv2.destroyAllWindows()