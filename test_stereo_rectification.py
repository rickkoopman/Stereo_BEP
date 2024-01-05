import cv2 
import numpy as np
import pickle as pkl

from gstreamer_pipeline import gstreamer_pipeline

with open('./stereoMaps.pkl', "rb") as f:
    stereo_maps = pkl.load(f)

    mapL = stereo_maps['mapL']
    roiL = stereo_maps['roiL']

    mapR = stereo_maps['mapR']
    roiR = stereo_maps['roiR']
    
capL = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
capR = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)

while True:
    _, frameL = capL.read()
    _, frameR = capR.read()

    niceL = cv2.remap(frameL, mapL[0], mapL[1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)[30:-30]
    niceR = cv2.remap(frameR, mapR[0], mapR[1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)[30:-30]

    # x, y, w, h = roiL
    # niceL = cv2.remap(frameL, mapL[0], mapL[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)[y:y+h, x:x+w]

    # x, y, w, h = roiR
    # niceR = cv2.remap(frameR, mapR[0], mapR[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)[y:y+h, x:x+w]

    cv2.imshow("Left", niceL)
    cv2.imshow("Right", niceR)

    keycode = cv2.waitKey(10) & 0xff
    if keycode in [ord('q'), 27]:
        break

cv2.destroyAllWindows()