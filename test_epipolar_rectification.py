import cv2 
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from gstreamer_pipeline import gstreamer_pipeline

with open('./calibrationL.pkl', 'rb') as f:
    cameraL = pkl.load(f)

with open('./calibrationR.pkl', 'rb') as f:
    cameraR = pkl.load(f)

with open('./epipolar.pkl', 'rb') as f:
    data = pkl.load(f)
    HL, HR = data['HL'], data['HR']
    
capL = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
capR = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)

while True:
    _, frameL = capL.read()
    _, frameR = capR.read()

    h, w = frameL.shape[:2]

    mapxL, mapyL = cv2.initUndistortRectifyMap(cameraL['mtx'], cameraL['dist'], None, cameraL['mtx_new'], (w,h), 5)
    mapxR, mapyR = cv2.initUndistortRectifyMap(cameraR['mtx'], cameraR['dist'], None, cameraR['mtx_new'], (w,h), 5)

    niceL = cv2.remap(frameL, mapxL, mapyL, cv2.INTER_LINEAR)
    niceR = cv2.remap(frameR, mapxR, mapyR, cv2.INTER_LINEAR)

    h, w = frameL.shape[:2]

    niceL = cv2.warpPerspective(niceL, HL, (w, h))
    niceR = cv2.warpPerspective(niceR, HR, (w, h))

    x, y, w, h = 100, 150, 500, 300
    niceL = niceL[y:y+h, x:x+w]
    niceR = niceR[y:y+h, x:x+w]

    cv2.imshow("rectified", np.hstack((cv2.resize(niceL, (480, 270)), cv2.resize(niceR, (480, 270)))))

    grayL = cv2.cvtColor(niceL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(niceR, cv2.COLOR_BGR2GRAY)

    numDisparities = 16*3
    blockSize = 15
    stereo = cv2.StereoBM.create(numDisparities, blockSize)
    disp = stereo.compute(grayL, grayR)[:, numDisparities:]
    disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    mean = disp.mean()
    std = disp.std()
    disp = np.where(disp > mean + std * 2, 0, disp)

    cv2.imshow('disp', disp)

    keycode = cv2.waitKey(10) & 0xff
    if keycode in [ord('q'), 27]:
        break
    if keycode == ord('p'):
        plt.imshow(disp)
        plt.show()

cv2.destroyAllWindows()