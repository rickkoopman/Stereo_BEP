import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from gstreamer_pipeline import gstreamer_pipeline

with open("./stereoMaps.pkl", "rb") as f:
    stereo_maps = pkl.load(f)

    mapL = stereo_maps["mapL"]
    roiL = stereo_maps["roiL"]

    mapR = stereo_maps["mapR"]
    roiR = stereo_maps["roiR"]

with open("./calibrationL.pkl", "rb") as f:
    mtxL = pkl.load(f)["mtx_new"]

with open("./calibrationR.pkl", "rb") as f:
    mtxR = pkl.load(f)["mtx_new"]

focal_length = (mtxL[0, 0] + mtxL[1, 1] + mtxR[0, 0] + mtxR[1, 1]) / 4
baseline = 0.04

capL = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
capR = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)

while True:
    _, frameL = capL.read()
    _, frameR = capR.read()

    # niceL = cv2.remap(frameL, mapL[0], mapL[1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    # niceR = cv2.remap(frameR, mapR[0], mapR[1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    x, y, w, h = roiL
    niceL = cv2.remap(
        frameL, mapL[0], mapL[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0
    )[30:-30]
    niceR = cv2.remap(
        frameR, mapR[0], mapR[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0
    )[30:-30]

    # cv2.imshow('rect', np.hstack((niceL, niceR)))

    grayL = cv2.cvtColor(niceL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(niceR, cv2.COLOR_BGR2GRAY)

    numDisparities = 16 * 12
    blockSize = 21

    filter = False
    if not filter:
        stereo = cv2.StereoBM.create(numDisparities, blockSize)
        disp = stereo.compute(grayL, grayR)[:, numDisparities:]
    else:
        matcherL = cv2.StereoBM.create(numDisparities, blockSize)
        matcherR = cv2.ximgproc.createRightMatcher(matcherL)

        dispL = matcherL.compute(grayL, grayR)
        dispR = matcherR.compute(grayR, grayL)

        lmbda = 8000
        sigma = 1.5

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcherL)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        disp = wls_filter.filter(dispL, grayL, disparity_map_right=dispR)[
            :, numDisparities:
        ]

    disp = disp / 16
    disp = np.where(disp <= 0, np.nan, disp)
    mean = np.nanmean(disp)
    std = np.nanstd(disp)
    disp = np.where(disp > mean + std * 2, np.nan, disp)

    offset_factor = 1.5
    dist = np.where(np.isnan(disp), np.nan, (baseline * focal_length) / np.nan_to_num(disp) * offset_factor)
    dist = np.where(dist > np.nanmean(dist) + np.nanstd(dist) * 2, np.nan, dist)

    disp_norm = cv2.normalize(np.nan_to_num(disp), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    print(np.nanmax(dist), np.nanmin(dist))

    cv2.imshow('disp', disp_norm)

    keycode = cv2.waitKey(10) & 0xFF
    if keycode in [ord("q"), 27]:
        break
    if keycode == ord("p"):
        fig, (ax0, ax1) = plt.subplots(1, 2)
        ax0.imshow(disp)
        ax1.imshow(dist)
        plt.show()

cv2.destroyAllWindows()
