import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import glob
import os

from gstreamer_pipeline import gstreamer_pipeline

class Stereo:
    def __init__(self):
        # setup capture devices
        self.capL = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
        self.capR = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)

        # specify paths
        self.image_path = './images'
        self.calibration_path = './calibration_data'

        # specify baseline
        self.baseline = 0.04

        # setup keys to check for
        self.exit_keys = [ord('q'), 27]
        self.photo_keys = [ord('f')]
        self.plot_keys = [ord('p')]

        # calibration parameters
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_EPS, 30, 0.001)

        # stereo calibration parameters
        self.criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.flags = 0
        self.flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        self.flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        self.flags |= cv2.CALIB_ZERO_TANGENT_DIST


    def take_pictures(self):
        # make directory if not exists
        if not os.path.exists(self.image_path):
            os.mkdir(self.image_path)

        count = 0
        while True:
            retL, frameL = self.capL.read()
            retR, frameR = self.capR.read()

            # check if reading capture devices went ok
            if retL and retR:
                # resize and display frames
                frameL_small = cv2.resize(frameL, (480, 270))
                frameR_small = cv2.resize(frameR, (480, 270))
                cv2.imshow('calibration', np.hstack((frameL_small, frameR_small)))

                # check for keypresses
                keycode = cv2.waitKey(10) & 0xff
                if keycode in self.photo_keys:
                    # save images to path
                    cv2.imwrite(os.path.join(self.image_path, f'left_{count}.png'), frameL)
                    cv2.imwrite(os.path.join(self.image_path, f'right_{count}.png'), frameR)

                    print(f'Pictures taken: {count}')
                    count += 1
                if keycode in self.exit_keys:
                    break

        cv2.destroyAllWindows()


    def calibrate(self, shape=(9, 6), square_size=0.021):
        # load images
        img_amount = len(glob.glob(os.path.join(self.image_path, '*'))) // 2
        photosL = [cv2.imread(os.path.join(self.image_path, f'left_{i}.png')) for i in range(img_amount)]
        photosR = [cv2.imread(os.path.join(self.image_path, f'right_{i}.png')) for i in range(img_amount)]

        # setup calibration
        w, h = shape
        objp = np.zeros((w * h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)
        objp = objp * square_size

        objpoints = []
        imgpointsL = []
        imgpointsR = []

        # loop over all images
        for i, (imgL, imgR) in enumerate(zip(photosL, photosR)):
            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

            # find the chessboard corners
            retL, cornersL = cv2.findChessboardCorners(grayL, (h, w), None)
            retR, cornersR = cv2.findChessboardCorners(grayR, (h, w), None)
            print(f'Processing images {i}')

            # check if both images were successful
            if retL and retR:
                objpoints.append(objp)

                corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), self.criteria)
                corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), self.criteria)

                imgpointsL.append(corners2L)
                imgpointsR.append(corners2R)

                cv2.drawChessboardCorners(imgL, (h, w), corners2L, retL)
                cv2.drawChessboardCorners(imgR, (h, w), corners2R, retR)

                # resize images before showing on screen
                imgL_small = cv2.resize(imgL, (480, 270))
                imgR_small = cv2.resize(imgR, (480, 270))

                cv2.imshow("corners", np.hstack((imgL_small, imgR_small)))
                cv2.waitKey(500)
            cv2.destroyAllWindows()

        # perform actual calibration
        retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera( objpoints, imgpointsL, grayL.shape[::-1], None, None)
        retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera( objpoints, imgpointsR, grayR.shape[::-1], None, None)

        # get image height and width
        img = photosL[-1]
        h, w = img.shape[:2]

        # calculate new camera matrix, incorporating distortion
        newcameramtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (w, h), 1, (w, h))
        newcameramtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (w, h), 1, (w, h))

        # calculate error of found camera parameters
        mean_errorL = 0
        mean_errorR = 0

        for i in range(len(objpoints)):
            imgpoints2L, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
            imgpoints2R, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)

            errorL = cv2.norm(imgpointsL[i], imgpoints2L, cv2.NORM_L2) / len(imgpoints2L)
            errorR = cv2.norm(imgpointsR[i], imgpoints2R, cv2.NORM_L2) / len(imgpoints2R)

            mean_errorL += errorL
            mean_errorR += errorR

        print("total error left: {}".format(mean_errorL / len(objpoints)))
        print("total error right: {}".format(mean_errorR / len(objpoints)))

        print(f"left:\n{mtxL}")
        print(f"right:\n{mtxR}")

        # save calibration data
        self.mtxL = mtxL
        self.newcameramtxL = newcameramtxL
        self.distL = distL
        self.roiL = roiL

        self.mtxR = mtxR
        self.newcameramtxR = newcameramtxR
        self.distR = distR
        self.roiR = roiR

        self.objpoints = objpoints
        self.imgpointsL = imgpointsL
        self.imgpointsR = imgpointsR
        self.imgsize = grayL.shape[:2]

        # save calibration data to disk
        with open(os.path.join(self.calibration_path, "calibrationL.pkl"), "wb") as f:
            pkl.dump(
                {
                    "mtx": mtxL,
                    "mtx_new": newcameramtxL,
                    "dist": distL,
                    "roi": roiL,
                },
                f,
            )

        with open(os.path.join(self.calibration_path, "calibrationR.pkl"), "wb") as f:
            pkl.dump(
                {
                    "mtx": mtxR,
                    "mtx_new": newcameramtxR,
                    "dist": distR,
                    "roi": roiR,
                },
                f,
            )

        with open(os.path.join(self.calibration_path, "calibrationData.pkl"), "wb") as f:
            pkl.dump(
                {
                    "objpoints": objpoints,
                    "imgpointsL": imgpointsL,
                    "imgpointsR": imgpointsR,
                    "imgsize": grayL.shape[::-1]
                },
                f
            )


    def load_calibration(self):
        with open(os.path.join(self.calibration_path, "calibrationL.pkl"), "rb") as f:
            data = pkl.load(f)
            self.mtxL = data['mtx']
            self.newcameramtxL = data['mtx_new']
            self.distL = data['dist']
            self.roiL = data['roi']

        with open(os.path.join(self.calibration_path, "calibrationR.pkl"), "rb") as f:
            data = pkl.load(f)
            self.mtxR = data['mtx']
            self.newcameramtxR = data['mtx_new']
            self.distR = data['dist']
            self.roiR = data['roi']

        with open(os.path.join(self.calibration_path, "calibrationData.pkl"), "rb") as f:
            data = pkl.load(f)
            self.objpoints = data['objpoints']
            self.imgpointsL = data['imgpointsL']
            self.imgpointsR = data['imgpointsR']
            self.imgsize = data['imgsize']

        with open(os.path.join(self.calibration_path, "stereoMaps.pkl"), "rb") as f:
            data = pkl.load(f)
            self.stereo_mapL = data['mapL']
            self.stereo_mapR = data['mapR']
            self.rect_roiL = data['roiL']
            self.rect_roiR = data['roiR']


    def rectify(self):
        print('Performing stereo calibration...')
        retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(
            self.objpoints,
            self.imgpointsL,
            self.imgpointsR,
            self.newcameramtxL,
            self.distL,
            self.newcameramtxR,
            self.distR,
            self.imgsize,
            self.criteria_stereo,
            flags=self.flags
        )

        print('Performing stereo rectification...')
        alpha = -1
        rectL, rectR, proj_matL, proj_matR, Q, roiL, roiR = cv2.stereoRectify(
            new_mtxL,
            distL,
            new_mtxR,
            distR,
            self.imgsize,
            Rot,
            Trns,
            alpha=alpha
        )

        stereo_mapL = cv2.initUndistortRectifyMap(new_mtxL, distL, rectL, proj_matL, self.imgsize, cv2.CV_32FC1)
        stereo_mapR = cv2.initUndistortRectifyMap(new_mtxR, distR, rectR, proj_matR, self.imgsize, cv2.CV_32FC1)

        # save rectification maps
        self.stereo_mapL = stereo_mapL
        self.stereo_mapR = stereo_mapR
        self.rect_roiL = roiL
        self.rect_roiR = roiR

        # save rectification maps to disk
        with open(os.path.join(self.calibration_path, "stereoMaps.pkl"), 'wb') as f:
            pkl.dump(
                {
                    "mapL": stereo_mapL,
                    "mapR": stereo_mapR,
                    "roiL": roiL,
                    "roiR": roiR
                }, 
                f
            )
    
    
    def test_undistort(self):
        while True:
            _, frameL = self.capL.read()
            _, frameR = self.capR.read()

            h, w = frameL.shape[:2]
            mapxL, mapyL = cv2.initUndistortRectifyMap(self.mtxL, self.distL, None, self.newcameramtxL, (w, h), 5)

            h, w = frameR.shape[:2]
            mapxR, mapyR = cv2.initUndistortRectifyMap(self.mtxR, self.distR, None, self.newcameramtxR, (w, h), 5)

            niceL = cv2.remap(frameL, mapxL, mapyL, cv2.INTER_LINEAR)
            niceR = cv2.remap(frameR, mapxR, mapyR, cv2.INTER_LINEAR)

            niceL_small = cv2.resize(niceL, (480, 270))
            niceR_small = cv2.resize(niceR, (480, 270))

            cv2.imshow('undistort', np.hstack((niceL_small, niceR_small)))

            keycode = cv2.waitKey(10) & 0xff
            if keycode in self.exit_keys:
                break
        cv2.destroyAllWindows()


    def test_rectification(self):
        while True:
            _, frameL = self.capL.read()
            _, frameR = self.capR.read()

            niceL = cv2.remap(frameL, self.stereo_mapL[0], self.stereo_mapl[1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)[30:-30]
            niceR = cv2.remap(frameR, self.stereo_mapR[0], self.stere_omapR[1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)[30:-30]

            niceL_small = cv2.resize(niceL, (480, 270))
            niceR_small = cv2.resize(niceR, (480, 270))

            cv2.imshow('rectification', np.hstack((niceL_small, niceR_small)))

            keycode = cv2.waitKey(10) & 0xff
            if keycode in self.exit_keys:
                break

        cv2.destroyAllWindows()

    
    def distance(self, numDisparities=16*12, blockSize=21, filter=False):
        self.focal_length = (self.mtxL[0, 0] + self.mtxL[1, 1] + self.mtxR[0, 0] + self.mtxR[1, 1]) / 4

        stereo_matcher = cv2.StereoBM.create(numDisparities, blockSize)
        if filter:
            matcherL = stereo_matcher
            matcherR = cv2.ximgproc.createRightMatcher(matcherL)
            
            lmbda = 8000
            sigma = 1.5

            wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo_matcher)
            wls_filter.setLambda(lmbda)
            wls_filter.setSigmaColor(sigma)
        
        while True:
            _, frameL = self.capL.read()
            _, frameR = self.capR.read()

            x, y, w, h = self.rect_roiL
            niceL = cv2.remap(frameL, self.stereo_mapL[0], self.stereo_mapL[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)[y:y+h, x:x+w]
            niceR = cv2.remap(frameR, self.stereo_mapR[0], self.stereo_mapR[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)[y:y+h, x:x+w]

            grayL = cv2.cvtColor(niceL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(niceR, cv2.COLOR_BGR2GRAY)

            if not filter:
                self.disp = stereo_matcher.compute(grayL, grayR)[:, numDisparities:]
            else:
                dispL = matcherL.compute(grayL, grayR)
                dispR = matcherR.compute(grayR, grayL)

                self.disp = wls_filter.filter(dispL, grayL, disparity_map_right=dispR)[:, numDisparities:]

            disp /= 16

            # remove negative disparity values
            disp = np.where(disp <= 0, np.nan, disp)

            # remove outliers
            mean = np.nanmean(disp)
            std = np.nanstd(disp)
            disp = np.where(disp > mean + std * 2, np.nan, disp)

            # our code seems to be consistantly off by a factor of 1.5
            offset_factor = 1.5
            
            # calculate distance and remove outliers
            dist = np.where(np.isnan(disp), np.nan, (self.baseline * self.focal_length) / np.nan_to_num(disp) * offset_factor)
            dist = np.where(dist > np.nanmean(dist) + np.nanstd(dist) * 2, np.nan, dist)

            dist_norm = cv2.normalize(np.nan_to_num(disp), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_U8)
            cv2.imshow('distance', np.hstack((frameL, dist_norm)))

            keycode = cv2.waitKey(10) & 0xff
            if keycode in self.exit_keys:
                break
            if keycode in self.plot_keys:
                fig, (ax0, ax1) = plt.subplots(1, 2)

                ax0.imshow(disp)
                ax0.set_title('Disparity')

                ax1.imshow(dist)
                ax1.set_title('Distance')

                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle()

                plt.show()
        
        cv2.destroyAllWindows