from stereo import Stereo

stereo = Stereo()
# stereo.calibrate()
# stereo.rectify()
stereo.load_calibration()
stereo.test_undistort()