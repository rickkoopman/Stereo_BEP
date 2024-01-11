# Bachelor End Project -- Stereo Vision

## Setup
Before you start running the script please specify the following in `Stereo.__init__`:
- `self.image_path`
- `self.calibration_path`
- `self.baseline`

## Steps
1. Take pictures with `Stereo.take_pictures`
2. Calibrate cameras with `Stereo.calibrate`
3. Perform rectification with `Stereo.rectify`
4. Get distance values with `Stereo.distance`

## Other methods
- During calibration or rectification steps, the resulting values are automatically dumped into a pickle file. To load in this data call the `Stereo.load_calibration` method.
- To see if calibration and rectification steps went correctly, there are the methods `Stereo.test_undistort` and `Stereo.test_rectification`.
