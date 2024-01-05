import pickle as pkl

with open('./calibration_data/calibrationData.pkl', 'rb') as f:
    data = pkl.load(f)
    print(data)