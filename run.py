import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.cluster
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.patches import Circle
from train import CheckAccuracy, TrainDataGeneration, ValidationDataGeneration, ValidAndPlot
from test import TestDataGeneration, TestAndPlot


data = np.load('data_train.npy')
test=np.load('data_test.npy')
truth=np.load('ground_truth.npy')


valid=data[1900:2100,3500:4200]
train_data=data[900:1050, 4700:5000]
test_data = test[1500:1800, 2850:3550]

train_truth_value, valid_truth_value = CheckAccuracy(truth)

train_dataset, train_locations, train_labels, traincarlocs, TrainTrain_Data = TrainDataGeneration(data[900:1050, 4700:5000])
valid_dataset, valid_locations, valid_labels, validcarlocs, u_valid, v_valid = ValidationDataGeneration(data[1900:2100,3500:4200])
test_dataset, test_locations, test_labels, testcarlocs = TestDataGeneration(test[1500:1800, 2850:3550])

#for plotting validation data
val0 = ValidAndPlot(train_dataset, TrainTrain_Data, valid_dataset, valid_labels, u_valid, v_valid, validcarlocs)

#for plotting training data
val1 = TrainAndPlot(train_dataset, TrainTrain_Data, valid_dataset, valid_labels, u_valid, v_valid, validcarlocs)

print(train_truth_value)
print(valid_truth_value)