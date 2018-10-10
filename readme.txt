To run the classification task, execute command - python run.py, that internally
executes the train.py and test.py functions to test the training(validation) image and test image respectively.

For now, the data sets have been cropped as follows -

	valid=data[1900:2100,3500:4200]
	train_data=data[900:1050, 4700:5000]
	test_data = test[1500:1800, 2850:3550]
	