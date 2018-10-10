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

data = np.load('data_train.npy')
test=np.load('data_test.npy')
truth=np.load('ground_truth.npy')

valid=data[1900:2100,3500:4200]
train_data=data[900:1050, 4700:5000]
test_data = test[1500:1800, 2850:3550]

test_dataset=np.zeros(3)
test_labels=np.zeros(1)
test_locations=np.zeros(2)
testcarlocs=np.zeros(2)

imgplot = plt.imshow(train_data)
plt.title('training data')
#plt.show()

def CheckAccuracy(truth):
	
	truth_train_car_locations=np.zeros(2)
	truth_valid_car_locations=np.zeros(2)
	u_train = np.zeros(1)
	v_train = np.zeros(1)

	#to check accuracy -
	for x in range(0, truth.shape[0]):
		for y in range(0, truth.shape[1]):
		    	#y,x
		    	#train_data=data[800:1300, 4100:5000] (having 5 red cars!)
			if(truth[x,0] in range(4100,5000) and truth[x,1] in range(800,1300)):
				truth_train_car_locations = np.vstack((truth_train_car_locations, [truth[x,0], truth[x,1]]))
				u_train = np.append(u_train, truth[x,0])
				v_train = np.append(v_train, truth[x,1])
			
			#valid=data[1900:2100,3500:4200] (having 3 red cars!)
			if(truth[x,0] in range(3500,4200) and truth[x,1] in range(1900,2100)):
				truth_valid_car_locations = np.vstack((truth_valid_car_locations, [truth[x,0], truth[x,1]]))

	def unique_rows(a):
		a = np.ascontiguousarray(a)
		unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
		return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

	truth_valid_car_locations = unique_rows(truth_valid_car_locations)
	truth_train_car_locations = unique_rows(truth_train_car_locations)
	truth_train_car_locations = np.delete(truth_train_car_locations, 0, axis=0)
	truth_valid_car_locations = np.delete(truth_valid_car_locations, 0, axis=0)

	return truth_train_car_locations, truth_valid_car_locations

#For training data y,x
def TrainDataGeneration(train_data):
	
	train_dataset=np.zeros(3)
	train_labels=np.zeros(1)
	train_locations=np.zeros(2)
	traincarlocs=np.zeros(2)
	TrainTrain_Data = np.zeros(1)

	u_train = np.zeros(1)
	v_train = np.zeros(1)

	for y in range (train_data[:,1,0].size):
		for x in range(train_data[1,:,0].size):
			if train_data[y,x,0]!=0 and train_data[y,x,1]!=0 and train_data[y,x,2]!=0 :
				d=[train_data[y,x,0],train_data[y,x,1],train_data[y,x,2]]
				train_dataset=np.vstack((train_dataset,d))
				train_locations=np.vstack((train_locations,[4700+x,900+y]))
				if (y in range ((964-900)-5, (964-900)+5) and x in range ((4865-4700)-5, (4865-4700)+5)) or (y in range ((1012-900) - 5, (1012-900)+5) and x in range ((4984-4700)-5, (4984-4700)+5)) or (y in range ((974-900)-5, (974-900)+5) and x in range ((4985-4700)-5, (4985-4700)+5)):
					train_labels=np.vstack((train_labels,[1]))
					traincarlocs=np.vstack((traincarlocs,[4700+x,900+y]))
					TrainTrain_Data=np.append(TrainTrain_Data,5)

					u_train = np.append(u_train, x)
					v_train = np.append(v_train, y)
				else:
					train_labels=np.vstack((train_labels,[0]))
					traincarlocs=np.vstack((traincarlocs,[4700+x,900+y]))
					TrainTrain_Data=np.append(TrainTrain_Data,0)
				
	TrainTrain_Data=np.append(TrainTrain_Data,0)

	return train_dataset, train_locations, train_labels, traincarlocs, TrainTrain_Data

#For validation data y,x
def ValidationDataGeneration(valid):
	
	valid_dataset=np.zeros(3)
	valid_labels=np.zeros(1)
	valid_locations=np.zeros(2)
	validcarlocs=np.zeros(2)
	u_valid = np.zeros(1)
	v_valid = np.zeros(1)

	for y in range (valid[:,1,0].size):
		for x in range(valid[1,:,0].size):
			if valid[y,x,0]!=0 and valid[y,x,1]!=0 and valid[y,x,2]!=0 :
				d=[valid[y,x,0],valid[y,x,1],valid[y,x,2]]
				valid_dataset=np.vstack((valid_dataset,d))
				valid_locations=np.vstack((valid_locations,[3500+x,1900+y]))
				if (y in range((1978-1900)-5, (1978-900)+5) and x in range((3533-3500)-5, (3533-3500)+5)) or (y in range((1936-1900)-5, (1936-1900)+5) and x in range((3828-3500)-5, (3828-3500)+5)) or (y in range((2024-1900)-5, (2024-1900)+5) and x in range((4152-3500)-5, (4152-3500)+5)):
					valid_labels=np.vstack((valid_labels,[1]))
					validcarlocs=np.vstack((validcarlocs,[3500+x,1900+y]))

					u_valid = np.append(u_valid, x)
					v_valid = np.append(v_valid, y)
				else:
					valid_labels=np.vstack((valid_labels,[0]))
					validcarlocs=np.vstack((validcarlocs,[3500+x,1900+y]))

	return valid_dataset, valid_locations, valid_labels, validcarlocs, u_valid, v_valid


def ValidAndPlot(train_dataset, TrainTrain_Data, valid_dataset, valid_labels, u_valid, v_valid, validcarlocs):

	TrainTrain_Data = np.delete(TrainTrain_Data, 0, axis=0)
	valid_labels = np.delete(valid_labels, 0, axis=0)
	car_locations=np.zeros(1)

	#Initialising KNN classifier here -
	knn = KNeighborsClassifier(n_neighbors=15)
	knn.fit(train_dataset,TrainTrain_Data.ravel())
	pred = knn.predict(valid_dataset)
	valid_labels = np.append(valid_labels, 1)

	print (accuracy_score(valid_labels,pred) * 100, '%')
	np.savetxt('pred.txt', pred, delimiter=',')

	#for plotting training data results
	for x in range (pred.size):
		if (pred[x] > 0):
			print(validcarlocs[x])
			u_valid = np.append(u_valid, validcarlocs[x][0])
			v_valid = np.append(v_valid, validcarlocs[x][1])
			car_locations=np.append(car_locations,validcarlocs[x])

	np.savetxt('validlocation.txt', car_locations, delimiter=',')

	#validation image:
	img = plt.imshow(valid)
	fig,ax = plt.subplots(1)
	ax.set_aspect('equal')
	ax.imshow(train_data)

	# Now, loop through coord arrays, and create a circle at each x,y pair
	for xx, yy in zip(u_valid, v_valid):

		#print('xx: ', xx)
		#print('xx-3500: ', int(xx-3800))
		#print('yy: ', yy)
		#print('yy-1900: ', int(yy-1900))
		circ = Circle((int(xx-3800), int(yy-1900)),5)
		circ.set_facecolor('r')
		ax.add_patch(circ)

	# Show the image
	plt.title('marked validation image image')
	#plt.show()

	return 0



