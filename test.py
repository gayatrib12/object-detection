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
#train_data=data[800:1300, 4100:5000]
train_data=data[900:1050, 4700:5000]
test_data = test[1500:1800, 2850:3550]

valid_dataset=np.zeros(3)
valid_labels=np.zeros(1)
valid_locations=np.zeros(2)
validcarlocs=np.zeros(2)

labelValid_Data=np.empty([1,1])
CarValid_Data=np.empty([1,1])
#CarValid_Data=np.array([0,0])

train_dataset=np.zeros(3)
train_labels=np.zeros(1)
train_locations=np.zeros(2)
traincarlocs=np.zeros(2)

# #For testing data y,x
# #testing_data=test[1500:1800, 2850:3550]
def TestDataGeneration(valid):

	test_dataset=np.zeros(3)
	test_labels=np.zeros(1)
	test_locations=np.zeros(2)
	testcarlocs=np.zeros(2)

	for y in range (test_data[:,1,0].size):
		for x in range(test_data[1,:,0].size):
			if test_data[y,x,0]!=0 and test_data[y,x,1]!=0 and test_data[y,x,2]!=0 :
				d=[test_data[y,x,0],test_data[y,x,1],test_data[y,x,2]]
				test_dataset=np.vstack((test_dataset,d))
				test_locations=np.vstack((test_locations,[2850+x,1500+y]))
				if (y==1525-1500 and x==2900-2850) or (y==1645-1500 and x==2990-2850) or (y==1685-1500 and x==3040-2850) or (y==1620-1500 and x==3185-2850) or (y==1740-1500 and x==3142-2850):
					test_labels=np.vstack((test_labels,[1]))
					testcarlocs=np.vstack((testcarlocs,[2850+x,1500+y]))
				else:
					test_labels=np.vstack((test_labels,[0]))
					testcarlocs=np.vstack((testcarlocs,[2850+x,1500+y]))
	
	return test_dataset, test_locations, test_labels, testcarlocs
		

def TestAndPlot(train_dataset, TrainTrain_Data, valid_dataset, valid_labels, u_valid, v_valid, validcarlocs):

	TrainTrain_Data = np.delete(TrainTrain_Data, 0, axis=0)
	valid_labels = np.delete(valid_labels, 0, axis=0)
	car_locations=np.zeros(1)

	#Initialising KNN classifier here -
	knn = KNeighborsClassifier(n_neighbors=15)
	knn.fit(train_dataset,TrainTrain_Data.ravel())
	pred = knn.predict(test_dataset)
	print (accuracy_score(valid_labels,pred) * 100, '%')
	np.savetxt('pred.txt', pred, delimiter=',')

	for x in range (pred.size):
		
		if (pred[x] > 0):
			print(testcarlocs[x])
			u_valid = np.append(u_valid, testcarlocs[x][0])
			v_valid = np.append(v_valid, testcarlocs[x][1])
			
			car_locations=np.append(car_locations,testcarlocs[x])

	car_locations = np.delete(car_locations, (0), axis=0)
	car_locations = unique_rows(car_locations)
	print (accuracy_score(valid_labels,pred))
	np.savetxt('testcarlocation.txt', car_locations, delimiter=',')

	#training image:
	img = plt.imshow(valid)
	fig,ax = plt.subplots(1)
	ax.set_aspect('equal')
	ax.imshow(train_data)

	# Now, loop through coord arrays, and create a circle at each x,y pair
	for xx, yy in zip(u_valid, v_valid):

		#print('xx: ', xx)
		# print('xx-2850: ', int(xx-2850))
		# print('yy: ', yy)
		# print('yy-1500: ', int(yy-1500))
		circ = Circle((int(xx-2850), int(yy-1500)),5)
		circ.set_facecolor('r')
		ax.add_patch(circ)


	# Show the image
	plt.title('marked training image')
	#plt.show()



	return 0



