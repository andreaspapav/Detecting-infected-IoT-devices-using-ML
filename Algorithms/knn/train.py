import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
import joblib

def knn():
	#Change this variable to use a different dataset.
	data = pd.read_csv("../dataset/binary_50k_labeled.csv")
	data = data.drop(data.columns[0], axis=1)
	data = data.drop(data.columns[0], axis=1)
	data.info()
	new = np.array(data)
	
	data_train, data_test = train_test_split(new,test_size=0.2,random_state=42)

	#Prepare data for implementing Random Forest
	x_train = data_train[:,:-1]
	y_train = data_train[:,-1]
	x_test = data_test[:,:-1]
	y_test = data_test[:,-1]
	
	print("kneighbors...")
	#Hyper parameters tuning
	# Number of neighbors
	n_neighbors = [int(x) for x in np.linspace(start = 1, stop = 10, num = 1)]
	# Weight types
	weights =['uniform', 'distance'] 
	# Algorithm types
	algorithm=['auto', 'ball_tree', 'kd_tree', 'brute']
	#leaf sizes
	leaf_size = [10,20,30,40]
	#p values
	p=[1,2,3,4,5]
	
	# Create the random grid
	random_grid = {'n_neighbors': n_neighbors,
			'weights': weights,
			'algorithm': algorithm,
			'leaf_size': leaf_size,
			'p': p
			}

	search = RandomizedSearchCV(estimator = classifier, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
	search.fit(x_train,y_train)
	print(search.best_params_)
	predictions = search.predict(x_test)
	print(search.score(x_train,y_train))

	#Uncomment after the hyperparameters are found.
	#classifier = KNeighborsClassifier(weights= 'distance', p= 1, n_neighbors= 1, leaf_size= 30, algorithm= 'brute')
	# classifier.fit(x_train,y_train)
	# predictions = classifier.predict(x_test)
	# print(classifier.score(x_train,y_train))

	print(classification_report(y_test,predictions))
	
	joblib.dump(classifier, 'knn_50k.pkl')
	
knn()
