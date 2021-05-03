import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def svm():
	#Change this variable to use a different dataset. (50k/100k/150k/200k)
	data = pd.read_csv("../dataset/binary_50k_labeled.csv")
	data = data.drop(data.columns[0], axis=1)
	data = data.drop(data.columns[0], axis=1)
	data.info()
	new = np.array(data)
	
	#Split training and testing set
	data_train, data_test = train_test_split(new,test_size=0.2,random_state=42)

	#Split training values and labels for both test and train set.
	x_train = data_train[:,:-1]
	y_train = data_train[:,-1]
	x_test = data_test[:,:-1]
	y_test = data_test[:,-1]

	param_grid = {'C': [0.1, 1, 10, 100, 1000],'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'kernel': ['rbf']}
	
	svm_random = RandomizedSearchCV(estimator = SVC(), param_distributions = param_grid, n_iter = 100, cv =2, verbose=2, random_state=42, n_jobs = -1)
	svm_random.fit(x_train, y_train)
	# print best parameter after tuning
	print(svm_random.best_params_)
	#Print Training score
	print(svm_random.score(x_train,y_train))
	predictions = svm_random.predict(x_test)

	#Uncoment lines 36-39 after the best parameters have been found
	# clf = SVC(kernel='rbf',gamma=0.001,C= 1000)
	# clf.fit(x_train, y_train)
	# print(clf.score(x_train,y_train))
	# predictions = clf.predict(x_test)

	#Classification report
	print(classification_report(y_test,predictions))
	joblib.dump(clf, 'svm_50k.pkl')

svm()
