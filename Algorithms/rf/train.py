import numpy as np
from sklearn.ensemble import RandomForestClassifier as rf
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import joblib
from sklearn.metrics import classification_report, confusion_matrix

def rf_classifier():
	#Simply change the dataset to read in in this variable. 50k/100k/150k/200k	
	data = pd.read_csv("../dataset/binary_50k_labeled.csv")
	data = data.drop(data.columns[0], axis=1)
	data = data.drop(data.columns[0], axis=1)
	print(data.info())
	new = np.array(data)
	print(new)	
	data_train, data_test = train_test_split(new,test_size=0.2,random_state=42)

	#Prepare data for implementing Random Forest
	x_train = data_train[:,:-1]
	y_train = data_train[:,-1]
	x_test = data_test[:,:-1]
	y_test = data_test[:,-1]

	#Hyper parameters tuning
	# Number of trees in random forest
	n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
	# Number of features to consider at every split
	max_features = ['auto', 'sqrt']
	# Maximum number of levels in tree
	max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
	max_depth.append(None)
	# Minimum number of samples required to split a node
	min_samples_split = [2, 5, 10]
	# Minimum number of samples required at each leaf node
	min_samples_leaf = [1, 2, 4]
	# Method of selecting samples for training each tree
	bootstrap = [True, False]
	# Create the random grid
	random_grid = {'n_estimators': n_estimators,
				   'max_features': max_features,
				   'max_depth': max_depth,
				   'min_samples_split': min_samples_split,
				   'min_samples_leaf': min_samples_leaf,
				   'bootstrap': bootstrap}
	
	#Hyperparameter tunning
	rf_random = RandomizedSearchCV(estimator = classifier, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

	rf_random.fit(x_train,y_train)
	
	#Print the best params found
	print(rf_random.best_params_)
	predictions = rf_random.predict(x_test)
	
	#Parameters chosen after tuning
	#classifier = rf(n_estimators= 1600, min_samples_split= 5, min_samples_leaf= 1, max_features= 'sqrt', max_depth=70, bootstrap= False)
	#classifier.fit(x_train,y_train)
	#print(classifier.score(x_train,y_train))
	#predictions = classifier.predict(x_test)

	print(classification_report(y_test,predictions))

	joblib.dump(classifier, 'rf_50k.pkl')	


rf_classifier()
