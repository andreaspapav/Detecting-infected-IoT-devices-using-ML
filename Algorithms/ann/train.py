import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def neural_network():
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

	parameter_space = {
	'hidden_layer_sizes': [(50,50), (50,100), (100,),(5,2)],
	'activation': ['tanh', 'relu'],
	'solver': ['sgd', 'adam','lbfgs'],
	'alpha': [0.0001, 0.05],
	'learning_rate': ['constant','adaptive'],
	}

	mlp = MLPClassifier(max_iter=100)
	search = GridSearchCV(mlp, parameter_space, refit = True, verbose = 3)
	search.fit(x_train,y_train)
	print(search.best_params_)
	predictions = search.predict(x_test)
	
	#Uncomment after hyerparameters have been selected.
	#clf = MLPClassifier(activation= 'tanh', alpha= 0.0001, hidden_layer_sizes= (50, 50), learning_rate= 'adaptive', solver= 'adam')
	#clf.fit(x_train,y_train)
	#print(clf.score(x_train,y_train))
	#predictions = clf.predict(x_test)
	
	print(classification_report(y_test,predictions))
	joblib.dump(clf, 'ann_50k.pkl')

neural_network()
