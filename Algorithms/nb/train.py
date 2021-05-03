import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB,BernoulliNB,CategoricalNB,ComplementNB,MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold

def naivebayes():
	data = pd.read_csv("../dataset/binary_50k_labeled.csv")
	data = data.drop(data.columns[0], axis=1)
	data = data.drop(data.columns[0], axis=1)
	print(data.info())
	new = np.array(data)
	print(new)
	#cv_method = RepeatedStratifiedKFold(n_splits=1,n_repeats=2,random_state=999)

	data_train, data_test = train_test_split(new,test_size=0.2,random_state=999)

	x_train = data_train[:,:-1]
	y_train = data_train[:,-1]
	x_test = data_test[:,:-1]
	y_test = data_test[:,-1]

	#Naive Bayes using GuassianNB()
	gnb = MultinomialNB()
	#HyperParameter tuning using GridSearchCV
	param_nb = {'alpha': np.arange(-30,30,1)}
	search = GridSearchCV(estimator=gnb,param_grid=param_nb,cv=2,verbose=2)
	search.fit(x_train,y_train)
	# print best parameter after tuning
	print(search.best_params_)
	# print how our model looks after hyper-parameter tuning
	print(search.best_estimator_)
	predictions = search.predict(x_test)
	#clf = GaussianNB(var_smoothing = 1e-09)
	#clf.fit(x_train,y_train)
	predictions = search.predict(x_test)

	#Classification report
	print(classification_report(y_test,predictions))
	
naivebayes()
