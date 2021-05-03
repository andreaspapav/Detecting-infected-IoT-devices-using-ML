import numpy as np
import pandas as pd
import joblib
from sklearn import preprocessing
from sklearn.metrics import classification_report

#Read dataset to test model on
data = pd.read_csv("../dataset/unseen_testing.csv")
#data = data.drop(data.columns[0], axis=1)
#data = data.iloc[:200000,:]
data.info()

#Apply pre-processing steps on "new" data
#proto column
column = data['proto'] 
encoder = preprocessing.LabelEncoder()
encoder.classes_ = np.load('../dataset/labels/binary_200k/proto_classes.pkl.npy',allow_pickle=True)
x0 = encoder.transform(column)
data['proto'] = x0

#state column
column = data['conn_state']
encoder = preprocessing.LabelEncoder()
encoder.classes_ = np.load('../dataset/labels/binary_200k/state_classes.pkl.npy',allow_pickle=True)
x0 = encoder.transform(column)
data['conn_state'] = x0

#history column
column = data['history']
encoder = preprocessing.LabelEncoder()
encoder.classes_ = np.load('../dataset/labels/binary_200k/history_classes.pkl.npy',allow_pickle=True)
print(encoder.classes_)
x0 = encoder.transform(column)
data['history'] = x0

#history column
column = data['label']
encoder = preprocessing.LabelEncoder()
encoder.classes_ = np.load('../dataset/labels/binary_200k/label_classes.pkl.npy',allow_pickle=True)
x0 = encoder.transform(column)
data['label'] = x0

#history column
column = data['id.resp_p']
encoder = preprocessing.LabelEncoder()
encoder.classes_ = np.load('../dataset/labels/binary_200k/port_classes.pkl.npy',allow_pickle=True)
x0 = encoder.transform(column)
data['id.resp_p'] = x0

#convert duration to seconds
data['duration'] = pd.to_timedelta(data['duration'])
data['duration'] = data['duration']/ np.timedelta64(1, 's')

#Testing data
x = data.iloc[:,:-1]
#Labels for testing data
y = data.iloc[:,-1]

models = ['ann_50k.pkl','ann_100k.pkl','ann_150k.pkl','ann_200k.pkl','svm_50k.pkl','svm_100k.pkl','svm_150k.pkl','svm_200k.pkl','rf_50k.pkl','rf_100k.pkl','rf_150k.pkl','rf_200k.pkl','ada_50k.pkl','ada_100k.pkl','ada_150k.pkl','ada_200k.pkl','knn_50k.pkl','knn_100k.pkl','knn_150k.pkl','knn_200k.pkl']

for model in models:
	ada_model = joblib.load(model)
	print(model)
	y_pred = ada_model.predict(x)
	print(classification_report(y, y_pred))
