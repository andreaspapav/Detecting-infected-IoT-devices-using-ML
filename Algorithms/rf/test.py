import numpy as np
import pandas as pd
import joblib
from sklearn import preprocessing

#Change this variable to use a different dataset.
data = pd.read_csv("../dataset/unseen_testing.csv")
#data = data.drop(data.columns[0], axis=1)
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

#Change this variable to use a different classifier.
rf_model = joblib.load('rf_200k.pkl')
print(rf_model.score(x,y))
