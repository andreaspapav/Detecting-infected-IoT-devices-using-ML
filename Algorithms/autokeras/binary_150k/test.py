import numpy as np
import pandas as pd
import joblib
from sklearn import preprocessing
import autokeras 
from keras.models import model_from_json

#Read dataset to test model on
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

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
