'''
Code for Pre-processing the dataset created, exports the classes for each column Encoded and creates the updated dataset for training..
Code by: Andreas Papavasiliou
'''
import numpy as np
import pandas as pd
from sklearn import preprocessing

#Change the name of the dataset that you want to Label Encode
data = pd.read_csv("binary_100k.csv")
data.info()
#Pre processing on specific columns, proto
#Encode proto label
x_transform = data['proto']
proto_le = preprocessing.LabelEncoder()
proto_le.fit(x_transform)
x0 = proto_le.transform(x_transform)
data['proto'] = x0
#export proto labels for testing use
np.save('labels/binary_100k/proto_classes.pkl', proto_le.classes_)

#conn_state label encoder
x_transform = data['conn_state']
state_le = preprocessing.LabelEncoder()
state_le.fit(x_transform)
x0 = state_le.transform(x_transform)
#replace data in dataset with lebeled one
data['conn_state'] = x0
#export conn_state labels for testing use
np.save('labels/binary_100k/state_classes.pkl', state_le.classes_)

#history label encoder
x_transform = data['history']
history_le = preprocessing.LabelEncoder()
history_le.fit(x_transform)
x0 = history_le.transform(x_transform)
#replace data in dataset with lebeled one
data['history'] = x0
#export conn_state labels for testing use
np.save('labels/binary_100k/history_classes.pkl', history_le.classes_)

#label column - label encoder
x_transform = data['label']
label_le = preprocessing.LabelEncoder()
label_le.fit(x_transform)
x0 = label_le.transform(x_transform)
#replace data in dataset with lebeled one
data['label'] = x0
#export labels for testing use
np.save('labels/binary_100k/label_classes.pkl', label_le.classes_)

#resp_p column - label encoder
x_transform = data['id.resp_p']
port_le = preprocessing.LabelEncoder()
port_le.fit(x_transform)
x0 = port_le.transform(x_transform)
#replace data in dataset with lebeled one
data['id.resp_p'] = x0
#export labels for testing use
np.save('labels/binary_100k/port_classes.pkl', port_le.classes_)


#Convert duration to seconds
data['duration'] = pd.to_timedelta(data['duration'])
data['duration'] = data['duration']/ np.timedelta64(1, 's')

#export labeled dataset.
data.to_csv("binary_100k_labeled.csv")
