import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import autokeras as ak
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import classification_report


data = pd.read_csv("../../dataset/binary_100k.csv")
#Drop column id created due to errors remove later
data = data.drop(data.columns[0], axis=1)
#data = data.drop(data.columns[0], axis=1)
#data = data.drop(data.columns[-1], axis=1)

print(data.info())

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=42)

# It tries 100 different models.
clf = ak.StructuredDataClassifier(overwrite=False, max_trials=100)
# Feed the structured data classifier with training data.
clf.fit(x_train, y_train)
# Predict with the best model.
predicted_y = clf.predict(x_test)
#print(predicted_y)
# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))
print(clf.evaluate(x_train,y_train))
#exported_model = clf.export_model()

#try:
#	exported_model.save("binary_model_autokeras", save_format="tf")
#except Exception:
#	exported_model.save("binary_model_autokeras.h5")

#joblib.dump(clf, 'binary100k.pkl')

#poulou = joblib.load('modelLabels.pkl')
#print(poulou.predict([np.asarray(x_test, dtype=np.str), -1]))




data = pd.read_csv("../../dataset/final_testing.csv")
data.info()
x = data.iloc[:,:-1]
y = data.iloc[:,-1]

print(clf.evaluate(x,y))

y_pred = clf.predict(x)

print(classification_report(y, y_pred))
