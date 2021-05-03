import numpy as np
import pandas as pd
import tensorflow as tf
#import pickle
import autokeras as ak
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import classification_report

data = pd.read_csv("../../dataset/binary_50k.csv")
#Drop column id created due to errors remove later
data = data.drop(data.columns[0], axis=1)

print(data.info())

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=42)

# It tries 100 different models.
clf = ak.StructuredDataClassifier(overwrite=False, max_trials=10)
# Feed the structured data classifier with training data.
clf.fit(x_train, y_train,epochs=10)
# Evaluate the best model with testing data.
print("Test results")
print(clf.evaluate(x_test, y_test))
print("Train results")
print(clf.evaluate(x_train,y_train))
exported_model = clf.export_model()

try:
	exported_model.save("binary_model_autokeras50k", save_format="tf")
except Exception:
	exported_model.save("binary_model_autokeras.h5")

joblib.dump(clf, "autokeras_binary_50k.pkl")
