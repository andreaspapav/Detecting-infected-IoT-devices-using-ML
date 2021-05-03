#import tensorflow as tf
#import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
#import autokeras as ak
import joblib

#Read dataset to test model on
data = pd.read_csv("../dataset/total_datset.csv")
data = data.drop(data.columns[0], axis=1)
data = data.drop(data.columns[-1], axis=1)
data.info()
#Training values
x = data.iloc[:200000,:-1]
#Labels for values
y = data.iloc[:200000,-1]

#Change pickle file to chose a different classifier
model = joblib.load('binary_50k/binary_50k.pkl')
print(model.evaluate(x,y))
