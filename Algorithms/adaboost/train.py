from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def adaboost():
        #Chnage this parameter for a different dataset.
        data = pd.read_csv("../dataset/binary_50k_labeled.csv")
        data = data.drop(data.columns[0], axis=1)
        data = data.drop(data.columns[0], axis=1)
        print(data.info())
        new = np.array(data)

        data_train, data_test = train_test_split(new,test_size=0.2,random_state=42)

        x_train = data_train[:,:-1]
        y_train = data_train[:,-1]
        x_test = data_test[:,:-1]
        y_test = data_test[:,-1]

        ada = AdaBoostClassifier()
        ada.fit(x_train,y_train)
        predictions = ada.predict(x_test)

        #Classification report
        print(classification_report(y_test,predictions))
        print(ada.score(x_train,y_train))
        joblib.dump(ada, 'ada_50k.pkl')

adaboost()
