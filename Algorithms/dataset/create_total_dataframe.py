import numpy as np
import pandas as pd
from zat.log_to_dataframe import LogToDataFrame
import os 

def clear_na(df):
	df = df.drop(columns=['uid','local_orig','local_resp','id.orig_h','id.resp_h','id.orig_p','service'])
	df.dropna(axis=0,inplace=True)
	new = df
	new[['tunnel_parents','label','detailed-label']] = df.iloc[:,-1].str.split(expand=True,)
	new = new.drop(new.columns[12], axis=1)
	new = new.drop(new.columns[12], axis=1)
	new.reset_index(drop=True, inplace=True)
	return new

directory = r'/cs/student/projects1/2017/apapavas/Dissertation/Algorithms/dataset/original_dataset'

total_df = pd.DataFrame()  

for filename in os.listdir(directory):
	file = "original_dataset/" + filename
	df = LogToDataFrame().create_dataframe(file)
	print("LOADED "+str(file))
	print(df.info())
	df = clear_na(df)
	print("CLEANED")
	total_df = total_df.append(df)

total_df.to_csv("total_datset.csv")
