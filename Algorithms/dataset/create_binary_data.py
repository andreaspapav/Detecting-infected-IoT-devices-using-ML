import pandas as pd

def read_df():
		df = pd.read_csv('total_datset.csv')
		return df

def get_small_dataset():
		df = read_df()
		#attacks = find_names(df)
		final_data = pd.DataFrame()
		temp_df = df.loc[df['detailed-label'] == 'PartOfAHorizontalPortScan']
		temp_df = temp_df.sample(n = 20000)
		#print(len(temp_df))
		final_data = final_data.append(temp_df)
		temp_df = df.loc[df['label'] == 'Benign']
		temp_df = temp_df.sample(n = 50000)
		final_data = final_data.append(temp_df)
		temp_df = df.loc[df['detailed-label'] == 'Okiru']
		temp_df = temp_df.sample(n = 8000)
		final_data = final_data.append(temp_df)
		temp_df = df.loc[df['detailed-label'] == 'DDoS']
		temp_df = temp_df.sample(n = 9000)
		final_data = final_data.append(temp_df)
		temp_df = df.loc[df['detailed-label'] == 'Attack']
		temp_df = temp_df.sample(n = 9000)
		final_data = final_data.append(temp_df)
		temp_df = df.loc[df['detailed-label'] == 'C&C']
		temp_df = temp_df.sample(n = 4000)
		final_data = final_data.append(temp_df)

		#final_data.to_csv("binary.csv")
		#print(final_data.info())
		final_data = final_data.drop(final_data.columns[0], axis=1)
		final_data = final_data.drop(final_data.columns[-1], axis=1)
		final_data.to_csv("binary_100k.csv")
		print(final_data.info())
		
if __name__ == '__main__':
		get_small_dataset()
