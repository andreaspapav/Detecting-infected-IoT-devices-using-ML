import pandas as pd

def read_df():
		df = pd.read_csv('total_datset.csv')
		return df

def get_small_dataset():
		df = read_df()
		#attacks = find_names(df)
		final_data = pd.DataFrame()
		temp_df = df.loc[df['detailed-label'] == 'C&C-HeartBeat']
		temp_df = temp_df.sample(n = 1170)
		#print(len(temp_df))
		final_data = final_data.append(temp_df)
		temp_df = df.loc[df['detailed-label'] == 'PartOfAHorizontalPortScan-Attack']
		temp_df = temp_df.sample(n = 5)
		final_data = final_data.append(temp_df)
		temp_df = df.loc[df['detailed-label'] == 'C&C-Torii']
		temp_df = temp_df.sample(n = 16)
		final_data = final_data.append(temp_df)
		temp_df = df.loc[df['detailed-label'] == 'C&C-FileDownload']
		temp_df = temp_df.sample(n = 53)
		final_data = final_data.append(temp_df)
		temp_df = df.loc[df['detailed-label'] == 'Okiru-Attack']
		temp_df = temp_df.sample(n = 3)
		final_data = final_data.append(temp_df)
		temp_df = df.loc[df['detailed-label'] == 'FileDownload']
		temp_df = temp_df.sample(n = 18)
		final_data = final_data.append(temp_df)
		temp_df = df.loc[df['detailed-label'] == 'C&C-HeartBeat-FileDownload']
		temp_df = temp_df.sample(n = 11)
		final_data = final_data.append(temp_df)
		temp_df = df.loc[df['detailed-label'] == 'C&C-HeartBeat-Attack']
		temp_df = temp_df.sample(n = 834)
		final_data = final_data.append(temp_df)
		temp_df = df.loc[df['detailed-label'] == 'C&C-PartOfAHorizontalPortScan']
		temp_df = temp_df.sample(n = 888)
		final_data = final_data.append(temp_df)
		temp_df = df.loc[df['detailed-label'] == 'C&C-Mirai']
		temp_df = temp_df.sample(n = 2)
		final_data = final_data.append(temp_df)

		final_data = final_data.drop(final_data.columns[0], axis=1)
		final_data = final_data.drop(final_data.columns[-1], axis=1)
		final_data.to_csv("unseen_testing.csv")
		print(final_data.info())

if __name__ == '__main__':
		get_small_dataset()
