import pandas as pd

def read_df():
	df = pd.read_csv('total_datset.csv')
	return df


def find_names(df):
	df = df.drop_duplicates(subset=['detailed-label'])
	attacks = df['detailed-label'].tolist()
	return attacks

#Prints attack family and number of records
def get_info():
	df = read_df()
	attacks = find_names(df)
	final_data = pd.DataFrame()
	for attack in attacks:
		print(attack)
		temp_df = df.loc[df['detailed-label'] == attack]
		print(len(temp_df))

if __name__ == '__main__':
	get_info()


