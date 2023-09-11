import pandas as pd
from os.path import join
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Split Dataset')
parser.add_argument('--dataset', type=str, default='iiclab', help='select Dataset type')
parser.add_argument('--file', type=str, default='whole_synchronized_data.csv', help='select Dataset type')

def db_query_split(dataframe, mode):
	timestamp = dataframe['time']
	df = pd.DataFrame(columns=dataframe.columns)

	for i, time in enumerate(timestamp):
		if i%20==0:    # 20초마다 데이터 저장
			df = pd.concat([df, dataframe.loc[timestamp==time]])
			dataframe = dataframe[dataframe['time'] != time]

	df.to_csv(join(dataset, 'csv',  mode + '_query_data.csv'), index=False)
	dataframe.to_csv(join(dataset, 'csv', mode + '_db_data.csv'), index=False)

if __name__ == '__main__':
	opt = parser.parse_args()
	dataset = opt.dataset
	csv_file = join(dataset, 'csv', opt.file)
	df = pd.read_csv(csv_file)
	data_list = df.values.tolist()
	train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

	train_df = pd.DataFrame(train_data, columns=df.columns)
	test_df = pd.DataFrame(test_data, columns=df.columns)

	mode = 'train'
	db_query_split(train_df, mode)
	mode = 'test'
	db_query_split(test_df, mode)
