import pandas as pd
from os.path import join
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Split Dataset')
parser.add_argument('--dataset', type=str, default='KingsCollege', help='select Dataset type')
parser.add_argument('--file', type=str, default='train_data.csv', help='select Dataset to split train, test, validation')

def db_query_split(dataframe, mode, columns):
	datas = dataframe.values.tolist()
	db, query = train_test_split(datas, test_size=0.5, random_state=42)
	db = pd.DataFrame(db, columns=columns)
	query = pd.DataFrame(query, columns=columns)
	db.to_csv(join(dataset, 'csv', mode + '_db_data.csv'), index=False)
	query.to_csv(join(dataset, 'csv',  mode + '_query_data.csv'), index=False)

if __name__ == '__main__':
	opt = parser.parse_args()
	dataset = opt.dataset
	csv_file = join(dataset, 'csv', opt.file)
	df = pd.read_csv(csv_file)
	columns = df.columns
	data_list = df.values.tolist()
	train_data, temp = train_test_split(data_list, test_size=0.5, random_state=42)
	test_data, val_data = train_test_split(temp, test_size=0.4377, random_state=42)

	train_df = pd.DataFrame(train_data, columns=df.columns)
	test_df = pd.DataFrame(test_data, columns=df.columns)
	val_df = pd.DataFrame(val_data, columns=df.columns)

	mode = 'train'
	db_query_split(train_df, mode, columns)
	mode = 'val'
	db_query_split(val_df, mode, columns)

	test_df.to_csv(join(dataset, 'csv', 'test_db_data.csv'), index=False)
