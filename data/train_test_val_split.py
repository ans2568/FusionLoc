import pandas as pd
from os.path import join
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Split Dataset')
parser.add_argument('--dataset', type=str, default='KingsCollege', help='select Dataset type')
parser.add_argument('--file', type=str, default='train_data.csv', help='select Dataset to split train, test, validation')
parser.add_argument('--tempsize', type=float, default=0.3, help='the size of temp_data from whole_data. temp_data is data to split between test_data and val_data.')
parser.add_argument('--valsize', type=float, default=0.3, help='the size of val_data from temp_data')
parser.add_argument('--querysize', type=float, default=0.3, help='the size of query_data from csv_files(train, test, val).')

def db_query_split(dataframe, mode, columns):
	datas = dataframe.values.tolist()
	db, query = train_test_split(datas, test_size=0.3, random_state=42)
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
	train_data, temp = train_test_split(data_list, test_size=opt.tempsize, random_state=42)
	test_data, val_data = train_test_split(temp, test_size=opt.valsize, random_state=42)

	train_df = pd.DataFrame(train_data, columns=df.columns)
	test_df = pd.DataFrame(test_data, columns=df.columns)
	val_df = pd.DataFrame(val_data, columns=df.columns)

	mode = 'train'
	db_query_split(train_df, mode, columns)
	mode = 'test'
	db_query_split(test_df, mode, columns)
	mode = 'val'
	db_query_split(val_df, mode, columns)
