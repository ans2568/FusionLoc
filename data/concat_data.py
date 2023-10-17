import csv

input_file = 'KingsCollege/csv/test_db_data.csv'
input_file2 = 'KingsCollege/csv/test_query_data.csv'
output_file = 'KingsCollege/csv/whole_data.csv'

csv_header = ['time', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'image_path']

data_list = []

with open(input_file, mode='r') as input:
	for line in input:
		parts = line.strip().split(',')
		data_list.append(parts)

with open(input_file2, mode='r') as input2:
	for line in input2:
		parts = line.strip().split(',')
		data_list.append(parts)

with open(output_file, mode='w', newline='') as output:
	csv_writer = csv.writer(output)
	csv_writer.writerow(csv_header)
	for data in data_list:
		csv_writer.writerow(data)