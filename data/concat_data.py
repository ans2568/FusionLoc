import os
import csv

dataset = '7_scenes'

def append_csv(file):
    d_list = []
    with open(file, mode='r') as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            d_list.append(parts)
    if dataset == '7_scenes':
        sorted_data_list = sorted(d_list, key=lambda x: x[0][-6:])
        return sorted_data_list
    else:
        return d_list

file_name = 'seq-0'
csv_format = '.csv'
csv_header = ['time', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'image_path']

file_list = []

if dataset == '7_scenes':
    for i in range(1, 7):
        input_file = os.path.join(dataset, 'csv', file_name + str(i) + csv_format)
        file_list.append(input_file)
else:
    input_file = os.path.join(dataset, 'csv', 'test_data' + csv_format)
    input_file2 = os.path.join(dataset, 'csv', 'train_data' + csv_format)
    file_list.append(input_file)
    file_list.append(input_file2)

output_file = os.path.join(dataset, 'csv', 'whole_data.csv')

data_list = []

for file in file_list:
    data_list.append(append_csv(file))

with open(output_file, mode='w', newline='') as output:
    csv_writer = csv.writer(output)
    csv_writer.writerow(csv_header)
    for datas in data_list:
        for data in datas:
            csv_writer.writerow(data)
