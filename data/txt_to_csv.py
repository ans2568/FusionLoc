import csv
import math

input_txt = 'KingsCollege/dataset_train.txt'
output_csv = 'KingsCollege/csv/train_data.csv'

csv_header = ['time', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'image_path']

with open(output_csv, mode='w', newline='') as file:
	csv_writer = csv.writer(file)
	csv_writer.writerow(csv_header)

	with open(input_txt, mode='r') as txt_file:
		for line in txt_file:
			data = line.strip().split()
			image_file = data[0]
			time = image_file[:-4]
			x, y, z, w, p, q, r = map(float, data[1:])
			theta = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
			z = 0
			p = 0
			q = 0
			r = theta
			csv_writer.writerow([time, x, y, z, w, p, q, r, image_file])

print(f'Data from "{input_txt}" has been processed and saved to "{output_csv}"')