import numpy as np
from scipy.spatial.transform import Rotation
import os

cur_file = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_file)

txt_data = []

def parse_txt(dir):
    global txt_data

    files = os.listdir(dir)
    for file in files:
        if (file.endswith('.txt')):
            image_path = os.path.join(dir, file[:-9] + '_color.png')
            txt_path = os.path.join(dir, file)
            # 파일 읽기
            with open(txt_path, 'r') as file:
                lines = file.readlines()

            # 텍스트 파일에서 읽은 값을 4x4 NumPy 변환 행렬로 변환
            transformation_matrix = np.array([list(map(float, line.split())) for line in lines])

            # 3x3 회전 행렬 추출
            rotation_matrix = transformation_matrix[:3, :3]
            translation = transformation_matrix[:3, 3]
            # 회전 행렬을 quaternion으로 변환
            rotation = Rotation.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()
            txt_data.append([image_path[-29:], translation[0], translation[1], translation[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
        elif (file.endswith('.png')):
            before_img = os.path.join(dir, file)
            image_path = os.path.join(dir, file[:-10] + '_color.png')
            os.rename(before_img, image_path)

dirs = os.listdir(cur_dir)
for dir in dirs:
    if dir == 'csv' or dir == 'seq-03' or dir =='seq-05':
        continue
    # if dir == 'csv' or dir =='seq-01' or dir =='seq-02' or dir =='seq-04' or dir =='seq-06':
    #     continue
    else:
        dir_path = os.path.join(cur_dir, dir)
        if os.path.isdir(dir_path):
            parse_txt(dir_path)

output_file = os.path.join(cur_dir, "dataset_train.txt")
with open(output_file, 'w') as file:
    # 파일에 쓸 내용 포맷 (file, x, y, z, qw, qx, qy, qz)
    format_str = "{} {} {} {} {} {} {} {}\n"
    for row in txt_data:
        file.write(format_str.format(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]))