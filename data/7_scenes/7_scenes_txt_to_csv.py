import os
import csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

# 디렉토리 및 파일 이름 설정
input_directory = "seq-06"
output_file = input_directory + ".csv"

# CSV 파일을 작성하기 위한 헤더 설정
csv_header = ["time", "x", "y", "z", "qw", "qx", "qy", "qz", "image_path"]

# CSV 파일 생성 및 헤더 쓰기
with open(output_file, mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_header)

# 디렉토리 내의 모든 .pose.txt 파일 처리
for file_name in os.listdir(input_directory):
    if file_name.endswith(".pose.txt"):
        full_file_path = os.path.join(input_directory, file_name)

        # 파일 내용 읽기
        with open(full_file_path, "r") as pose_file:
            lines = pose_file.readlines()

        # Transformation 데이터 추출
        transformation = np.array([list(map(float, line.split())) for line in lines])
        rotation_mat = transformation[:3, :3]
        translation_mat = transformation[:3, 3]
        R = Rotation.from_matrix(rotation_mat)
        quaternion = R.as_quat() # return [x, y, z, w]
        qx = 0
        qy = 0
        qz = quaternion[2]
        qw = quaternion[3]
        px = translation_mat[0]
        py = translation_mat[1]
        pz = 0
        time = full_file_path.split('.pose.txt')[0]
        image_path = time + '.color.png'
        data = [time, px, py, pz, qw, qx, qy, qz, image_path]

        # # CSV 파일에 데이터 추가
        with open(output_file, mode="a", newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data)