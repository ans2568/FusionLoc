import os
import csv
import cv2
import numpy as np

mode = 'test'
scene = '7scenes_chess'
dataset_dir = os.path.join(scene, mode)
path = os.path.join(dataset_dir, 'poses')
csv_dir = os.path.join(dataset_dir, 'csv')
poses_file = os.listdir(path)
poses_file.sort()

for idx, file in enumerate(poses_file):
    T =np.loadtxt(os.path.join(path, file))
    H = np.delete(T, -1, axis=0)
    R = H[:3, :3]
    t = H[:, 3]
    rot_vec = cv2.Rodrigues(R)[0]
    est_theta = rot_vec[2][0]
    column = ['time', 'encoder_L','encoder_R','angular_vel_x', 'angular_vel_y','angular_vel_z','linear_acc_x',
                'linear_acc_y','linear_acc_z','pose_x','pose_y','pose_theta','GT_pose_x','GT_pose_y','GT_pose_theta',
                'GPS_lon','GPS_lat','gt','image_path','lidar_path']

    sequence = file.replace('.pose.txt', '')
    data = np.asanyarray([sequence, None, None, None, None, None, None, None, None, None,
                          None, None, t[0], t[1], est_theta, None, None, None,
                           mode + '/rgb/' + sequence + '.color.png', None])
    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)

    with open(os.path.join(csv_dir, mode + '.csv'), mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        if idx == 0:
            csv_writer.writerow(column)
        csv_writer.writerow(data)