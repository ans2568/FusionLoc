import os
import csv
import argparse

import cv2
import numpy as np
import open3d as o3d
from datetime import datetime

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter


parser = argparse.ArgumentParser(description='Converting a ROS bag file to a CSV file')
parser.add_argument('--bag', type=str, default='/path/to/ros2_ws/bag/data.bag', help='ROS bag file')
parser.add_argument('--csvDir', type=str, default='../data/jackal_dataset/csv', help='The directory to save CSV file')
parser.add_argument('--lidarDir', type=str, default='../data/jackal_dataset/lidar', help='The directory to save pcd file')
parser.add_argument('--cameraDir', type=str, default='../data/jackal_dataset/camera', help='The directory to save png file')
parser.add_argument('--cameraTopic', type=str, default='/image_raw/compressed', help='Camera topic name in rosbag')
parser.add_argument('--lidarTopic', type=str, default='/scan', help='PointCloud topic name in rosbag')
parser.add_argument('--odometryTopic', type=str, default='/odom', help='Odometry topic name in rosbag')

if __name__ == "__main__":
    opt = parser.parse_args()

    csv_dir = opt.csvDir
    lidar_dir = opt.lidarDir
    camera_dir = opt.cameraDir

    lidar_topic = opt.lidarTopic
    camera_topic = opt.cameraTopic
    odometry_topic = opt.odometryTopic

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    if not os.path.exists(lidar_dir):
        os.makedirs(lidar_dir)
    if not os.path.exists(camera_dir):
        os.makedirs(camera_dir)

    storage_options = StorageOptions(uri=opt.bag, storage_id="sqlite3")
    converter_options = ConverterOptions('cdr', 'cdr')
    storage_filter = StorageFilter(topics=[lidar_topic, camera_topic, odometry_topic])

    # open the bag file
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    reader.set_filter(storage_filter)

    lidar_time = None
    camera_time = None
    odom_time = None

    csv_file = open(os.path.join(csv_dir, "whole_data.csv"), mode="w")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time", "x", "y", "z", "qw","qx", "qy", "qz", "image_path", "lidar_path"])

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        timestamp = datetime.fromtimestamp(timestamp/ 1e9)
        millisecond = str(timestamp.timestamp()).split('.')[1]
        sec = "{0}".format(str(timestamp.year)+str(timestamp.month)+str(timestamp.day)+"_"+str(timestamp.hour)+str(timestamp.minute).zfill(2)+str(timestamp.second).zfill(2))
        if topic == lidar_topic:
            type = get_message('sensor_msgs/msg/LaserScan')
            msg = deserialize_message(data, type)
            if not lidar_time:
                lidar_time = timestamp
            if int(timestamp.timestamp() - lidar_time.timestamp()) >= 1:
                file_name = "{0}.pcd".format(sec)
                angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
                xs = np.array(msg.ranges) * np.cos(angles)
                ys = np.array(msg.ranges) * np.sin(angles)
                zs = np.zeros_like(np.array(msg.ranges))
                points = np.column_stack((xs, ys, zs))
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(os.path.join(lidar_dir, file_name), pcd)
                csv_writer.writerow([sec, "", "", "", "", "", "", "", "", os.path.join(lidar_dir, file_name)])
                lidar_time = timestamp

        elif topic == camera_topic:
            type = get_message('sensor_msgs/msg/CompressedImage')
            msg = deserialize_message(data, type)
            if not camera_time:
                camera_time = timestamp
            if int(timestamp.timestamp() - camera_time.timestamp()) >= 1:
                file_name = "{0}.png".format(sec)
                np_arr = np.frombuffer(msg.data, np.uint8)
                image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join(camera_dir, file_name), image_np)
                csv_writer.writerow([sec, "", "", "", "", "", "", "", os.path.join(camera_dir, file_name), ""])
                camera_time = timestamp

        elif topic == odometry_topic:
            type = get_message('nav_msgs/msg/Odometry')
            msg = deserialize_message(data, type)
            if not odom_time:
                odom_time = timestamp
            if int(timestamp.timestamp() - odom_time.timestamp()) >= 1:
                if msg.child_frame_id == "base_footprint" and msg.header.frame_id == "odom":
                    position = msg.pose.pose.position
                    orientation = msg.pose.pose.orientation
                    csv_writer.writerow([sec, position.x, position.y, position.z, orientation.w, orientation.x, orientation.y, orientation.z, "", ""])
                    odom_time = timestamp

    print("Finish to make csv file")
    print("Start to synchronize csv file")
    # finish to make csv file
    import pandas as pd

    df = pd.read_csv(os.path.join(opt.csvDir, 'whole_data.csv'))

    grouped = df.groupby(['time'], as_index=False)
    result = grouped.first()
    result_filtered = result.dropna(subset=['time', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'image_path', 'lidar_path'])
    result_filtered.loc[:, ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']] = result_filtered.loc[:, ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']].apply(lambda x: x.apply(lambda y: f"{float(y):.16f}"))

    result_filtered.to_csv(os.path.join(opt.csvDir, 'whole_synchronized_data.csv'), index=False)