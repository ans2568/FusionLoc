import os
import csv
import argparse

import math
import cv2
import numpy as np
import open3d as o3d
from datetime import datetime

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter

bag_file = '/path/to/ros2_ws/bag/data.bag'
csv_dir = '../data/gazebo_dataset/csv'
lidar_dir = '../data/gazebo_dataset/lidar'
camera_dir = '../data/gazebo_datasetcamera'

if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
if not os.path.exists(lidar_dir):
    os.makedirs(lidar_dir)
if not os.path.exists(camera_dir):
    os.makedirs(camera_dir)

storage_options = StorageOptions(uri=bag_file, storage_id="sqlite3")
converter_options = ConverterOptions('cdr', 'cdr')
storage_filter = StorageFilter(topics=['/scan', '/camera/image_raw/compressed', '/tf'])

# open the bag file
reader = SequentialReader()
reader.open(storage_options, converter_options)
reader.set_filter(storage_filter)
lidar_topic = '/scan'
camera_topic = '/camera/image_raw/compressed'
tf_topic = '/tf'

lidar_time = None
camera_time = None
tf_time = None

csv_file = open(os.path.join(csv_dir, "whole_data.csv"), mode="w")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["time", "encoder_L", "encoder_R", "angular_vel_x", "angular_vel_y", "angular_vel_z", "linear_acc_x", "linear_acc_y", "linear_acc_z", "pose_x","pose_y", "pose_theta", "GT_pose_x", "GT_pose_y", "GT_pose_theta", "GPS_lon", "GPS_lat", "gt", "image_path", "lidar_path"])

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
            csv_writer.writerow([sec, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", os.path.join(lidar_dir, file_name)])
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
            csv_writer.writerow([sec, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", os.path.join(camera_dir, file_name), ""])
            camera_time = timestamp

    elif topic == tf_topic:
        type = get_message('tf2_msgs/msg/TFMessage')
        msg = deserialize_message(data, type)
        if not tf_time:
            tf_time = timestamp
        if int(timestamp.timestamp() - tf_time.timestamp()) >= 1:
            for transform in msg.transforms:
                if transform.child_frame_id == "base_footprint" and transform.header.frame_id == "odom":
                    pose_x = transform.transform.translation.x
                    pose_y = transform.transform.translation.y
                    quaternion = transform.transform.rotation
                    theta = math.atan2(2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y), 
                                    1 - 2 * (quaternion.y**2 + quaternion.z**2))
                    csv_writer.writerow([sec, "", "", "", "", "", "", "", "", "", "", "", pose_x, pose_y, theta, "", "", "", "", ""])
                    tf_time = timestamp
