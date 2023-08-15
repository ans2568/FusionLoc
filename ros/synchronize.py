import pandas as pd

df = pd.read_csv('../data/gazebo_dataset/csv/whole_data.csv')

grouped = df.groupby(['time'], as_index=False)
result = grouped.first()
result_filtered = result.dropna(subset=['time', 'GT_pose_x', 'GT_pose_y', 'GT_pose_theta', 'image_path', 'lidar_path'])
result_filtered.loc[:, ['GT_pose_x', 'GT_pose_y', 'GT_pose_theta']] = result_filtered.loc[:, ['GT_pose_x', 'GT_pose_y', 'GT_pose_theta']].apply(lambda x: x.apply(lambda y: f"{float(y):.16f}"))

result_filtered.to_csv('../data/gazebo_dataset/csv/whole_synchronized_data.csv', index=False)