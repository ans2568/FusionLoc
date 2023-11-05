import pandas as pd
from os.path import join
from pathlib import Path

root = Path(__file__).parent.parent

class Struct:
    def __init__(self, dataset):
        self.struct = []
        self.dataset = dataset
        if dataset == 'gazebo' or dataset == 'iiclab':
            self.df = pd.read_csv(join(root, 'data', dataset, 'csv', 'whole_synchronized_data.csv'))
        elif dataset == 'NIA' or self.dataset == 'KingsCollege' or dataset == '7_scenes':
            self.df = pd.read_csv(join(root, 'data', dataset, 'csv', 'whole_data.csv'))

    def append(self, timestamp:str):
        if self.dataset == 'iiclab':
            self.time=(self.df.loc[self.df['time'] == timestamp, 'time'].values[0])
            self.gt_x=(self.df.loc[self.df['time'] == timestamp, 'x'].values[0])
            self.gt_y=(self.df.loc[self.df['time'] == timestamp, 'y'].values[0])
            self.gt_theta=(self.df.loc[self.df['time'] == timestamp, 'qz'].values[0])
            self.image_path=(self.df.loc[self.df['time'] == timestamp, 'image_path'].values[0])
            self.lidar_path=(self.df.loc[self.df['time'] == timestamp, 'lidar_path'].values[0])
        elif self.dataset == 'KingsCollege' or self.dataset == '7_scenes':
            self.time=(self.df.loc[self.df['time'] == timestamp, 'time'].values[0])
            self.gt_x=(self.df.loc[self.df['time'] == timestamp, 'x'].values[0])
            self.gt_y=(self.df.loc[self.df['time'] == timestamp, 'y'].values[0])
            self.gt_theta=(self.df.loc[self.df['time'] == timestamp, 'qz'].values[0])
            self.image_path=(self.df.loc[self.df['time'] == timestamp, 'image_path'].values[0])
        else:
            self.time=(self.df.loc[self.df['time'] == timestamp, 'time'].values[0])
            self.gt_x=(self.df.loc[self.df['time'] == timestamp, 'GT_pose_x'].values[0])
            self.gt_y=(self.df.loc[self.df['time'] == timestamp, 'GT_pose_y'].values[0])
            self.gt_theta=(self.df.loc[self.df['time'] == timestamp, 'GT_pose_theta'].values[0])
            self.image_path=(self.df.loc[self.df['time'] == timestamp, 'image_path'].values[0])
            self.lidar_path=(self.df.loc[self.df['time'] == timestamp, 'lidar_path'].values[0])
        if self.dataset == 'KingsCollege' or self.dataset == '7_scenes':
            list = [self.time, self.gt_x, self.gt_y, self.gt_theta, self.image_path]
        else:
            list = [self.time, self.gt_x, self.gt_y, self.gt_theta, self.image_path, self.lidar_path]
        self.struct.append(list)

    def get(self):
        return self.struct
