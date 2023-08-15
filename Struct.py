import pandas as pd

class Struct:
    def __init__(self, dataset):
        self.struct = []
        self.struct_rest = []
        if dataset == 'gazebo':
            self.df = pd.read_csv('data/gazebo_dataset/csv/whole_synchronized_data.csv')
        elif dataset == 'NIA':
            self.df = pd.read_csv('data/NIA/csv/whole_data.csv')
        elif dataset == 'iiclab':
            self.df = pd.read_csv('data/iiclab_real/csv/whole_synchronized_data.csv')

    def append(self, timestamp:str):
        self.time=(self.df.loc[self.df['time'] == timestamp, 'time'].values[0])
        self.gt_x=(self.df.loc[self.df['time'] == timestamp, 'GT_pose_x'].values[0])
        self.gt_y=(self.df.loc[self.df['time'] == timestamp, 'GT_pose_y'].values[0])
        self.gt_theta=(self.df.loc[self.df['time'] == timestamp, 'GT_pose_theta'].values[0])
        self.image_path=(self.df.loc[self.df['time'] == timestamp, 'image_path'].values[0])
        self.lidar_path=(self.df.loc[self.df['time'] == timestamp, 'lidar_path'].values[0])
        list = [self.time, self.gt_x, self.gt_y, self.gt_theta, self.image_path, self.lidar_path]
        self.struct.append(list)

    def get(self):
        return self.struct
