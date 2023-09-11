import pandas as pd
from os.path import join

class Struct:
    def __init__(self, dir, dataset):
        self.struct = []
        csv_dir = join(dir, dataset, 'csv')
        if dataset == 'gazebo' or dataset == 'iiclab':
            self.df = pd.read_csv(join(csv_dir, 'whole_synchronized_data.csv'))
        elif dataset == 'NIA':
            self.df = pd.read_csv(join(csv_dir, 'whole_data.csv'))

    def append(self, timestamp:str):
        self.time=(self.df.loc[self.df['time'] == timestamp, 'time'].values[0])
        self.gt_x=(self.df.loc[self.df['time'] == timestamp, 'x'].values[0])
        self.gt_y=(self.df.loc[self.df['time'] == timestamp, 'y'].values[0])
        self.gt_theta=(self.df.loc[self.df['time'] == timestamp, 'theta'].values[0])
        self.image_path=(self.df.loc[self.df['time'] == timestamp, 'image_path'].values[0])
        self.lidar_path=(self.df.loc[self.df['time'] == timestamp, 'lidar_path'].values[0])
        list = [self.time, self.gt_x, self.gt_y, self.gt_theta, self.image_path, self.lidar_path]
        self.struct.append(list)

    def get(self):
        return self.struct