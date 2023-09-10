import os
import pandas as pd

scene = '7scenes_chess'
mode = 'train'
path = os.path.join(scene, mode, 'csv')

data = pd.read_csv(os.path.join(path,mode+'.csv'), index_col=False)

timestamp = data['time']
train_dataset = pd.DataFrame(columns=data.columns)

for i, time in enumerate(timestamp):
    if i%20==0:    # 10초마다 데이터 저장
        train_dataset = pd.concat([train_dataset, data.loc[timestamp==time]])
        data = data[data['time'] != time]

train_dataset.to_csv(os.path.join(path,  mode + '_query_data.csv'), index=False)
data.to_csv(os.path.join(path, mode + '_db_data.csv'), index=False)