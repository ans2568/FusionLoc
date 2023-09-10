import pandas as pd
import os

path = '7scenes_chess'

test_data = pd.read_csv(os.path.join(path, 'test/csv/test.csv'))
train_data = pd.read_csv(os.path.join(path, 'train/csv/train.csv'))

combined_data = pd.concat([test_data, train_data], ignore_index=True)

combined_data.to_csv(os.path.join(path, 'whole_data.csv'), index=False)