import json
import os
import datetime as dt
from tqdm import tqdm
import pandas as pd
import numpy as np

path = 'split/'
os.makedirs(path, exist_ok=True)


def f2cat(filename: str) -> str:
    return filename.split('.')[0]


class Simplified():
    def __init__(self, input_path='./input'):
        self.input_path = input_path

    def list_all_categories(self):
        files = os.listdir(os.path.join(self.input_path, 'train_simplified'))
        return sorted([f2cat(f) for f in files if 'csv' in f], key=str.lower)

    def read_training_csv(self, category, nrows=None, usecols=None, drawing_transform=False,
                          drop_not_recognized=False):
        df = pd.read_csv(os.path.join(self.input_path, 'train_simplified', category + '.csv'),
                         nrows=nrows, parse_dates=['timestamp'], usecols=usecols)
        if drop_not_recognized:
            df = df[df['recognized'] == True]
            df.drop('recognized', axis=1, inplace=True)
        if drawing_transform:
            df['drawing'] = df['drawing'].apply(json.loads)
        return df


start = dt.datetime.now()
s = Simplified('/home/ypw/data/doodle/')
NCSVS = 100
categories = s.list_all_categories()
print(len(categories))

with tqdm(categories) as pbar:
    for y, cat in enumerate(pbar):
        df = s.read_training_csv(cat)
        df['y'] = y
        df['cv'] = (df.key_id // 10 ** 7) % NCSVS
        for k in range(NCSVS):
            filename = path + 'train_k{}.csv'.format(k)
            chunk = df[df.cv == k]
            chunk = chunk.drop(['key_id'], axis=1)
            if y == 0:
                chunk.to_csv(filename, index=False)
            else:
                chunk.to_csv(filename, mode='a', header=False, index=False)

for k in tqdm(range(NCSVS)):
    filename = path + 'train_k{}.csv'.format(k)
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['rnd'] = np.random.rand(len(df))
        df = df.sort_values(by='rnd').drop('rnd', axis=1)
        df.to_csv(filename, index=False)

end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))
