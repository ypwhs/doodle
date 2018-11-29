from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import ast
import cv2
import numpy as np
import pandas as pd
import horovod.torch as hvd
from PIL import Image


class SplitDataset(Dataset):

    def __init__(self, filename, mode='train', size=256, transform=None):
        self.mode = mode
        self.size = size
        self.doodle = pd.read_csv(filename, usecols=['drawing', 'y'])
        self.transform = transform

    @staticmethod
    def _draw(raw_strokes, size=256, lw=6, time_color=True):
        img = np.zeros((size, size, 3), np.uint8)
        for t, stroke in enumerate(raw_strokes):
            for i in range(len(stroke[0]) - 1):
                color = 255 - min(t, 10) * 13 if time_color else 255
                _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                             (stroke[0][i + 1], stroke[1][i + 1]), color, lw)

        return Image.fromarray(img)

    def __len__(self):
        return len(self.doodle)

    def __getitem__(self, idx):
        raw_strokes = ast.literal_eval(self.doodle.drawing[idx])
        sample = self._draw(raw_strokes, size=self.size, lw=2, time_color=True)
        if self.transform:
            sample = self.transform(sample)
        if self.mode == 'train':
            return sample, self.doodle.y[idx]
        else:
            return sample


def get_split_dataloader(filename, width=256, batch_size=128, transform=None, num_workers=4):
    dataset = SplitDataset(filename, size=width, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return loader
