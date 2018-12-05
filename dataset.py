from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import ast
import cv2
import numpy as np
import pandas as pd
import horovod.torch as hvd
from PIL import Image


class SplitDataset(Dataset):

    def __init__(self, filename, mode='train', size=256, transform=None, bw_mode='black'):
        self.mode = mode
        self.size = size
        if self.mode == 'train':
            self.doodle = pd.read_csv(filename, usecols=['drawing', 'y'])
        else:
            self.doodle = pd.read_csv(filename, usecols=['drawing'])
        self.transform = transform
        self.bw_mode = bw_mode

    @staticmethod
    def n_color(n):
        x = np.arange(0, 255, 255 / n, dtype=np.uint8)
        x = cv2.applyColorMap(x, cv2.COLORMAP_RAINBOW)[:, 0, ::-1]
        return x.tolist()

    @staticmethod
    def _draw(strokes, size=256, lw=2, bw_mode='random'):
        img = np.zeros((size, size, 3), np.uint8)
        if bw_mode == 'random':
            img += np.random.randint(0, 2) * 255
        elif bw_mode == 'white':
            img += 255
        colors = SplitDataset.n_color(len(strokes))
        for t, stroke in enumerate(strokes):
            color = colors[t]
            for i in range(len(stroke[0]) - 1):
                cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]),
                         color=color, thickness=lw, lineType=cv2.LINE_AA)
        return Image.fromarray(img)

    def __len__(self):
        return len(self.doodle)

    def __getitem__(self, idx):
        raw_strokes = ast.literal_eval(self.doodle.drawing[idx])
        sample = SplitDataset._draw(raw_strokes, size=self.size, lw=2, bw_mode=self.bw_mode)
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
