import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image

class MNISTDataset(Dataset):
    def __init__(self, csv_path, is_train=True, transform=None):
        data = pd.read_csv(csv_path)
        if is_train:
            self.labels = data["label"].values
            self.images = data.drop("label", axis=1).values.reshape(-1, 28, 28)
        else:
            self.labels = None
            self.images = data.values.reshape(-1, 28, 28)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.uint8)
        image = Image.fromarray(image)  # 转换为PIL图像
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image  # 测试集无标签
