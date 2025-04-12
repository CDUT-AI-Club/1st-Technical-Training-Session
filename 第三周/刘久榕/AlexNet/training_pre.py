""" import pandas as pd
import os
temp_pd=pd.DataFrame()

def read_file(path_in):
    text=None
    try:
        f=open(path_in,"r",encoding="UTF-8")
        text=f.read()
        f.close()
    except:
        print("hey",path_in,"does not exist!!")
        pass
    return text


csv=read_file("F:/虚拟桌面/TUNNELS/第三周/digit-recognizer/train.csv")
data = pd.read_csv(csv)

 """
# import pandas as pd

# csv_path = "F:/虚拟桌面/TUNNELS/第三周/digit-recognizer/train.csv"
# try:
#     data = pd.read_csv(csv_path)
# except FileNotFoundError:
#     print("hey", csv_path, "does not exist!!")
#     data = None


# topics=list(data.label.unique())#查找数组中的唯一（unique）元素
# topics = sorted(topics)
# topics_dict = {topic: 0 for topic in topics}


# 

# import torch
# print(torch.__version__)            # 应输出 2.3.1
# print(torch.cuda.is_available()) 
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("using {} device.".format(device))   # 应返回 True（GPU 可用）
# import torchvision
# import torchaudio
# print(torchvision.__version__)      # 应输出 0.18.1
# print(torchaudio.__version__)       # 应输出 2.3.1


# import pandas as pd

# # 读取 CSV 文件
# df = pd.read_csv("F:/虚拟桌面/TUNNELS/第三周/digit-recognizer/test.csv")

# # 获取某列的唯一值（例如列名为 "category"）
# unique_elements = df["label"].unique()

# sorted_unique_elements = sorted(unique_elements)
# # 生成字典：键为自然数，值为元素
# index_to_element = {idx: element for idx, element in enumerate(sorted_unique_elements)}

# index_to_element = {k: int(v) for k, v in index_to_element.items()}
# print(index_to_element)

# features = df.drop("label", axis=1).values  # 提取特征（NumPy 数组）
# labels = df["label"].values                 # 提取标签（NumPy 数组）

# import torch

# # 转换为 Tensor（注意数据类型）
# features_tensor = torch.tensor(features, dtype=torch.float32)  # 特征通常是 float
# labels_tensor = torch.tensor(labels, dtype=torch.long)         # 标签通常是 int/long

# from torch.utils.data import TensorDataset

# dataset = TensorDataset(features_tensor, labels_tensor)  # 特征和标签配对


# from torch.utils.data import DataLoader

# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


import os
import sys
import json

import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from model import AlexNet

def pre_processing(ad,anwser):
    csv_path = ad
    df = pd.read_csv(csv_path)

    # 获取某列的唯一值（例如列名为 "category"）
    unique_elements = df["label"].unique()

    sorted_unique_elements = sorted(unique_elements)
    # 生成字典：键为自然数，值为元素
    if anwser == "yes":
        index_to_element = {idx: element for idx, element in enumerate(sorted_unique_elements)}

        data_serializable = {k: int(v) for k, v in index_to_element.items()}
        json_str=json.dumps(data_serializable,indent=4)#indent指定缩进的空格数
        import json
        with open('topics_dict.json', 'w') as json_file:
            json.dump(data_serializable, json_file)


    features = df.drop("label", axis=1).values  # 提取特征（NumPy 数组）
    labels = df["label"].values                 # 提取标签（NumPy 数组）

    # 转换为 Tensor（注意数据类型）
    features_tensor = torch.tensor(features, dtype=torch.float32)  # 特征通常是 float
    labels_tensor = torch.tensor(labels, dtype=torch.long)         # 标签通常是 int/long

    from torch.utils.data import TensorDataset

    dataset = TensorDataset(features_tensor, labels_tensor)  # 特征和标签配对
    return dataset

data=pre_processing("F:/虚拟桌面/TUNNELS/第三周/digit-recognizer/train.csv","no")

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    data,batch_size=batch_size,shuffle=True,num_workers=0#windows系统下设置为零
)

train_bar=tqdm(train_loader,file=sys.stdout)
for step,data in enumerate(train_bar):
    images,labels=data