import os
import json
import sys

import numpy as np
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import *
import tqdm

from model import AlexNet

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform=transforms.Compose(
    [transforms.Resize((28, 28)),  # cannot 28, must (28, 28)
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



csv_path = "F:/虚拟桌面/TUNNELS/第三周/digit-recognizer/test.csv"
df = pd.read_csv(csv_path)

# 获取某列的唯一值（例如列名为 "category"）

processed_images = []
for img_array in df:
    # 将按行展开的像素值重新调整为 28x28 图像
    img_array = img_array.reshape(28, 28)
    
    # 将灰度图像转换为 RGB 图像（如果需要）
    # 这里假设图像原本是灰度图像，将其转换为 RGB 图像
    img_array = np.stack((img_array,) * 3, axis=-1)
    
    # 将 NumPy 数组转换为 PIL.Image 对象
    pil_image = Image.fromarray(img_array.astype(np.uint8))
    
    # 应用预处理
    if is_train:
        processed_image = data_transform["train"](pil_image)
    else:
        processed_image = data_transform["val"](pil_image)
    
    # 将处理后的图像添加到列表中
    processed_images.append(processed_image)

# 将处理后的图像转换为 Tensor
features_tensor = torch.stack(processed_images).float()

# 重新调整数据形状以满足 AlexNet 的输入要求
# X_F = df.values.reshape(-1, 1, 28, 28)  # 提取特征（NumPy 数组）


                    
# 转换为 Tensor（注意数据类型）
features_tensor = torch.tensor(features_tensor, dtype=torch.float32)  # 特征通常是 float

from torch.utils.data import TensorDataset

predict_data = TensorDataset(features_tensor)  # 特征和标签配对

# read class_indict
json_path='./topics_dict.json'
assert os .path.exists(json_path),"file:{} not exist".format(json_path)

with open(json_path,"r") as f:
    topics_dict=json.load(f)

# create model
model = AlexNet(num_classes=10).to(device)

# load model weights
weights_path="./vgg16Net.pth"
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
model.load_state_dict(torch.load(weights_path))

batch_size=32
dataloader = torch.utils.data.DataLoader(
    predict_data,batch_size=batch_size,shuffle=False,num_workers=0#windows系统下设置为零
)

# for data in dataloader:
#     model.eval()
#     with torch.no_grad():
#         # predict class
#         output = torch(model(data))
#         predict = torch.softmax(output, dim=0)
#         predict_cla = torch.argmax(predict).numpy()

#     print_res = "class: {}   prob: {:.3}".format(topics_dict[str(predict_cla)],
#                                                     predict[predict_cla].numpy())
#     plt.title(print_res)
#     for i in range(len(predict)):
#         print("class: {:10}   prob: {:.3}".format(topics_dict[str(i)],
#                                                     predict[i].numpy()))
#     plt.show()

predict_result=pd.DataFrame()

for data in dataloader:
    model.eval()
    with torch.no_grad():
        # 将数据移动到正确的设备上
        inputs = data[0].to(device)
        # 直接使用模型进行预测
        output = model(inputs)
        predict = torch.softmax(output, dim=1)  # dim=1 因为是批次数据
        predict_cla = torch.argmax(predict, dim=1).cpu().numpy()  # 对每个样本取最大值
        predict_cla_df = pd.DataFrame(predict_cla, columns=['label'])
        predict_result=pd.concat([predict_result,predict_cla_df],ignore_index=True)

    # 因为是批次数据,这里需要循环处理每个预测结果
    for i in range(len(predict_cla)):
        print_res = "class: {}   prob: {:.3f}".format(#{:.3f} 的作用是将预测的概率值格式化为小数点后三位
            #: 表示开始格式说明。.3 表示小数点后保留三位。f 表示格式化为浮点数。
            topics_dict[str(predict_cla[i])],
            predict[i][predict_cla[i]].cpu().numpy()
        )
        print(print_res)

predict_result.to_csv('prediction.csv') 

