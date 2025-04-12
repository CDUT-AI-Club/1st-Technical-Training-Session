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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(28),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((28, 28)),  # cannot 28, must (28, 28)
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

csv_path = "F:/虚拟桌面/TUNNELS/第三周/digit-recognizer/train.csv"
df = pd.read_csv(csv_path)
X=df.drop("label",axis=1)
Y=df.label

#划分训练集和数据集
X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.2,random_state=42)

def pre_processing(anwser,data,labels):
    
    # 获取某列的唯一值（例如列名为 "category"）
    if anwser == "yes":
        unique_elements = labels.unique()

        sorted_unique_elements = sorted(unique_elements)
        # 生成字典：键为自然数，值为元素
        
        index_to_element = {idx: element for idx, element in enumerate(sorted_unique_elements)}

        data_serializable = {k: int(v) for k, v in index_to_element.items()}
        json_str=json.dumps(data_serializable,indent=4)#indent指定缩进的空格数
        import json
        with open('topics_dict.json', 'w') as json_file:
            json.dump(data_serializable, json_file)

    # 重新调整数据形状以满足 AlexNet 的输入要求
    X_F = data.values.reshape(-1, 1, 28, 28)  # 提取特征（NumPy 数组）
    Y_F = labels.values                 # 提取标签（NumPy 数组）

    # 转换为 Tensor（注意数据类型）
    features_tensor = torch.tensor(X_F, dtype=torch.float32)  # 特征通常是 float
    labels_tensor = torch.tensor(Y_F, dtype=torch.long)         # 标签通常是 int/long

    from torch.utils.data import TensorDataset

    dataset = TensorDataset(features_tensor, labels_tensor)  # 特征和标签配对
    return dataset

training_data=pre_processing("no",X_train,Y_train)
test_data=pre_processing("no",X_val,Y_val)

train_num=len(training_data)
val_num=len(test_data)

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    training_data,batch_size=batch_size,shuffle=True,num_workers=0#windows系统下设置为零
)


validate_loader=torch.utils.data.DataLoader(
    test_data,batch_size=batch_size,shuffle=True,num_workers=0#windows系统下设置为零
)


print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

 # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))


net=AlexNet(num_classes=10,init_weights=True)

net.to(device)

loss_function=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(),lr=0.0002)#net.parameters()网络中所有参数

epochs=10
save_path='./AlexNet.pth'
best_acc=0.0
train_steps=len(train_loader)
for epoch in  range(epochs):
    #train
    net.train()#使dropout只在训练集中起作用
    running_loss=0.0
    train_bar=tqdm(train_loader,file=sys.stdout)#显示 train_loader 加载数据的进度,file=sys.stdout将进度条输出到控制台
    for step,data in enumerate(train_bar):
        images,labels=data
        optimizer.zero_grad()
        outputs=net(images.to(device))
        loss=loss_function(outputs,labels.to(device))
        loss.backward()
        optimizer.step()

        #print statistics
        running_loss+=loss.item()

        train_bar.desc='train epoch[{}/{}] loss{:.3f}'.format(epoch+1,epochs,loss)

    #validate
    net.eval()#使dropout在预测中不起作用
    acc=0.0
    with torch.no_grad():
        val_bar=tqdm(validate_loader,file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            #torch.max 的返回值是一个包含两个张量的元组：第一个张量包含每一行的最大值。第二个张量包含每一行最大值对应的索引。
            #outputs 通常是模型对输入数据的预测结果，它是一个二维张量，其中每一行代表一个样本的预测结果，
            #dim=1：指定在第1维（即列维度）上寻找最大值。
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            #torch.eq 函数用于比较两个张量对应位置的元素是否相等，返回一个布尔类型的张量,True是1，False是0，然后求和得到正确预测的数量。
            #.item() 将这个数值从张量中提取出来，加到累积的正确预测数量 acc 中。

    val_accurate = acc / val_num
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
            (epoch + 1, running_loss / train_steps, val_accurate))

    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)

