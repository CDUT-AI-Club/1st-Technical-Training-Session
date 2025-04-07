import os
import json
import sys
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet
from mnist_dataset import MNISTDataset  # 自定义数据集加载类

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    # 数据预处理（适配AlexNet输入）
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize(224),          # 调整到AlexNet输入尺寸
            transforms.Grayscale(num_output_channels=3),  # 单通道转3通道
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3通道归一化
        ]),
        "val": transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # # 加载数据集（替换原ImageFolder部分）
    # train_dataset = MNISTDataset(
    #     csv_path="./data/train.csv",
    #     is_train=True,
    #     transform=data_transform["train"]
    # )
    # validate_dataset = MNISTDataset(
    #     csv_path="./data/test.csv",  # 注意：实际应用中建议从train.csv拆分验证集
    #     is_train=False,
    #     transform=data_transform["val"]
    # )
    # 新数据集加载代码（添加在原有位置）
    from torch.utils.data import random_split  # 需要新增导入

    # 加载完整训练集（带标签）
    full_dataset = MNISTDataset(
        csv_path="./data/train.csv",
        is_train=True,
        transform=data_transform["train"]  # 初始用train transform
    )

    # 按8:2拆分训练/验证
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, validate_dataset = random_split(full_dataset, [train_size, val_size])

    # 验证集改用val transform（保持预处理一致）
    validate_dataset.dataset.transform = data_transform["val"]

    # 测试集单独加载（不参与训练/验证）
    test_dataset = MNISTDataset(
        csv_path="./data/test.csv",
        is_train=False,
        transform=data_transform["val"]
    )

    # 数据加载参数
    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 线程数
    print('Using {} dataloader workers'.format(nw))

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=nw
    # )
    # validate_loader = torch.utils.data.DataLoader(
    #     validate_dataset,
    #     batch_size=4,
    #     shuffle=False,
    #     num_workers=nw
    # )
    # 修改为（将num_workers设为0）
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # 主要修改点
    )
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0  # 主要修改点
    )

    train_num = len(train_dataset)
    val_num = len(validate_dataset)
    print("Using {} images for training, {} images for validation".format(train_num, val_num))

    # 初始化模型（输出类别改为10）
    net = AlexNet(num_classes=10, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # 训练参数
    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)

    # 训练循环
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(
                epoch + 1, epochs, loss)

        # 验证
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # 保存最佳模型
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

if __name__ == '__main__':
    main()


