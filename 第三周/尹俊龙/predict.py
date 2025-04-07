import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from mnist_dataset import MNISTDataset
from model import AlexNet

def generate_submission():
    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # 数据预处理（与train.py一致）
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载模型（关键修改：添加 weights_only=True）
    model_path = "./AlexNet.pth"
    assert os.path.exists(model_path), f"模型文件不存在于：{os.path.abspath(model_path)}"
    
    model = AlexNet(num_classes=10)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)  # 修复警告
    )
    model.to(device)
    model.eval()

    # 加载测试数据
    test_csv_path = "./data/test.csv"
    assert os.path.exists(test_csv_path), f"测试数据不存在于：{os.path.abspath(test_csv_path)}"
    
    test_dataset = MNISTDataset(
        csv_path=test_csv_path,
        is_train=False,
        transform=data_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 执行预测
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy().tolist())

    # 生成提交文件（修正括号和缩进）
    submission = pd.DataFrame({
        "ImageId": range(1, len(predictions)+1),  # 确保括号闭合
        "Label": predictions
    })
    submission.to_csv("submission.csv", index=False)
    print(f"生成提交文件成功！记录数: {len(submission)}")

if __name__ == "__main__":
    generate_submission()
    