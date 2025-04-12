# def pre_processing(ad,anwser):
#     csv_path = ad
#     df = pd.read_csv(csv_path)

#     # 获取某列的唯一值（例如列名为 "category"）
#     unique_elements = df["label"].unique()

#     sorted_unique_elements = sorted(unique_elements)
#     # 生成字典：键为自然数，值为元素
#     if anwser == "yes":
#         index_to_element = {idx: element for idx, element in enumerate(sorted_unique_elements)}

#         data_serializable = {k: int(v) for k, v in index_to_element.items()}
#         json_str=json.dumps(data_serializable,indent=4)#indent指定缩进的空格数
#         import json
#         with open('topics_dict.json', 'w') as json_file:
#             json.dump(data_serializable, json_file)


#     features = df.drop("label", axis=1).values  # 提取特征（NumPy 数组）
#     labels = df["label"].values                 # 提取标签（NumPy 数组）

#     # 转换为 Tensor（注意数据类型）
#     features_tensor = torch.tensor(features, dtype=torch.float32)  # 特征通常是 float
#     labels_tensor = torch.tensor(labels, dtype=torch.long)         # 标签通常是 int/long

#     from torch.utils.data import TensorDataset

#     dataset = TensorDataset(features_tensor, labels_tensor)  # 特征和标签配对
#     return dataset

def pre_processing(ad,data,labels,anwser1,anwser2,):
    
    import pandas as pd
    import torch
    import json

    csv_path = ad
    df = pd.read_csv(csv_path)
    X=df.drop("label",axis=1)
    Y=df.label
    # 获取某列的唯一值（例如列名为 "category"）
    if anwser1 == "yes":
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
    if anwser2 == "have":
        Y_F = labels.values# 提取标签（NumPy 数组）
        labels_tensor = torch.tensor(Y_F, dtype=torch.long)         # 标签通常是 int/long
                     

    # 转换为 Tensor（注意数据类型）
    features_tensor = torch.tensor(X_F, dtype=torch.float32)  # 特征通常是 float
    

    from torch.utils.data import TensorDataset

    dataset = TensorDataset(features_tensor, labels_tensor)  # 特征和标签配对
    return dataset
