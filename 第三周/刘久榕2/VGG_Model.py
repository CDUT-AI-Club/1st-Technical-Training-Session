#VGG网络亮点，通过堆叠多个3#3卷积层和2#2最大池化层，提取特征，然后通过全连接层进行分类。
import torch
import torch.nn as nn


# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


#conv的strive默认为1，padding为1
#pool的stride默认为kernel_size=2，stride=2，padding默认为0
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):#初始化函数，features是提取特征层的网络，num_classes是分类数，init_weights是否要进行初始化
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(#3层全连接层，分类网络结构
            nn.Linear(512*7*7, 2048),#提取特征后得到的特征维度为512*7*7，这里将其展平为4096
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):#是否要进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)#初始化权重
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)#初始化偏置
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):#提取配置变量的特征层的网络
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)#对其进行非关键字参数传入进去，返回一个Sequential对象
    #非关键参数：按照参数定义的顺序直接传递的参数值，而不需要使用参数名=值的形式明确指定参数名。

#定义配置文件，以字典形式存储
cfgs = {#VGG n层网络的配置文件，数字是卷积核数量，M表示最大池化层
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", **kwargs):#实例化模型，VGG有多种变体（vgg11/vgg13/vgg16/vgg19），通过实例化函数可以动态选择不同配置
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)#**kwargs可变长度字典，在里面输入的num_classes, init_weights等参数会被自动传递给VGG的初始化函数
    return model

vgg_model=vgg(model_name='vgg13')