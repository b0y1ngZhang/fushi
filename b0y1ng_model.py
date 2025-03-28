import common
import torch# PyTorch框架
from torch import nn #Neural Network - 包含常见的层和损失函数

class b0y1ng_model(nn.Module):# 继承PyTorch的nn.Module 定义深度学习模型
    def __init__(self):
        super(b0y1ng_model,self).__init__()# 调用nn.Module的初始化办法
        # 卷积层
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            # in_channels=1(灰度验证码图片)
            # 输出64个特征图, 3x3卷积核, 保持原尺寸(padding=1)
            nn.ReLU(),# 激活函数
            nn.MaxPool2d(2)# 2x2最大池化, 特征图尺寸减半
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )# [1, 256, 7, 20]
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )# [1, 512, 3, 10]: 512通道, 特征图尺寸缩小至[3, 10]

        self.layer5=nn.Sequential(# 全连接层
            nn.Flatten(),# 展平[1, 512, 3, 10]到[1, 15360]
            nn.Linear(in_features=15360, out_features=4096),# 全连接层 降维到4096维
            nn.Dropout(0.2),# 防止过拟合 20%概率随机失活
            nn.ReLU(),# 激活函数
            nn.Linear(in_features=4096, out_features=common.captcha_size*common.captcha_array.__len__()),
        )#512x3x10

    # 前向传播
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # 依次经过layer1-layer5 输出形状[batch_size, 144]
        return x
        # 适用于多标签分类任务(每个字符独立预测)

# 测试模型
if __name__ == '__main__':
    data=torch.ones(1,1,60,160)# 生成一个形状为[1, 1, 60, 160]的全1张量，模拟灰度验证码图片输入
    m=b0y1ng_model()# 创建模型实例
    x=m(data)# 进行前向传播 得到预测输出
    print(x.shape)# 输出x在每个维度上的大小-[batch size, 特征维度]