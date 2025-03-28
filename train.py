import torch# PyTorch的核心库
from torch import nn# 包含了神经网络相关的模块
from torch.optim import Adam# 优化器 常用与深度学习训练
from torch.utils.data import DataLoader# 批量加载数据
from mydataset import my_dataset
from b0y1ng_model import b0y1ng_model
from torch.utils.tensorboard import SummaryWriter

if __name__=='__main__':
    text_dataset=my_dataset("./datasets/test")# "./datasets/test"下的测试集
    test_dataloader=DataLoader(text_dataset,batch_size=40,shuffle=True)# 加载数据 每次加载40个 被随机打乱

    train_dataset = my_dataset("./datasets/train")# 同上
    train_dataloader = DataLoader(train_dataset, batch_size=40, shuffle=True)

    w=SummaryWriter("logs")# 创建一个日志 存放在 "logs" 目录下 供TensorBoard使用
    b0y1ng_model=b0y1ng_model()# 加载自定义模型
    loss_fn = nn.MultiLabelSoftMarginLoss()# 损失函数 适用于多标签分类问题
    optim=Adam(b0y1ng_model.parameters(),lr=0.001)# 使用Adam 学习率为0.001
    w=SummaryWriter("logs")
    total_step=0

    for epoch in range(10):# 训练10个epoch 在深度学习中 "epoch" 是指对整个训练数据集进行一次完整的训练过程
        print("外层训练次数{}".format(epoch))
        for i,(images,labels) in enumerate(train_dataloader):
            # 对 train_dataloader 中的每个批次进行遍历 i 是批次索引
            # (images,labels) 是批次数据
            b0y1ng_model.train()# 将模型设置为训练模式 启用 Batch Normalization 和 Dropout 等训练专用操作
            outputs=b0y1ng_model(images)# 将批次数据 images 输入模型 得到预测输出 outputs
            loss=loss_fn(outputs,labels)# 通过损失函数计算输出与标签之间的损失
            optim.zero_grad()# 清除模型中所有参数的梯度 避免梯度累积
            loss.backward()# 反向传播 计算损失对模型参数的梯度
            optim.step()# 利用计算得到的梯度 按优化算法更新模型参数
            total_step+=1# 训练次数累加
            if i%100==0:
                print("训练次数{}，损失率{}".format(i,loss.item()))# 提取张量 loss 中的值
                w.add_scalar("loss",loss,total_step)# 将当前批次的损失值记录下来 后续可以使用 TensorBoard 查看损失变化曲线



torch.save(b0y1ng_model,"model.pth")# 保存训练好的模型到 "model.pth" 文件

        #print(images.shape)
        #w.add_images("imgs",images,global_step=i)