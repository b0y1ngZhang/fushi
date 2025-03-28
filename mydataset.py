#加载数据集
import os# 用于遍历文件夹
from PIL import Image# 用于打开图片文件
from torch.utils.data import dataset# 创建自定义PyTorch数据集
from torch.utils.tensorboard import SummaryWriter# 用于TensorBoard可视化
from torchvision import transforms# 用于对图像的转换
import one_hot# 自定义模块

class my_dataset(dataset.Dataset):# 继承torch.utils.data中的dataset.Dataset
    def __init__(self,root_dir):# 初始化 root_dir: 根目录
        super(my_dataset,self).__init__()
        self.image_path=[os.path.join(root_dir,image_name) for image_name in os.listdir(root_dir)]
        #os.listdir(root_dir): 用于获取根目录下的所有文件名
        #os.path.join(root_dir,image_name): 将根目录和文件名拼接 得到完整的文件路径
        self.transform=transforms.Compose(# 定义self.transform进行图片预处理
            [
                transforms.ToTensor(),# 将PIL.Image转化为PyTorch张量 并归一化到[0, 1]
                transforms.Resize((60,160)),# 将图片重新缩放到60x160
                transforms.Grayscale(),# 转换为单通道灰度图(即深度为1)
            ]
        )
        #print(self.image_path)# 仅用于调试 输出所有图片路径
    def __len__(self):
        return self.image_path.__len__()# 返回图片数量(即self.image_path长度)
    def __getitem__(self, index):# 获取单个样本
        image_path=self.image_path[index]# 获取第index张图片路径
        image=self.transform(Image.open(image_path))
        # 使用PIL的Image.open打开image_path打开图片
        # 用self.transform进行预处理(转张量 调大小 灰度化)
        label=image_path.split("/")[-1]# 获取文件名
        label=label.split("_")[0]# 截取文件名的前缀作为标签
        label_tensor=one_hot.text2Vec(label)
        # 调用one_hot模块的text2Vec() 把label转化为one_hot encoding 形状为[4, 36]
        label_tensor=label_tensor.view(1,-1)[0]
        #将[4, 36]拉平成[1, 144]
        #取出第 0 维数据 最终label_tensor形状为[144]
        return image, label_tensor# 返回图片张量image和标签张量label_tensor

if __name__=='__main__':
    writer = SummaryWriter("logs")# 初始化TensorBoard记录器 日志保存在"logs"目录
    train_data=my_dataset("./datasets/train/")# 读取"./datasets/train/"目录下的图片
    img,label=train_data[0]# 取出第一个样本的图片和标签
    print(img.shape,label.shape)
    # 输出图片和标签的张量形状
    # img.shape: (1, 60, 160)(单通道 60x160 图片),
    # label.shape: [144](4 个字符，每个字符 36 维)
    writer.add_image("img",img,1)# 将img记录到TensorBoard用于可视化
    writer.close()# 关闭 SummaryWriter