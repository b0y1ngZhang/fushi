import torch
from PIL import Image
from torch.utils.data import DataLoader
import common
from mydataset import my_dataset
import one_hot
from b0y1ng_model import b0y1ng_model
from torchvision import transforms

def test_predict():
    test_dataset = my_dataset("./datasets/test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    m = torch.load("model.pth")
    m.eval()
    correct = 0
    test_len = test_dataset.__len__()
    print(test_len)
    for i, (images, labels) in enumerate(test_dataloader):
        # print(labels.shape)#[4, 144]
        labels = labels.view(-1, common.captcha_array.__len__())
        # print(labels.shape)#[160=4*40, 36]
        label_text = one_hot.vec2Text(labels)
        output = m(images)
        output = output.view(-1, common.captcha_array.__len__())
        output_test = one_hot.vec2Text(output)
        # print(output.shape,output_test)
        if label_text == output_test:
            correct += 1
            print("正确值:{}, 预测值:{}".format(label_text, output_test))
        else:
            print("正确值:{}, 预测失败值:{}".format(label_text, output_test))
    print("正确率:{}%".format(correct/test_len  * 100))

def test_pic(path):#单张图片预测
    img=Image.open(path)
    trans=transforms.Compose([
        transforms.Resize((60,160)),
        transforms.Grayscale(),#"灰度一下"
        transforms.ToTensor()
    ])
    img_tensor=trans(img)
    img_tensor=img_tensor.reshape((1,1,60,160))
    print(img_tensor.shape)

    mymodel = b0y1ng_model()
    torch.save(mymodel.state_dict(), "model.pth")
    m=torch.load("model.pth",weights_only=True)#加载模型
    m.eval()

    output=m(img_tensor)
    output = output.view(-1, common.captcha_array.__len__())
    output_label = one_hot.vec2Text(output)
    return output_label

if __name__=="__main__":
   #test_predict()
    test_pic("./datasets/test/0ap7_1742846785.png")
