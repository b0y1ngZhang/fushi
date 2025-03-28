#one_hot编码是将类别变量转换为机器学习算法易于利用的一种形式的过程
import torch
import common

def text2Vec(text):
    # 把text转化成二维张量vec
    # 4行36行
    vec=torch.zeros(common.captcha_size,len(common.captcha_array))
    # 初始化一个全零张量 行数为验证码长度 列数为字符集大小
    for i in range(len(text)):# 遍历
        vec[i,common.captcha_array.index(text[i])]=1
        # eg
        # text2Vec("aab1")
        # 假设在 common.captcha_array, 'a' 的索引是 0，'b' 是 1，'1' 是 27
        # 则返回的 vec:
        # [[1, 0, 0, ..., 0],  # 'a' -> one-hot 编码
        #  [1, 0, 0, ..., 0],  # 'a' -> one-hot 编码
        #  [0, 1, 0, ..., 0],  # 'b' -> one-hot 编码
        #  [0, 0, 0, ..., 1]]  # '1' -> one-hot 编码
    return vec

def vec2Text(vec):
    vec=torch.argmax(vec,dim=1)# 获取每行最大值的索引(即 one-hot 码中 1 所在的位置)
    text=""
    for i in vec:
        text+=common.captcha_array[i]
        # common.captcha_array[i]: 利用索引找到对应的字符 拼接成字符串返回
    return text

if __name__ == '__main__':
    vec=text2Vec("aab1")
    vec=vec.view(1,-1)[0]# 将其展平为一维张量
    print(vec,vec.shape)
    print(vec2Text(vec))