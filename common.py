import random# 用于生成随机验证码字符串
import time# 用于生成带时间戳的文件名，以保证文件名唯一

captcha_array=list("0123456789abcdefghijklmnopqrstuvwxyz")# 包含0-9、a-z字符，作为验证码候选集
captcha_size=4#长度为4

from captcha.image import ImageCaptcha# 用于生成验证码图片

if __name__ == '__main__':# 确保代码是直接运行的，而不是被其他模块导入时执行
    for i in range(1400):
        image = ImageCaptcha()# 创建验证码图片生成器的实例
        image_text="".join(random.sample(captcha_array, captcha_size))
        # random.sample(captcha_array, captcha_size): 随机抽取4个字符(不重复)
        # "".join(...): 将随机字符拼接成一个字符串
        image_path="./datasets/test/{}_{}.png".format(image_text,int(time.time()))
        # 在./datasets/test中生成验证码图片的文件名, {验证码值}_{时间戳}.png
        print(image_path)# 打印文件路径-方便调试
        image.write(image_text,image_path)# 生成image_text的图片, 并保存到image_path