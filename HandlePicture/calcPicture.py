from __future__ import division, print_function, absolute_import
from PIL import Image
import Mnist.mnist_cnn as mnist_cnn


# 读取图片，转成灰度格式，并将图片切分返回
def split_image(image):
    # 读取图片转成灰度格式
    image_l = Image.open(image).convert('L')

    # 显示图片
    # image_L.show()

    # 切分图片 box = (left, up, right, down)
    region = []
    for num in range(0, 4):
        box = (0 + num * 12, 0, 12 + num * 12, 15)
        img_1215 = image_l.crop(box)
        img_2828 = img_1215.resize((28, 28))
        region.append(img_2828)

    return region


def verification_code2string(image):
    images = split_image(image)
    result = ''
    for img in images:
        # mnist_cnn.out_picture2string(img)
        tmp = ''.join(str(s) for s in mnist_cnn.out_picture2string(img) if s not in [None])
        result = result + tmp

    mnist_cnn.session.close()
    return result


if __name__ == '__main__':
    # print(split_image('imageRandeCode.jpg'))
    print('识别结果为：' + verification_code2string('imageRandeCode (4).jpg'))
