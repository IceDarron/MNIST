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


def image2num_mnist_cnn(image):
    images = split_image(image)
    result = ''
    for img in images:
        tmp = ''.join(str(s) for s in mnist_cnn.out_image2num(img) if s not in [None])
        result = result + tmp
    return result


def open_mnist_cnn():
    mnist_cnn.restore()


def close_mnist_cnn():
    mnist_cnn.session.close()


########################################################################################################################


if __name__ == '__main__':
    open_mnist_cnn()
    print('识别结果为：' + image2num_mnist_cnn('imageRandeCode.jpg'))
    close_mnist_cnn()
    exit(0)

