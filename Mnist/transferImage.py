#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import numpy as np
from PIL import Image
import struct


# 解析mnist模型中的picture
# 解析图片文件
def read_image(filename):
    f = open(filename, 'rb')

    index = 0
    buf = f.read()

    f.close()

    magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    for i in range(images):
        # for i in xrange(2000):
        image = Image.new('L', (columns, rows))

        for x in range(rows):
            for y in range(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')

        print('save ' + str(i) + 'image')
        image.save('Image/' + str(i) + '.png')


# 解析mnist模型中的picture
# 解析图片文件对应标签
def read_label(filename, savefilename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()

    f.close()

    magic, labels = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')

    labelarray = [0] * labels
    # labelArr = [0] * 2000

    for x in range(labels):
        labelarray[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')

    save = open(savefilename, 'w')

    save.write(','.join(map(lambda x: str(x), labelarray)))
    save.write('\n')

    save.close()
    print('save labels success')


# 读取图片，转灰度，resize到28
# 传入mnist模型中predict
def image2array():
    # 读取图片转成灰度格式
    img = Image.open('ImageExtra/4.png').convert('L')

    # resize的过程
    if img.size[0] != 28 or img.size[1] != 28:
        img = img.resize((28, 28))

    # 暂存像素值的一维数组
    arr = []

    for i in range(28):
        for j in range(28):
            # mnist 里的颜色是0代表白色（背景），1.0代表黑色
            pixel = 1.0 - float(img.getpixel((j, i))) / 255.0
            # pixel = 255.0 - float(img.getpixel((j, i))) # 如果是0-255的颜色值
            arr.append(pixel)

    arr1 = np.array(arr).reshape((1, 28, 28, 1))

    print(arr1)

    return arr1


if __name__ == '__main__':
    image2array()
    # read_image('MNIST_data/t10k-images.idx3-ubyte')
    # read_label('MNIST_data/t10k-labels.idx1-ubyte', 'Image/label.txt')
