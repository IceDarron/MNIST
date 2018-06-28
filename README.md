MNIST 简单的手写数字0-9识别
===

#### author rongxn(IceDarron)
#### time 20170914

mnist.py为TensorFlow官网的简单例子。正确率约在91%。

distinguish.py利用mnist训练出来的模型来预测手写数字。

transferImage.py里面含有mnist需要的图片解析与图片转换为mnist所能识别的张量型式的函数。

mnist_cnn.py为cnn结构的例子，精度较高。
