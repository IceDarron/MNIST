import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import numpy as np
from PIL import Image

# save_path = "H:\\Jetbrains\\PyCharm\\Workspace\\MNIST\\Mnist\\save_model\\m100\\model.ckpt"
# data_download_path = 'H:\\Jetbrains\\PyCharm\\Workspace\\MNIST\\Mnist\\MNIST_data'
save_path = "/Users/tree/PycharmProjects/MNIST/Mnist/save_model/m100/model.ckpt"
data_download_path = '/Users/tree/PycharmProjects/MNIST/Mnist/MNIST_data'

# 用于下载训练数据
mnist = input_data.read_data_sets(data_download_path, one_hot=True)
print("download dataSet finished!!!")

# x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。
# 我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。
# 我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。
# （这里的None表示此张量的第一个维度可以是任何长度的。）
x = tf.placeholder(tf.float32, [None, 784], name="input_x")

# 权重值
W = tf.Variable(tf.zeros([784, 10]), name="W")
# 偏置量
b = tf.Variable(tf.zeros([10]), name="b")

# 模型
y = tf.nn.softmax(tf.matmul(x, W) + b, name="input_y")

# 新的占位符用于输入正确值
y_ = tf.placeholder("float", [None, 10])

#  计算交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化我们创建的变量
init = tf.global_variables_initializer()

# 开启会话
sess = tf.Session()

# 启动模型
sess.run(init)


# 训练
def train_model():
    for i in range(100):
        # 100个批处理数据点
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, c = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
        print(c)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print(i, ":", accuracy)


# save variables
def save():
    saver = tf.train.Saver()
    saver.save(sess, save_path)


# restore variables
def restore():
    saver = tf.train.Saver()
    saver.restore(sess, save_path)


def image2array(filename):
    im = Image.open(filename)
    x_s = 28
    y_s = 28
    out = im.resize((x_s, y_s), Image.ANTIALIAS)
    im_arr = np.array(out.convert('L'))
    num0 = 0
    num255 = 0
    threshold = 100

    for x in range(x_s):
        for y in range(y_s):
            if im_arr[x][y] > threshold:
                num255 = num255 + 1
            else:
                num0 = num0 + 1

    if num255 > num0:
        print("convert!")
        for x in range(x_s):
            for y in range(y_s):
                im_arr[x][y] = 255 - im_arr[x][y]
                if im_arr[x][y] < threshold:  im_arr[x][y] = 0
                # if(im_arr[x][y] > threshold) : im_arr[x][y] = 0
                # else : im_arr[x][y] = 255
                # if(im_arr[x][y] < threshold): im_arr[x][y] = im_arr[x][y] - im_arr[x][y] / 2

    out = Image.fromarray(np.uint8(im_arr))
    out.save('plot/' + filename.split('/')[1])
    # print im_arr
    nm = im_arr.reshape((1, 784))

    nm = nm.astype(np.float32)
    nm = np.multiply(nm, 1.0 / 255.0)

    return nm


def image2num(filename):
    test_array = image2array(filename)
    ans = tf.argmax(y, 1)
    print("The prediction answer is:")
    print(sess.run(ans, feed_dict={x: test_array}))


########################################################################################################################
def out_image2array(im):
    x_s = 28
    y_s = 28
    out = im.resize((x_s, y_s), Image.ANTIALIAS)
    im_arr = np.array(out.convert('L'))
    num0 = 0
    num255 = 0
    threshold = 100

    for x in range(x_s):
        for y in range(y_s):
            if im_arr[x][y] > threshold:
                num255 = num255 + 1
            else:
                num0 = num0 + 1

    if num255 > num0:
        print("convert!")
        for x in range(x_s):
            for y in range(y_s):
                im_arr[x][y] = 255 - im_arr[x][y]
                if im_arr[x][y] < threshold:  im_arr[x][y] = 0
                # if(im_arr[x][y] > threshold) : im_arr[x][y] = 0
                # else : im_arr[x][y] = 255
                # if(im_arr[x][y] < threshold): im_arr[x][y] = im_arr[x][y] - im_arr[x][y] / 2

    # out = Image.fromarray(np.uint8(im_arr))
    nm = im_arr.reshape((1, 784))

    nm = nm.astype(np.float32)
    nm = np.multiply(nm, 1.0 / 255.0)

    return nm


def out_image2num(image):
    test_array = out_image2array(image)
    ans = tf.argmax(y, 1)
    print("The prediction answer is:")
    print(sess.run(ans, feed_dict={x: test_array}))


########################################################################################################################
# def image2num_mnist(image):
#     images = split_image(image)
#     result = ''
#     for img in images:
#         tmp = ''.join(str(s) for s in mnist.out_image2num(img) if s not in [None])
#         result = result + tmp
#     return result
#
#
# def open_mnist():
#     mnist.restore()
#
#
# def close_mnist():
#     mnist.session.close()


if __name__ == '__main__':
    # train_model()
    # save()
    restore()
    image2num("plot/0.png")
    sess.close()
