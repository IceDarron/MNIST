from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from PIL import Image


def weight_variable(shape, dtype, name):
    initial = tf.truncated_normal(shape=shape, stddev=0.1, dtype=dtype, name=name)
    return tf.Variable(initial, dtype=tf.float32)


def bias_variable(shape, dtype, name):
    initial = tf.constant(0.1, shape=shape, dtype=dtype, name=name)
    return tf.Variable(initial, dtype=tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 绝对路径
local_path = 'H:/Jetbrains/PyCharm/Workspace/MNIST/Mnist/'

# 模型保存路径
save_path = local_path + "save_model/cnn/cnn.ckpt"

# 加载数据集
mnist = input_data.read_data_sets(local_path + "MNIST_data", one_hot=True)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# convolution 1
weight_conv1 = weight_variable([5, 5, 1, 32], dtype="float", name='weight_conv1')
bias_conv1 = bias_variable([32], dtype="float", name='bias_conv1')
hidden_conv1 = tf.nn.relu(conv2d(x_image, weight_conv1) + bias_conv1)
hidden_pool1 = max_pool_2x2(hidden_conv1)

# convolution 2
weight_conv2 = weight_variable([5, 5, 32, 64], dtype="float", name='weight_conv2')
bias_conv2 = bias_variable([64], dtype="float", name='bias_conv2')
hidden_conv2 = tf.nn.relu(conv2d(hidden_pool1, weight_conv2) + bias_conv2)
hidden_pool2 = max_pool_2x2(hidden_conv2)

# function 1
hidden_pool2_flat = tf.reshape(hidden_pool2, [-1, 7 * 7 * 64])
weight_fc1 = weight_variable([7 * 7 * 64, 1024], dtype="float", name='weight_fc1')
bias_fc1 = bias_variable([1024], dtype="float", name='bias_fc1')
hidden_fc1 = tf.nn.relu(tf.matmul(hidden_pool2_flat, weight_fc1) + bias_fc1)
keep_prob = tf.placeholder("float")
hidden_fc1_dropout = tf.nn.dropout(hidden_fc1, keep_prob)

# function 2
weight_fc2 = weight_variable([1024, 10], dtype="float", name='weight_fc2')
bias_fc2 = bias_variable([10], dtype="float", name='weight_fc2')
y_fc2 = tf.nn.softmax(tf.matmul(hidden_fc1_dropout, weight_fc2) + bias_fc2)

# create tensorflow structure
cross_entropy = -tf.reduce_sum(y * tf.log(y_fc2))
optimize = tf.train.AdamOptimizer(1e-4)
train = optimize.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_fc2, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# initial all variables
init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)


# train
def trainmodel():
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        session.run(train, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
        if i % 100 == 0:
            print("step %4d: " % i)
            print(session.run(accuracy, feed_dict={x: batch[0], y: batch[1], keep_prob: 1}))

    print(session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1}))


# save variables
def save():
    saver = tf.train.Saver()
    saver.save(session, save_path)


# restore variables
def restore():
    saver = tf.train.Saver()
    saver.restore(session, save_path)


def gettestpicarray(filename):
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


def testmypicture():
    testpicture = "plot/8.png"
    # testpicture = "Image/7.png"
    onetestx = gettestpicarray(testpicture)
    ans = tf.argmax(y_fc2, 1)
    print("The prediction answer is:")
    print(session.run(ans, feed_dict={x: onetestx, keep_prob: 1}))


########################################################################################################################
# 对外提供的接口
def out_picturecode(img):
    x_s = 28
    y_s = 28
    out = img.resize((x_s, y_s), Image.ANTIALIAS)

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
        for x in range(x_s):
            for y in range(y_s):
                im_arr[x][y] = 255 - im_arr[x][y]
                if im_arr[x][y] < threshold:  im_arr[x][y] = 0

    out = Image.fromarray(np.uint8(im_arr))
    nm = im_arr.reshape((1, 784))

    nm = nm.astype(np.float32)
    nm = np.multiply(nm, 1.0 / 255.0)

    return nm


def out_picture2string(img):
    restore()
    picture_code = out_picturecode(img)
    ans = tf.argmax(y_fc2, 1)
    result = session.run(ans, feed_dict={x: picture_code, keep_prob: 1})
    return result


########################################################################################################################


if __name__ == '__main__':
    trainmodel()
    save()
    restore()
    testmypicture()
    session.close()
