import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf

# 用于下载训练数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("download dataSet finished!!!")

# x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。
# 我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。
# 我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。
# （这里的None表示此张量的第一个维度可以是任何长度的。）
x = tf.placeholder(tf.float32, [None, 784])

# 权重值
W = tf.Variable(tf.zeros([784, 10]))
# 偏置量
b = tf.Variable(tf.zeros([10]))

# 模型
y = tf.nn.softmax(tf.matmul(x, W) + b)

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

# 开启模型保存
saver = tf.train.Saver()

# 启动模型
sess.run(init)

# 保存模型
saver.save(sess, "save_model/m100/model.ckpt")

# 训练
for i in range(100):
    # 100个批处理数据点
    batch_xs, batch_ys = mnist.train.next_batch(100)
    _, c = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    print(c)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print(i, ":", accuracy)
