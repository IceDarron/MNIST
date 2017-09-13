from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


def imageprepare():
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    file_name = 'Image/4.png'  # 导入自己的图片地址
    # in terminal 'mogrify -format png *.jpg' convert jpg to png
    im = Image.open(file_name).convert('L')

    im.save("plot/sample.png")
    plt.imshow(im)
    plt.show()
    tv = list(im.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]

    for i in range(len(tva)):
        if tva[i] == 1:
            tva[i] = 0
        else:
            tva[i] = 1
        if (i + 1) % 28 == 0:
            print(tva[i])
        else:
            print(tva[i], end=' ')

    return tva


result = imageprepare()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

init = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "save_model/m100/model.ckpt")  # 这里使用了之前保存的模型参数
    sess.run(init)
    print("Model restored.")

    prediction = tf.argmax(y, 1)
    predint = prediction.eval(feed_dict={x: [result]}, session=sess)

    y_ = tf.placeholder("float", [None, 10])
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    num = {
        0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }

    for n in num:
        accuracy = sess.run(accuracy, feed_dict={x: [result], y_: [num[n]]})
        if accuracy == 1:
            print('recognize result:')
            print(n)
            break
