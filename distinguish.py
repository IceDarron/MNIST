from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def imageprepare(file_name):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
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


result = [imageprepare("Image/4.png")]
# result = gettestpicarray("Image/4.png")
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])
init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "save_model/m100/model.ckpt")  # 这里使用了之前保存的模型参数
    print("Model restored.")

    ans = tf.argmax(y, 1)
    print("The prediction answer is:")
    print(session.run(ans, feed_dict={x: result}))

