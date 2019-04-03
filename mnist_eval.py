import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference  # 导入两个py文件
import mnist_train

EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default()as g:
        x = tf.placeholder(  # 生成两个占位符
            tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(
            tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        y = mnist_inference.inference(x, None)  # 前向传播

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # tf.argmax(vector, 1)：返回的是vector中的最大值的索引号

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # cast( x,dtype)将x的数据格式转化成dtype
        # tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
        variable_aeverages = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY  # 指数滑动平均
        )
        variable_to_restore = variable_aeverages.variables_to_restore()
        '''
        通过使用variables_to_restore函数，可以使在加载模型的时候将影子变量直接映射到变量的本身，
        所以我们在获取变量的滑动平均值的时候只需要获取到变量的本身值而不需要去获取影子变量。
        '''
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(  # tf.train.get_checkpoint_state函数通过checkpoint文件找到模型文件名。
                    # 如果路径有效则返回一个CheckpointState proto文件，否则返回None
                    mnist_train.MODEL_SAVE_PATH
                )
                if ckpt and ckpt.model_checkpoint_path:

                    saver.restore(sess, ckpt.model_checkpoint_path)
                    '''
                    restore则是将训练好的参数提取出来。
                    sess: 保存参数的会话。
                    save_path: 保存参数的路径。
                    当从文件中恢复变量时，不需要事先对他们进行初始化，因为“恢复”自身就是一种初始化变量的方法。
                    '''
                    global_step = ckpt.model_checkpoint_path \
                        .split('/')[-1].split('-')[-1]
                    '''
                    split函数：拆分字符串。通过指定分隔符对字符串进行切片，并返回分割后的字符串列表（list）。
                    split函数返回值为：分割后的字符串列表。
                    -1是取最后一个的意思
                    '''
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s),validation""accuracy= %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("/path/to/mnist_data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
