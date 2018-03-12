#ExponentialMovingAverage滑动平均模型的使用
import tensorflow as tf
v1 = tf.Variable(0,dtype=tf.float32)
#模拟神经网络的迭代轮数，可用于动态控制衰减率
step = tf.Variable(0, trainable=False)
ema = tf.train.ExponentialMovingAverage(0.99,step)
#定义一个更新变量滑动票平均的操作，这里需要给定一个列表每次执行时列表中的变量都会宝贝更新
maintain_averages_op = ema.apply([v1])
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #通过ema.average(v1)获得滑动平均之后变量的取值。在初始化之后变量v1的值和v1的滑动平均都为0
    print(sess.run([v1, ema.average(v1)]))
    #更新变量V1的值到5
    sess.run(tf.assign(v1, 5))
    #更新V1的滑动平均值。衰减度为min{0.99，（1+step）/(10+step)≈0.1 }=0.1
    #所以V1的滑动平均会被更新为 0.1* 0+ 0.9 * 5 = 4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1 , ema.average(v1)]))
    #更新step的值为10000
    sess.run(tf.assign(step,10000))
    #更新变量V1的值到10
    sess.run(tf.assign(v1, 10))
    # 更新V1的滑动平均值。衰减度为min{0.99，（1+step）/(10+step)≈0.999 }=0.99
    # 所以V1的滑动平均会被更新为 0.99* 4.5+ 0.01 * 10 = 4.555
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
    #再次更新滑动平均值，得到更新滑动平均值为 0.99 *4.555 + 0.01*10 = 4.60945
    sess.run(maintain_averages_op)
    print(sess.run([v1 , ema.average(v1)]))
