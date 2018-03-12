#tensorflow通过变量实现神经网络的参数并实现向前传播的过程
import tensorflow as tf
#通过seed参数设置随机种子，保证每次运行的结果一样
w1 = tf.Variable(tf.random_normal([2, 3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3, 1],stddev=1,seed=1))
x = tf.constant([[0.7,0.9]])
#向前传播算法，矩阵相乘，节点值与权重矩阵
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
#sess = tf.Session
#变量初始化
#sess.run(w2.initializer)
#sess.run(w1.initializer)
#sess.run(init_op)
print(sess.run(y))
sess.close()