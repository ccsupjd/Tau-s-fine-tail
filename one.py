# -- coding: utf-8 --
#从0开始tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
a = tf.constant([1.0,2.0],name="a")
b = tf.constant([2.0,3.0],name="b")
result = a + b
#result = tf.add(a,b,"add")
#print(result)
with tf.Session() as sess:
    sess.run(result)


