#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on 7th Feb 2018

@author: Qianqian
"""

#import into workspace under the alias tf
import tensorflow as tf

#initialize two variables that are actually constants
#tensors are all about arrays!
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)
#because ofabstract tensor
# Print the result
#print(result) Tensor("Mul:0", shape=(4,), dtype=int32)

#run this code in an interactive session
#在被称之为 会话 (Session) 的上下文 (context) 中执行图
'''
# Intialize the Session
sess = tf.Session()

# Print the result
print(sess.run(result))

# Close the session
sess.close()
'''
# Initialize Session and run `result` "with" 代码块 来自动完成关闭
with tf.Session() as sess:
  with tf.device('/cpu:0'):
    output = sess.run(result)
    print(output)
