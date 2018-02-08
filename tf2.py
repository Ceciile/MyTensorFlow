#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on 8th Feb 2018

@author: Qianqian
"""
import tensorflow as tf

# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")

# 创建一个 op, 其作用是使 state 增加 1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.initialize_all_variables()

# 启动图, 运行 op
with tf.Session() as sess:
  # 运行 'init' op
  sess.run(init_op)
  # 打印 'state' 的初始值
  print sess.run(state)
  # 运行 op, 更新 'state', 并打印 'state'
  for _ in range(3):
    sess.run(update)
    print sess.run(state)

#Fetch为了取回操作的输出内容  多个 tensor

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)
#TensorFlow 发布的新版本的 API 修改了:其余的 tf.sub 和 tf.neg 也要相应修改为 tf.subtract 和 tf.negative
with tf.Session() as sess:
  result = sess.run([mul, intermed])
  print result
