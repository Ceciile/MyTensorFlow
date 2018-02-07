#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on 7th Feb 2018

@author: Qianqian
"""

# Import `tensorflow` 
import tensorflow as tf 

# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data :array of shape [None, 784] instead of the [None, 28, 28] 
#784=shape of your grayscale images
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer :operates on the unscaled output of earlier layers
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
#probability error in discrete classification tasks in which the classes are mutually exclusive

# Define an optimizer :SGD, ADAM and RMSprop
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
