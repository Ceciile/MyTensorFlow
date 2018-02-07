#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on 7th Feb 2018

@author: Qianqian
"""
import os
import skimage
import numpy as np
from skimage import data
# Import `tensorflow` 
import tensorflow as tf 

# Import the `transform` module from `skimage`
from skimage import transform 
# Import `rgb2gray` from `skimage.color`
from skimage.color import rgb2gray

#BelgiumTS for Classification (cropped images), which are called "BelgiumTSC_Training" and "BelgiumTSC_Testing".
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "./"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_directory)

# Rescale the images in the `images` array
images28 = [transform.resize(image, (28, 28)) for image in images]

# Convert `images28` to an array
images28 = np.array(images28)
# Convert `images28` to grayscale
images28 = rgb2gray(images28)

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


print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

#initialized session to start epochs or training loops

tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})###run_metadata_ptr  ValueError: Cannot feed value of shape (4575, 28, 3) for Tensor u'Placeholder:0', which has shape '(?, 28, 28)'
        if i % 10 == 0:
            print("Loss: ", loss)
'''
tf.set_random_seed(1234)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(201):
        _, loss_value = sess.run([train_op, loss], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
#2018-02-07 16:31:25.574487: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
'''
        print('DONE WITH EPOCH')
