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
print(result) #Tensor("Mul:0", shape=(4,), dtype=int32)


