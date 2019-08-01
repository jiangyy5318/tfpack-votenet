#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
sampling_module=tf.load_op_library('tf_sampling_so.so')

with tf.Graph().as_default():
    with tf.device('/gpu:0'):
        b = tf.zeros((100,200,3))
        c = sampling_module.farthest_point_sample(b, 10)

        with tf.Session() as sess:
            c1 =  sess.run(c)
            print(c1)
