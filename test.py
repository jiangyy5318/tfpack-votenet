
import torch
import numpy as np
import tensorflow as tf
import torch

# batch_size = 5
# nb_digits = 10
# # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
# y = torch.LongTensor(batch_size,1).random_() % nb_digits
# # One hot encoding buffer that you create out of the loop and just keep reusing
# y_onehot = torch.FloatTensor(batch_size, nb_digits)
#
# # In your for loop
# y_onehot.zero_()
# y_onehot.scatter_(1, y, 1)
#
# print(y)
# print(y_onehot)

a = np.random.rand(5*8*3).reshape((5,8,3))
b = np.random.rand(5*6*3).reshape((5,6,3))
c = tf.losses.huber_loss(predictions=tf.expand_dims(a, 2) - tf.expand_dims(b, 1),
                         reduction=tf.losses.Reduction.NONE)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    c1 = sess.run(c)
    print(c1.shape)


# heading_label_one_hot = torch.FloatTensor(3, heading_class_label.shape[1], 6).zero_()
# heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1)
#
# print(heading_label_one_hot)
# import tensorflow as tf
#
# # tensor = tf.constant([[1, 2, 3], [4, 5, 6]],dtype=tf.int32)
# # b = tf.ones_like(tensor)
# # print(b.dtype)
#
# a = tf.Variable([[1,2,3],[-1,2,6]], dtype=tf.float32)
# #b = tf.zeros_like(a, dtype=tf.int32)
#
# b = tf.square(tf.norm(a, axis=1, ord=2))
# c = tf.reduce_sum(tf.square(a),axis=1)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     b1,c1 = sess.run([b,c])
#     print(b1)
#     print(c1)
