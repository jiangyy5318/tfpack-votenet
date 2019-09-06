
# import tensorflow as tf
# a = [[1,2,3],[4,5,6],[7,8,9]]
# b = [[2,1],[0,1],[1,1]]
# tf_a = tf.Variable(a)
# tf_b = tf.Variable(b, dtype=tf.int32)
# tf_c = tf.batch_gather(tf_a, indices=tf_b, axis=1)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     c = sess.run(tf_c)
#     print(c)

import tensorflow as tf
tensor_a = tf.Variable([[[1,0],[2,0],[3,0]],[[4,0],[5,0],[6,0]],[[7,0],[8,0],[9,0]]])
tensor_b = tf.Variable([[0,1],[1,1],[2,1]],dtype=tf.int32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.batch_gather(tensor_a,tensor_b)))
    #print(sess.run(tf.tile(tensor_a, [1,3])))
    #print(sess.run(tf.batch_gather(tensor_a,tensor_c)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
a = torch.Tensor(np.array([[[1,0],[2,0],[3,0]],[[4,0],[5,0],[6,0]],[[7,0],[8,0],[9,0]]]))
print(a.shape)
seed_inds = torch.Tensor(np.array(([[0,1],[1,1],[2,1]]))).long()
batch_size = 3
num_seed = 2

seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,2).long()
#print(seed_inds_expand)

seed_gt_votes = torch.gather(a, 1, seed_inds_expand)
print(seed_gt_votes)
#seed_gt_votes += end_points['seed_xyz'].repeat(1,1,3)

