

import torch
import numpy as np

a = np.random.rand(24).reshape(2,3,4)
b = np.random.randint(0,4,size=6).reshape((2,3))
c = torch.gather(torch.tensor(a), 2, torch.tensor(b).unsqueeze(-1))
print(c)

import tensorflow as tf
c1 = tf.gather(a, tf.expand_dims(b, -1), axis=2)
with tf.Session() as sess:
    c2 = sess.run(c1)
    print(c2)

# b = torch.argmax(torch.tensor(a), -1)
# b1 = np.argmax(a, -1)
# print(b)
# print(b1)