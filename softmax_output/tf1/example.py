import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
from tensorflow import layers
from utils import *
import kfac
import numpy as np

# tf.random.set_random_seed(127)
# https://github.com/tensorflow/kfac/blob/cf6265590944b5b937ff0ceaf4695a72c95a02b9/kfac/examples/convnet.py#L587
batch_size = 4
weights = load_pk('weights.pk')
data = load_pk('data.pk')[:batch_size, :]
labels = load_pk('labels.pk')[:batch_size]
input_dim, output_dim = weights.shape

x = tf.placeholder("float", [None, input_dim])
y = tf.placeholder("float", [None])

init = tf.constant_initializer(weights)
layer = tf.layers.Dense(output_dim, kernel_initializer=init, use_bias=False)
logits = layer(x)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

layer_collection = kfac.LayerCollection()
layer_collection.register_softmax_cross_entropy_loss(logits, name="logits")
layer_collection.auto_register_layers()

optimizer = kfac.PeriodicInvCovUpdateKfacOpt(
                                             invert_every=1,
                                             cov_update_every=1,
                                             learning_rate=0.0001,
                                             cov_ema_decay=0.95,
                                             damping=0.001,
                                             layer_collection=layer_collection)

train_op = optimizer.minimize(loss)

with tf.control_dependencies([train_op]):  # make sure this happens after the train block
    cov_vars = optimizer.get_cov_vars()

config = tf.ConfigProto(allow_soft_placement=True)
with tf.train.MonitoredTrainingSession(config=config) as sess:  # have to use this because of something to do with the device placement?
    loss_, logits_, covs_, _ = sess.run([loss, logits, cov_vars, train_op], feed_dict={x:data, y:labels})

print(loss_)
print(covs_[0])
print(covs_[1])
