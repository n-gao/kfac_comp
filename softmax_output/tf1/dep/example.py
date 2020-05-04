import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
from tensorflow import layers
from utils import *
import kfac
import numpy as np

tf.random.set_random_seed(127)
# https://github.com/tensorflow/kfac/blob/cf6265590944b5b937ff0ceaf4695a72c95a02b9/kfac/examples/convnet.py#L587

weights = load_pk('../weights.pk')
data = load_pk('../data.pk')
labels = load_pk('../labels.pk')
batch_size = 64
_USE_MANUAL_REG = False
_INVERT_EVERY = 1
_COV_UPDATE_EVERY = 1
_REPORT_EVERY = 1

def fc_layer(layer_id, inputs, output_size):
  layer = tf.layers.Dense(
      output_size,
      weights=[weights],
      name="fc_%d" % layer_id, use_bias=False)
  preactivations = layer(inputs)
  activations = tf.nn.tanh(preactivations)
  return preactivations, activations, (layer.kernel, layer.bias)


def load_numpy_as_iterator(tensor, num_epochs, batch_size, dtype=tf.float32):
    x = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(tensor)).repeat(num_epochs).batch(batch_size)
    x = tf.compat.v1.data.make_one_shot_iterator(x).get_next()
    return x


def train_mnist_single_machine(data, labels, num_epochs,
                               use_fake_data=False,
                               device=None,
                               manual_op_exec=False):
  # Load a dataset.
  tf.logging.info("Loading MNIST into memory.")

  examples = load_numpy_as_iterator(data, num_epochs)
  labels = load_numpy_as_iterator(labels, num_epochs)

  # Build a ConvNet.
  layer_collection = kfac.LayerCollection()
  loss, accuracy, examples = build_model(
      examples, labels, num_labels=10, layer_collection=layer_collection)
  if not _USE_MANUAL_REG:
    layer_collection.auto_register_layers()

  # Without setting allow_soft_placement=True there will be problems when
  # the optimizer tries to place certain ops like "mod" on the GPU (which isn't
  # supported).
  config = tf.ConfigProto(allow_soft_placement=True)

  # Fit model.
  # if manual_op_exec:
  #   return minimize_loss_single_machine_manual(
  #       loss, accuracy, layer_collection, device=device, session_config=config)
  # else:
  return minimize_loss_single_machine(
        loss, accuracy, examples, layer_collection, device=device, session_config=config)

def build_model(examples,
                labels,
                num_labels,
                layer_collection):

  # pre, act, params
  logits, _, params0 = fc_layer(layer_id=0, inputs=examples, output_size=5)

  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits))
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(tf.cast(labels, dtype=tf.int32),
                       tf.argmax(logits, axis=1, output_type=tf.int32)),
              dtype=tf.float32))

  layer_collection.register_softmax_cross_entropy_loss(logits, name="logits")

  return loss, accuracy, examples
# import kfac

# kfac.PeriodicInvCovUpdateKfacOpt()

def minimize_loss_single_machine(loss,
                                 accuracy,
                                 logits,
                                 layer_collection,
                                 device=None,
                                 session_config=None):
  device_list = [] if not device else [device]

  # Train with K-FAC.
  g_step = tf.train.get_or_create_global_step()
  optimizer = kfac.PeriodicInvCovUpdateKfacOpt(
      invert_every=_INVERT_EVERY,
      cov_update_every=_COV_UPDATE_EVERY,
      learning_rate=0.0001,
      cov_ema_decay=0.95,
      damping=0.001,
      layer_collection=layer_collection,
      placement_strategy="round_robin",
      cov_devices=device_list,
      inv_devices=device_list,
      trans_devices=device_list,
      momentum=0.9)

  with tf.device(device):
    train_op = optimizer.minimize(loss, global_step=g_step)

  cov_vars = optimizer.get_cov_vars()

  with tf.train.MonitoredTrainingSession(config=session_config) as sess:
    while not sess.should_stop():
      global_step_, loss_, accuracy_, _, cov_vars_, logits_ = sess.run(
          [g_step, loss, accuracy, train_op, cov_vars, logits])

  return accuracy_, loss_, cov_vars_, logits_

# data
batch_size = 8
input_data = load_numpy_as_iterator(np.load('../data/input.npy'), 1, batch_size)
weights = np.load('../data/weights.npy')
labels = load_numpy_as_iterator(np.load('../data/labels.npy'), 1, batch_size, dtype=tf.int32)
labels = tf.reshape(labels, (-1,))
output_size = weights.shape[1]
# model
layer = tf.layers.Dense(output_size, weights=[weights], name="fc", use_bias=False)
logits = layer(input_data)
activations = tf.nn.tanh(logits)
# activations = tf.nn.tanh(preactivations)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels, dtype=tf.int32),tf.argmax(logits, axis=1, output_type=tf.int32)),
          dtype=tf.float32))

layer_collection = kfac.LayerCollection()
layer_collection.register_softmax_cross_entropy_loss(logits, name="logits")
layer_collection.auto_register_layers()

# accuracy, loss, cov_vars, logits = train_mnist_single_machine(data, labels, 1)  # 1 gives nans?
#
# print('accuracy: ', accuracy)
# print('loss: ', loss)
# print(cov_vars)
# this is printing the !!last!! 5 items of the dataset because the iterator had iterated already. I am not sure how to
# get the exact inputs, but by inference we can assume the first 64 were used in training
# for ex1, ex2 in zip(logits, data[batch_size:]):
#     print(ex1)
#     print(ex2)
#     print('\n')

#
# model = Sequential()
# n_cols = data.shape[1]
# model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
# model.add(Activation('softmax'))
#
# with tf.GradientTape() as tape:
#   logits = model(images)
#   loss_value = loss(logits, labels)
# grads = tape.gradient(loss_value, model.trainable_variables)
# optimizer.apply_gradients(zip(grads, model.trainable_variables))