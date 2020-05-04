
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import kfac
import numpy as np
import tensorflow as tf

from kfac.examples import mnist


__all__ = [
    "conv_layer",
    "fc_layer",
    "max_pool_layer",
    "build_model",
    "minimize_loss_single_machine",
    "train_mnist_single_machine",
]


# Inverse update ops will be run every _INVERT_EVRY iterations.
_INVERT_EVERY = 1

# Covariance matrices will be update  _COV_UPDATE_EVERY iterations.
_COV_UPDATE_EVERY = 1

# Displays loss every _REPORT_EVERY iterations.
_REPORT_EVERY = 1

# Use manual registration
_USE_MANUAL_REG = False


def fc_layer(layer_id, inputs, output_size):
  layer = tf.layers.Dense(
      output_size,
      kernel_initializer=tf.random_normal_initializer(),
      name="fc_%d" % layer_id)
  preactivations = layer(inputs)
  activations = tf.nn.tanh(preactivations)

  # layer.weights is a list. This converts it a (hashable) tuple.
  return preactivations, activations, (layer.kernel, layer.bias)


def conv_layer(layer_id, inputs, kernel_size, out_channels):
  layer = tf.layers.Conv2D(
      out_channels,
      kernel_size=[kernel_size, kernel_size],
      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
      padding="SAME",
      name="conv_%d" % layer_id)
  preactivations = layer(inputs)
  activations = tf.nn.relu(preactivations)

  return preactivations, activations, (layer.kernel, layer.bias)


def max_pool_layer(layer_id, inputs, kernel_size, stride):
  with tf.variable_scope("pool_%d" % layer_id):
    return tf.nn.max_pool(
        inputs, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1],
        padding="SAME",
        name="pool")


def build_model(examples,
                labels,
                num_labels,
                layer_collection,
                register_layers_manually=False):
  tf.logging.info("Building model.")
  pre0, act0, params0 = conv_layer(
      layer_id=0, inputs=examples, kernel_size=5, out_channels=16)
  act1 = max_pool_layer(layer_id=1, inputs=act0, kernel_size=3, stride=2)
  pre2, act2, params2 = conv_layer(
      layer_id=2, inputs=act1, kernel_size=5, out_channels=16)
  act3 = max_pool_layer(layer_id=3, inputs=act2, kernel_size=3, stride=2)
  flat_act3 = tf.reshape(act3, shape=[-1, int(np.prod(act3.shape[1:4]))])
  logits, _, params4 = fc_layer(
      layer_id=4, inputs=flat_act3, output_size=num_labels)
  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits))
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(tf.cast(labels, dtype=tf.int32),
                       tf.argmax(logits, axis=1, output_type=tf.int32)),
              dtype=tf.float32))

  with tf.device("/cpu:0"):
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)

  layer_collection.register_softmax_cross_entropy_loss(
      logits, name="logits")

  if register_layers_manually:
    layer_collection.register_conv2d(params0, (1, 1, 1, 1), "SAME", examples,
                                     pre0)
    layer_collection.register_conv2d(params2, (1, 1, 1, 1), "SAME", act1,
                                     pre2)
    layer_collection.register_fully_connected(params4, flat_act3, logits)

  return loss, accuracy


def minimize_loss_single_machine(loss,
                                 accuracy,
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
  cov = optimizer.get_cov_vars()
  tf.logging.info("Starting training.")
  with tf.train.MonitoredTrainingSession(config=session_config) as sess:
    while not sess.should_stop():
      global_step_, loss_, accuracy_, _, cov_ = sess.run(
          [g_step, loss, accuracy, train_op, cov])

      if global_step_ % _REPORT_EVERY == 0:
        tf.logging.info("global_step: %d | loss: %f | accuracy: %s",
                        global_step_, loss_, accuracy_)

  return accuracy_, cov_

def train_mnist_single_machine(num_epochs,
                               use_fake_data=False,
                               device=None,
                               manual_op_exec=False):
    # Load a dataset.
    tf.logging.info("Loading MNIST into memory.")
    (examples, labels) = mnist.load_mnist_as_iterator(num_epochs,
                                                      128,
                                                      use_fake_data=use_fake_data,
                                                      flatten_images=False)

    # Build a ConvNet.
    layer_collection = kfac.LayerCollection()
    loss, accuracy = build_model(
        examples, labels, num_labels=10, layer_collection=layer_collection,
        register_layers_manually=_USE_MANUAL_REG)
    if not _USE_MANUAL_REG:
      layer_collection.auto_register_layers()

    # Without setting allow_soft_placement=True there will be problems when
    # the optimizer tries to place certain ops like "mod" on the GPU (which isn't
    # supported).
    config = tf.ConfigProto(allow_soft_placement=True)

    # Fit model.
    return minimize_loss_single_machine(
        loss, accuracy, layer_collection, device=device, session_config=config)

import sys
acc, cov = train_mnist_single_machine(1)
print(acc)
print(cov)
for i, tensor in enumerate(cov):
    # tf.print(tensor, output_stream=sys.stderr)
    print(i, tensor[0].shape)



kfac_optimizer = kfac.PeriodicInvCovUpdateKfacOpt(learning_rate=0.005, damping=0.001, layer_collection=layer_collection)
kfac_train_op = kfac_optimizer.minimize(loss)