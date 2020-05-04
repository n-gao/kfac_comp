import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
from utils import load_pk
from tf2KFAC import tf2KFAC

kfac = tf2KFAC(tinv=10,
               tcov=1,
               lr0=0.0001,
               cov_weight=1.,
               cov_moving_weight=0.95,
               damping=0.001,
               conv_approx='mg',
               damping_method='factored_tikhonov',
               ft_method='original',
               norm_constraint=0.95,
               inp_dim=10,
               out_dim=5)

batch_size = 4

weights = load_pk('weights.pk')
data = load_pk('data.pk')
labels = load_pk('labels.pk')

data = tf.reshape(tf.convert_to_tensor(data[:batch_size, :], dtype=tf.float32), (batch_size, -1))
labels = tf.reshape(tf.convert_to_tensor(labels[:batch_size], dtype=tf.int32), (-1,))
weights = tf.convert_to_tensor(weights, dtype=tf.float32)

# build the model
weights = tf.Variable(weights)
with tf.GradientTape(True) as g:
    logits = data @ weights

    dist = tf.nn.softmax(logits, axis=-1)  # this is the model distribution

    nl_dist = -tf.math.log(dist)

    no_mean_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

# sensitivities = g.gradient(dist, logits)  # * batch_size
sensitivities = g.gradient(nl_dist, logits)  # * batch_size
# sensitivities = g.gradient(no_mean_loss, logits)  # * batch_size
# sensitivities = g.gradient(loss, logits) # * batch_size
print(sensitivities)
print(sensitivities.shape)

grads = g.gradient(loss, weights)

sensitivities = [sensitivities]
activations = [data]
grads = [grads]

ng = kfac.compute_updates(activations, sensitivities, grads, 0)

print(kfac.m_aa[0])
print(kfac.m_ss[0])


