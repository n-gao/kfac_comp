import numpy as np
import tensorflow as tf
from collections import OrderedDict


# import kfac
#
# kfac_optimizer = kfac.PeriodicInvCovUpdateKfacOpt(learning_rate=0.005, damping=0.001, layer_collection=layer_collection)
# kfac_train_op = kfac_optimizer.minimize(prediction)
# Notes
# kwargs["cov_ema_decay"] = kwargs["cov_ema_decay"]**cov_update_every makes sense


class tf2KFAC():
    def __init__(self,
                 tinv=10,
                 tcov=1,
                 lr0=0.0001,
                 cov_weight=1.,
                 cov_moving_weight=0.95,
                 damping=0.001,
                 conv_approx='mg',
                 damping_method='factored_tikhonov',
                 ft_method='original',
                 norm_constraint=0.95,
                 inp_dim=2,
                 out_dim=1):

        self.cov_weight = cov_weight
        # cov weight 1. appears to be the default setting in tf1 code
        # set to 0.05 to match david pfaus paper, though I believe they didnt realise
        # it changes which changes the relative value of cov moving weight

        self.tinv = tinv
        self.tcov = tcov
        self.cov_moving_weight = cov_moving_weight  # the is set to 0.95 as an input
        self.damping = damping
        self.lr0 = lr0
        self.lr = lr0
        self.decay = 1e-4
        self.norm_constraint = norm_constraint

        # dictionaries holding the moving averages
        self.m_aa = {}
        self.m_ss = {}
        self.n_spatial_locations = {}  # n_conv, n_spatial_locations etc
        # this prep really depends on how we arrange the layers so I have hardcoded sizes here
        n_dim_in_0 = inp_dim
        n_dim_out_0 = out_dim
        self.layers = [0]
        self.n_spatial_locations = [1.]
        for layer in self.layers:
            self.m_aa[layer] = tf.ones((n_dim_in_0, n_dim_in_0))
            self.m_ss[layer] = tf.ones((n_dim_out_0, n_dim_out_0))

        # for later when we have conv layers
        # these change depending on the approximation
        # and may change depending on if is sensitivities or activations
        if conv_approx == 'mg':  # martens grosse 2015
            self.compute_norm_m_xx = lambda n, cv: float(n * cv)
            self.compute_norm_g = lambda cv: cv
            self.compute_grad_norm = lambda cv: cv

        # contain different damping methods and computations of the nat grads
        if damping_method == 'factored_tikhonov':
            self.ops = FactoredTikhonov(ft_method, tinv)
        elif damping_method == 'tikhonov':
            self.ops = Tikhonov(tinv)

    def compute_lr(self, iteration):
        return self.lr0 / (1 + self.decay * iteration)

    def compute_updates(self, activations, sensitivities, grads, iteration):

        # update the cov_moving weight dependent on iteration
        cov_moving_weight = tf.minimum(1. - 1. / (1. + iteration), self.cov_moving_weight)
        cov_moving_normalize = self.cov_weight + cov_moving_weight

        # update the learning rate
        self.lr = self.compute_lr(iteration - 1)

        nat_grads = []
        for l, a, s, g, cv in zip(self.layers, activations, sensitivities, grads, self.n_spatial_locations):
            n_samples = a.shape[0]

            # different normaizations for different approximations
            norm_a = self.compute_norm_m_xx(n_samples, cv)
            norm_s = self.compute_norm_m_xx(n_samples, cv)
            grad_norm = self.compute_grad_norm(cv)

            if iteration % self.tcov == 0:
                # absorb the spatial dim when we have conv layers
                a = self.absorb_cv(a, l)
                s = self.absorb_cv(s, l)

                # compute the expectation of the kronecker factors
                aa = self.outer_product(a, l) / norm_a
                ss = self.outer_product(s, l) / norm_s

                # update the moving averages
                self.update_m_aa_and_m_ss(aa, ss, cov_moving_weight, cov_moving_normalize, l, iteration)

            # computes the nat grads depending on the damping method
            ng = self.ops.compute_nat_grads(self.m_aa[l],
                                            self.m_ss[l],
                                            g,
                                            grad_norm,
                                            self.damping,
                                            l,
                                            iteration)

            nat_grads.append(ng)

        # compute the norm constraint
        eta = self.compute_norm_constraint(nat_grads, grads)

        # apply the learning rate to the updates - in the kfac code this is applied inside an external optimiser
        # so if we want to compare the nat grads we may have to remove self.lr here and compare the raw grads
        updates = [-1. * eta * self.lr * ng for ng in nat_grads]

        return updates

    # m_xx = (cov_moving_weight * m_xx + cov_weight * xx)  / normalization
    def update_m_aa_and_m_ss(self, aa, ss, cov_moving_weight, cov_normalize, layer, iteration):

        # tensorflow and or ray has a weird thing about inplace operations?? make sure these are updating
        self.m_aa[layer] *= cov_moving_weight / cov_normalize  # multiply
        self.m_aa[layer] += (self.cov_weight * aa) / cov_normalize  # add

        self.m_ss[layer] *= cov_moving_weight / cov_normalize
        self.m_ss[layer] += (self.cov_weight * ss) / cov_normalize
        return

    def compute_norm_constraint(self, nat_grads, grads):
        sq_fisher_norm = 0
        for ng, g in zip(nat_grads, grads):
            sq_fisher_norm += tf.reduce_sum(ng * g)
        eta = tf.minimum(1., tf.sqrt(self.norm_constraint / (self.lr ** 2 * sq_fisher_norm)))
        return eta

    @staticmethod
    def outer_product(x, l):
        return tf.matmul(x, x, transpose_a=True)  # can change this for when we have conv layers

    @staticmethod
    def absorb_cv(x, l):  # can change to absorb the conv dimension via reshapes
        return x


def compute_eig_decomp(maa, mss):
    # enforce symmetry
    maa = (tf.linalg.matrix_transpose(maa) + maa) / 2
    mss = (tf.linalg.matrix_transpose(mss) + mss) / 2

    # get the eigenvalues and eigenvectors of a symmetric positive matrix
    with tf.device("/cpu:0"):
        vals_a, vecs_a = tf.linalg.eigh(maa)
        vals_s, vecs_s = tf.linalg.eigh(mss)

    # zero negative eigenvalues. eigh outputs VALUES then VECTORS
    # print('zero')
    # print(vals_a.shape, vecs_a.shape)
    vals_a = tf.maximum(vals_a, tf.zeros_like(vals_a))
    vals_s = tf.maximum(vals_s, tf.zeros_like(vals_s))

    return vals_a, vecs_a, vals_s, vecs_s


class Tikhonov():
    def __init__(self, tinv):
        print('Tikhonov damping')
        self.tinv = tinv
        self.eigen_decomp = None

    def compute_nat_grads(self, maa, mss, g, grad_norm, damping, layer, iteration):
        if iteration % self.tinv == 0:
            self.eigen_decomp = compute_eig_decomp(maa, mss)
        vals_a, vecs_a, vals_s, vecs_s = self.eigen_decomp

        v1 = tf.linalg.matmul(vecs_a, g / grad_norm, transpose_a=True) @ vecs_s
        divisor = tf.expand_dims(vals_s, -2) * tf.expand_dims(vals_a, -1)
        v2 = v1 / (
                    divisor + damping / grad_norm)  # comes from pulling the lambda out cv*F + \lambda = cv*(F + \lambda / cv)
        ng = vecs_a @ tf.linalg.matmul(v2, vecs_s, transpose_b=True)

        return ng


class FactoredTikhonov():
    def __init__(self, ft_method, tinv):
        print('Factored Tikhonov damping')
        self.ft_method = 'original'  # this is what is in the code, in the theory pi can be set to 1
        self.tinv = tinv
        self.eigen_decomp = None

    def compute_nat_grads(self, maa, mss, g, grad_norm, damping, layer, iteration):

        maa, mss = self.damp(maa, mss, grad_norm, damping, layer, iteration)

        if iteration % self.tinv == 0:
            self.eigen_decomp = compute_eig_decomp(maa, mss)
        vals_a, vecs_a, vals_s, vecs_s = self.eigen_decomp

        v1 = tf.linalg.matmul(vecs_a, g / grad_norm, transpose_a=True) @ vecs_s
        divisor = tf.expand_dims(vals_s, -2) * tf.expand_dims(vals_a, -1)
        v2 = v1 / divisor
        ng = vecs_a @ tf.linalg.matmul(v2, vecs_s, transpose_b=True)

        return ng

    def damp(self, m_aa, m_ss, grad_norm, damping, name, iteration):  # factored tikhonov damping
        dim_a = m_aa.shape[-1]
        dim_s = m_ss.shape[-1]
        batch_shape = list((1 for _ in m_aa.shape[:-2]))  # needs to be cast as list or disappears in tf.eye

        if self.ft_method == 'ones_pi':
            pi = tf.expand_dims(tf.expand_dims(tf.ones(batch_shape), -1), -1)
        else:
            tr_a = self.get_tr_norm(m_aa)
            tr_s = self.get_tr_norm(m_ss)
            pi = tf.expand_dims(tf.expand_dims((tr_a * dim_s) / (tr_s * dim_a), -1), -1)

        eye_a = tf.eye(dim_a, batch_shape=batch_shape)
        eye_s = tf.eye(dim_s, batch_shape=batch_shape)

        m_aa_damping = tf.sqrt(
            pi * damping / grad_norm)  # comes from pulling the lambda out cv*F + \lambda = cv*(F + \lambda / cv)
        m_ss_damping = tf.sqrt(damping / (pi * grad_norm))

        m_aa += eye_a * m_aa_damping
        m_ss += eye_s * m_ss_damping

        return m_aa, m_ss

    @staticmethod
    def get_tr_norm(m_xx):
        trace = tf.linalg.trace(m_xx)
        # need to double check what the default min value is from the original code
        return tf.maximum(1e-10 * tf.ones_like(trace), trace)