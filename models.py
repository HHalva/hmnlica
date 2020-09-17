import jax
import jax.numpy as np
import numpy as onp
import scipy as osp

from jax import grad, jit, vmap
from jax import random, scipy
from jax import nn
from jax.config import config

from utils import l2normalize, find_mat_cond_thresh

import pdb
import seaborn as sns
import matplotlib.pyplot as plt


def unif_invertible_layer_weights(key, dim, mat_cond_threshold,
                                  weight_range=[-1., 1.],
                                  bias_range=[0., 1.]):
    """create uniformly distributed square random weight parameter
       that has good condition number to ensure invertibility.
       The weights are normalized to have unit L2-norm"""
    a, b = weight_range
    c, d = bias_range
    ct = mat_cond_threshold
    cond_w = ct + 1
    w_key, b_key = random.split(key)
    while cond_w > ct:
        w_key, subkey = random.split(w_key)
        W = random.uniform(subkey, (dim, dim), minval=a, maxval=b)
        W = l2normalize(W, 0)
        cond_w = onp.linalg.cond(W)
    b = random.uniform(b_key, (dim,), minval=c, maxval=d)
    return W, b


def init_invertible_mlp_params(key, dim, num_layers,
                               weight_range=[-1., 1.], bias_range=[0., 1.]):
    keys = random.split(key, num_layers)
    ct = find_mat_cond_thresh(dim, weight_range)
    return [unif_invertible_layer_weights(k, d, ct, weight_range, bias_range)
            for k, d in zip(keys, [dim for i in range(num_layers)])]


def invertible_mlp_fwd(params, inputs, lrelu_slope=0.1):
    z = inputs
    for W, b in params[:-1]:
        z = np.dot(z, W)
        z = nn.leaky_relu(z+b, lrelu_slope)
    final_W, final_b = params[-1]
    z = np.dot(z, final_W) + final_b
    return z


def invertible_mlp_inverse(params, inputs, lrelu_slope=0.1):
    z = inputs
    params_rev = params[::-1]
    final_W, final_b = params_rev[0]
    z = z - final_b
    z = np.dot(z, np.linalg.inv(final_W))
    for W, b in params_rev[1:]:
        z = nn.leaky_relu(z, 1./lrelu_slope)
        z = z - b
        z = np.dot(z, np.linalg.inv(W))
    return z


def init_layer_params(m, n, key, weight_range=[-1., 1.], bias_range=[0., 0.1]):
    w_key, b_key = random.split(key)
    W = random.uniform(w_key, (n, m),
                       minval=weight_range[0], maxval=weight_range[1],
                       dtype=np.float64)
    W = l2normalize(W, 1)
    b = random.uniform(b_key, (n,),
                       minval=bias_range[0], maxval=bias_range[1],
                       dtype=np.float64)
    return W, b


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_mlp_params(key, sizes):
    keys = random.split(key, len(sizes))
    return [init_layer_params(m, n, k)
            for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def mlp(params, inputs, lrelu_slope=0.2, min_s=0.01):
    z = inputs
    for w, b in params[:-1]:
        z = np.dot(z, w.T)+b
        z = nn.leaky_relu(z, lrelu_slope)
    final_w, final_b = params[-1]
    z = np.dot(z, final_w.T) + final_b
    return z
