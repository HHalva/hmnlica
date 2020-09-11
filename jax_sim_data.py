import jax
import jax.numpy as np
import numpy as onp
import scipy as osp

from jax import grad, jit, vmap
from jax import random, scipy
from jax import nn
from jax.config import config

import pdb
import seaborn as sns
import matplotlib.pyplot as plt

# enable 64fp precision
config.update('jax_enable_x64', True)


def gen_source_data(num_comp, num_latent_states, sequence_length,
                    state_stickiness=0.999, Lrange=None, radius=2,
                    random_seed=0):
    """function for generating laplace distributed HMM sources data"""
    # init a random seed such that it's different depending on num states
    onp.random.seed(random_seed+num_comp)

    # create transition matrix
    transition_matrix = onp.zeros((num_latent_states,
                                   num_latent_states), dtype=onp.float64)
    for i in range(num_latent_states):
        for j in range(num_latent_states):
            if j == i:
                transition_matrix[i, j] = state_stickiness
            if j == i+1:
                transition_matrix[i, j] = 1.-state_stickiness
            if i == num_latent_states-1 and j == 0:
                transition_matrix[i, j] = 1.-state_stickiness
    transition_matrix /= transition_matrix.sum(1, keepdims=True)

    # get initial state distrib as left eig vec
    e_vals, e_vecs = osp.linalg.eig(transition_matrix, left=True, right=False)
    one_eig_idx = onp.argmin(onp.abs(onp.real(e_vals)-1.))
    assert onp.imag(e_vals[one_eig_idx]) == 0
    ev_one = onp.real(e_vecs[:, one_eig_idx])
    init_state_probs = ev_one / ev_one.sum()

    # create latent state sequence
    state_sequence = onp.zeros(sequence_length, dtype=onp.int)
    for i in range(sequence_length):
        if i == 0:
            m = onp.random.multinomial(1, pvals=init_state_probs)
        else:
            m = onp.random.multinomial(
                1, pvals=transition_matrix[state_sequence[i-1], :])
        state_sequence[i] = onp.argmax(m)

    # create mean modulation parameters for different latent states
    # these are created on the circumference of an n-sphere
    rads = onp.zeros((num_latent_states, num_comp-1))
    for i in range(num_comp-1):
        if i == num_comp-2:
            rads[:, i] = onp.linspace(0, 2*onp.pi, num_latent_states,
                                      endpoint=False)
        else:
            rads[:, i] = onp.linspace(0, onp.pi, num_latent_states,
                                      endpoint=False)
    means = onp.zeros((num_latent_states, num_comp))
    for i in range(num_latent_states):
        for j in range(num_comp):
            if j == 0:
                means[i, j] = radius*onp.cos(rads[i, j])
            elif j == num_comp-1:
                coord = radius
                for l in range(j):
                    coord *= onp.sin(rads[i, l])
                means[i, j] = coord
            else:
                coord = radius
                for k in range(j+1):
                    if k == j:
                        coord *= onp.cos(rads[i, k])
                    else:
                        coord *= onp.sin(rads[i, k])
                means[i, j] = coord

    # create variances such that there isnt too much overlap
    # also ensure good condition number
    dists = means.repeat(num_latent_states, 0)-onp.tile(
        means, (num_latent_states, 1))
    dists = onp.sqrt((dists**2).sum(1))
    min_dist = onp.min(dists[dists.nonzero()])
    cond_num_best = 1e+9
    for i in range(1000):
        sigmasq = onp.random.uniform(0.01, min_dist/1.65,
                                     size=(num_latent_states, num_comp))
        sigmasq = sigmasq**(1/num_comp) / num_latent_states
        natural_params_linear = means / sigmasq
        natural_params_square = -1. / sigmasq
        natural_params = onp.zeros(shape=(num_latent_states, 2 * num_comp))
        natural_params[:, 0::2] = natural_params_linear
        natural_params[:, 1::2] = natural_params_square
        inv_nat_params = onp.linalg.pinv(natural_params)
        cond_num = onp.linalg.norm(
            natural_params)*onp.linalg.norm(inv_nat_params)
        if cond_num < cond_num_best:
            cond_num_best = cond_num
            sigmasq_best = sigmasq.copy()
    sigmasq = sigmasq_best
    D = onp.zeros(shape=(num_latent_states, num_comp, num_comp),
                  dtype=onp.float64)
    for k in range(num_latent_states):
        D[k] = onp.diag(sigmasq[k])

    # generate HMM data for independent components (sources)
    source_data = onp.zeros((sequence_length, num_comp))
    for t in range(sequence_length):
        k = state_sequence[t]
        source_data[t] = onp.random.multivariate_normal(means[k], D[k])
    return source_data, state_sequence, means, D, transition_matrix


def l2normalize(W, axis=0):
    """function to normalize MLP weight matrices"""
    l2norm = np.sqrt(np.sum(W*W, axis, keepdims=True))
    W = W / l2norm
    return W


def find_mat_cond_thresh(N, weight_range, iter4condthresh=10000,
                         cond_thresh_ratio=0.25, random_seed=0):
    """find condition threshold to ensure invertibility of MLP weights"""
    random_seed = onp.random.seed(random_seed)

    # determine condThresh
    cond_list = onp.zeros([iter4condthresh])
    for i in range(iter4condthresh):
        W = onp.random.uniform(weight_range[0], weight_range[1],
                               [N, N])
        #W = onp.random.standard_normal((N, N))
        W = l2normalize(W)
        cond_list[i] = onp.linalg.cond(W)
    cond_list.sort()
    cond_thresh = cond_list[int(iter4condthresh*cond_thresh_ratio)]
    return cond_thresh


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


def matching_sources_corr(est_sources, true_sources, method="pearson"):
    x = onp.array(est_sources.copy(), dtype=onp.float64)
    y = onp.array(true_sources.copy(), dtype=onp.float64)
    dim = x.shape[1]

    # calculate correlations
    if method == "pearson":
        corr = onp.corrcoef(y, x, rowvar=False)
        corr = corr[0:dim, dim:]
    elif method == "spearman":
        corr, pvals = osp.stats.spearmanr(y, x)
        corr = corr[0:dim, dim:]

    # sort variables to try find matching components
    ridx, cidx = osp.optimize.linear_sum_assignment(-onp.abs(corr))

    # calc with best matching components
    corr_sort_diag = corr[ridx, cidx]
    x_sort = x[:, cidx]
    return corr_sort_diag, x_sort, cidx
