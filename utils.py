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
