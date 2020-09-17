import jax.numpy as jnp
import numpy as np
import scipy as sp


def sample_n_sphere(n, k):
    """ Sample uniformly on n-sphere (Marsaglia method)

    Keyword args:
    n -- number of dimensions (here number of components)
    k -- number of points on sphere (here latent states)
    """
    x = np.random.normal(size=(k, n))
    x /= np.linalg.norm(x, 2, axis=1, keepdims=True)
    return x


def dists_on_sphere(x):
    """Calculate sum of squared arc distances
    on an n-sphere for k points.

    Keyword args:
    x -- matrix (shape (k, n)) of k points on an n-sphere
    """
    k = x.shape[0]
    dist_mat = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                dist_mat[i, j] = -1
            else:
                dist_mat[i, j] = np.arccos(np.dot(x[i], x[j]))**2
    return dist_mat


def sample_distant_nsphere_points(n, k, iters=100000):
    """Get maximally distant points on n-sphere when
    sampling uniformly repeatedly.

    Keyword args:
    n -- number of dimensions (here independent components)
    k -- number of points on sphere (here latent states)
    iters -- how many rounds to sample (default=10000)
    """
    best_dist = 0
    for i in range(iters):
        points = sample_n_sphere(n, k)
        dists = dists_on_sphere(points)
        total_dist = jnp.min(dists[dists > 0])
        if total_dist > best_dist:
            best_dist = total_dist.copy()
            best_points = points
    return best_points


def l2normalize(W, axis=0):
    """function to normalize MLP weight matrices"""
    l2norm = jnp.sqrt(jnp.sum(W*W, axis, keepdims=True))
    W = W / l2norm
    return W


def find_mat_cond_thresh(N, weight_range, iter4condthresh=10000,
                         cond_thresh_ratio=0.25, random_seed=0):
    """find condition threshold to ensure invertibility of MLP weights"""
    random_seed = np.random.seed(random_seed)

    # determine condThresh
    cond_list = np.zeros([iter4condthresh])
    for i in range(iter4condthresh):
        W = np.random.uniform(weight_range[0], weight_range[1],
                              [N, N])
        #W = np.random.standard_normal((N, N))
        W = l2normalize(W)
        cond_list[i] = np.linalg.cond(W)
    cond_list.sort()
    cond_thresh = cond_list[int(iter4condthresh*cond_thresh_ratio)]
    return cond_thresh


def matching_sources_corr(est_sources, true_sources, method="pearson"):
    x = np.array(est_sources.copy(), dtype=np.float64)
    y = np.array(true_sources.copy(), dtype=np.float64)
    dim = x.shape[1]

    # calculate correlations
    if method == "pearson":
        corr = np.corrcoef(y, x, rowvar=False)
        corr = corr[0:dim, dim:]
    elif method == "spearman":
        corr, pvals = sp.stats.spearmanr(y, x)
        corr = corr[0:dim, dim:]

    # sort variables to try find matching components
    ridx, cidx = sp.optimize.linear_sum_assignment(-np.abs(corr))

    # calc with best matching components
    corr_sort_diag = corr[ridx, cidx]
    x_sort = x[:, cidx]
    return corr_sort_diag, x_sort, cidx
