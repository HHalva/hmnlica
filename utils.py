import pdb
import jax.numpy as jnp
import numpy as np
import scipy as sp


def sample_n_sphere(n, k):
    """ Sample k points uniformly on n-sphere (Marsaglia method).

    Args:
        n (int): number of dimensions (here number of components).
        k (int): number of points on sphere (here latent states).

    Returns:
        Matrix (k, n) of k sampled points on n-sphere.
    """
    x = np.random.normal(size=(k, n))
    x /= np.linalg.norm(x, 2, axis=1, keepdims=True)
    return x


def dists_on_sphere(x):
    """Calculate sum of squared arc distances
    on an n-sphere for k points.

    Args:
        x (matrix): (k, n) matrix of k points on an n-sphere.

    Returns:
        Distance matrix (k, k) between all the k-points.
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
    """Get k maximally distant points on n-sphere when
    sampling uniformly repeatedly.

    Args:
        n (int): number of dimensions (here independent components).
        k (int): number of points on sphere (here latent states).
        iters (int): how many rounds to sample (default=10000).

    Returns:
        (k, n) matrix of coordinates of maximally mutually distant
        points.
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
    """Normalizes MLP weight matrices.

    Args:
        W (matrix): weight matrix.
        axis (int): axis over which to normalize.

    Returns:
        Matrix l2 normalized over desired axis.
    """
    l2norm = jnp.sqrt(jnp.sum(W*W, axis, keepdims=True))
    W = W / l2norm
    return W


def find_mat_cond_thresh(dim, weight_range, iter4condthresh=10000,
                         cond_thresh_ratio=0.25, random_seed=0):
    """Find condition threshold to help ensure invertibility of matrix

    Empirical estimate of acceptable upper threshold conditioning number.
    Assumes weights are uniform initialized. Generates large number of matrices
    and calculates desired percentile of their condition numbers.

    Args:
        dim (int): dimension of (square) matrix.
        weight_range (list): list of [lower_bound, upper_bound] for
            for uniform distribution initializer.
        iter4condthresh (int): number of iterations to run.
        cond_thresh_ratio (float): in range 0.0-1.0 to control percentile
            for what is considered a 'good' conditioning number out of
            iterations.
        random_seed (int): numpy random seed.

    Returns:
        Condition threshold (float)
    """
    random_seed = np.random.seed(random_seed)
    cond_list = np.zeros([iter4condthresh])
    for i in range(iter4condthresh):
        W = np.random.uniform(weight_range[0], weight_range[1],
                              [dim, dim])
        W = l2normalize(W, 0)
        cond_list[i] = np.linalg.cond(W)
    cond_list.sort()
    cond_thresh = cond_list[int(iter4condthresh*cond_thresh_ratio)]
    return cond_thresh


def SmoothLeakyRelu(slope):
    """Smooth Leaky ReLU activation function.

    Args:
        slope (float): slope to control degree of non-linearity.

    Returns:
       Lambda function for computing smooth Leaky ReLU.
    """
    return lambda x: smooth_leaky_relu(x, alpha=slope)


def smooth_leaky_relu(x, alpha=1.0):
    """Calculate smooth leaky ReLU on an input.

    Source: https://stats.stackexchange.com/questions/329776/approximating-leaky-relu-with-a-differentiable-function

    Args:
        x (float): input value.
        alpha (float): controls level of nonlinearity via slope.

    Returns:
        Value transformed by the smooth leaky ReLU.
    """
    return alpha*x + (1 - alpha)*jnp.logaddexp(x, 0)


def matching_sources_corr(est_sources, true_sources, method="pearson"):
    """Finding matching indices between true and estimated sources.

    Args:
        est_sources (array): data on estimated independent components.
        true_sources (array): data on true independent components.
        method (str): "pearson" or "spearman" correlation method to use.

    Returns:
        corr_sort_diag (array): pairwise correlation matrix between
                                matched sources.
        s_est_sort (array): estimed sources array but columns sorted
                            according to best matching index.
        cid (array): vector of the best matching indices.
    """
    e_s = np.array(est_sources.copy())
    s = np.array(true_sources.copy())
    dim = e_s.shape[1]

    # calculate correlations
    if method == "pearson":
        corr = np.corrcoef(s, e_s, rowvar=False)
        corr = corr[0:dim, dim:]
    elif method == "spearman":
        corr, pvals = sp.stats.spearmanr(s, e_s)
        corr = corr[0:dim, dim:]

    # sort variables to try find matching components
    ridx, cidx = sp.optimize.linear_sum_assignment(-np.abs(corr))

    # calc with best matching components
    corr_sort_diag = corr[ridx, cidx]
    s_est_sort = e_s[:, cidx]
    return corr_sort_diag, s_est_sort, cidx
