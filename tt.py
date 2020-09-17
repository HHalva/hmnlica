import pdb
import numpy as np
import matplotlib.pyplot as plt


def sample_n_sphere(n, k):
    """ Sample uniformly on n-sphere (Marsaglia method)

    Keyword args:
    n -- number of dimensions (here number of components)
    k -- number of points on sphere (here latent states)
    """
    x = np.random.normal(size=(k, n))
    x /= np.linalg.norm(x, 2, axis=1, keepdims=True)
    return x


def sq_dists_on_sphere(x):
    """Calculate sum of squared square-euclidean distances
    on an n-sphere for k points.

    Keyword args:
    x -- matrix (shape (k, n)) of k points on an n-sphere
    """
    dist_mat_tril = np.tril(np.einsum('ij,kj->ik', x, x), -1)
    sq_dist = np.sum(dist_mat_tril**2)
    return sq_dist


def sample_distant_nsphere_points(n, k, iters=1000):
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
        total_sq_dist = sq_dists_on_sphere(points)
        if total_sq_dist > best_dist:
            best_dist = total_sq_dist.copy()
            best_points = points
    return best_points


pdb.set_trace()
