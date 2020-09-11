import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

import pdb

def visualize_init(signals, sources, est_sources, state_seq):
    f, ax = plt.subplots(figsize=(8, 8))
    K = max(state_seq)+1
    for k in range(K):
        idx = (state_seq == k).nonzero()[0]
        ax = sns.kdeplot(sources[idx, 0],
                         sources[idx, 1],
                         shade=True, shade_lowest=False)
    plt.title("Distribution of true sources")
    plt.show()

    f, ax = plt.subplots(figsize=(8, 8))
    for k in range(K):
        idx = (state_seq == k).nonzero()[0]
        ax = sns.kdeplot(signals[idx, 0],
                         signals[idx, 1],
                         shade=True, shade_lowest=False)
    plt.title("Distribution of signals (mixed sources)")
    plt.show()

    f, ax = plt.subplots(figsize=(8, 8))
    for k in range(K):
        idx = (state_seq == k).nonzero()[0]
        ax = sns.kdeplot(est_sources[idx, 0],
                         est_sources[idx, 1],
                         shade=True, shade_lowest=False)
    plt.title("Distribution of estimated sources at initialization")
    plt.show()
    return


def visualize_train(sources, est_sources, mu_est, D_est):
    " All three below should be the same if perfect fit"
    sns.set_style("white")
    T = sources.shape[0]
    K = mu_est.shape[0]
    # choose subset to speed up plotting
    idx = np.random.choice(np.arange(T), 10000, replace=False)
    # plot true sources
    ax = sns.kdeplot(sources[idx, 0], sources[idx, 1], cmap=None,
                     shade=False, shade_lowest=False, alpha=0.5)
    # plot sample from distribution with estimated parameters
    for k in range(K):
        mvn = sp.stats.multivariate_normal(mu_est[k], D_est[k, :, :])
        samp = mvn.rvs((10000,))
        ax = sns.kdeplot(samp.T[0], samp.T[1],
                         shade=True, shade_lowest=False, alpha=0.5)
    # plot source predictions (by function estimator)
    ax = sns.kdeplot(est_sources[idx, 0], est_sources[idx, 1], cmap="rainbow",
                     shade=False, shade_lowest=False, alpha=0.7)
    plt.show()
