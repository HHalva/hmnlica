import numpy as np
import scipy as sp
import jax.numpy as jnp
import jax.random as jrandom

from jax import grad, value_and_grad, jit, vmap, jacfwd, jacrev
from jax import lax, ops
from jax.config import config
from jax.experimental import optimizers
from functools import partial

from models import invertible_mlp_inverse, init_mlp_params
from models import mlp
from hmm_functions import mbatch_emission_likelihood, emission_likelihood
from hmm_functions import mbatch_fwd_bwd_algo, mbatch_m_step
from hmm_functions import forward_backward_algo, viterbi_algo
from utils import matching_sources_corr

import pdb
import time


# train HM-nICA
def train(x, state_seq, train_dict, seed_dict):
    """Train HM-NLICA model using a minibatch implementation of the algorithm
    described in the paper.

    Args:
        x (matrix): data of observed signals over time in (T, N) matrix.
        state_seq (array): vector containing true state sequence.
        train_dict (dict.): dictionary of variables related to optimization
            of form:
                {'K': num. of latent states to estimate (int),
                 'mix_depth': num. layers in mixing/estimator MLP (int), for
                    example 'mix_depth=1' is linear ICA,
                 'hidden_size': num. hidden units per MLP layer (int),
                 'learning_rate': step size for optimizer (float),
                 'num_epochs': num. training epochs (int),
                 'subseq_len': length of time sequences in a minibatch (int),
                 'minib_size': num. sub-sequences in a minibatch (int),
                 'decay_rate': multiplier for decaying learning rate (float),
                 'decay_steps': num. epochs per which to decay lr (int)}.
        seed_dict (dict.): dictionary of seeds for reproducible stochasticity
            of form:
                {'est_mlp_seed': seed to initialize MLP parameters (int),
                 'est_distrib_seed': seed to initialize exp fam params (int)}.

    Returns:
        Output description.
    """
    # set data dimensions
    N = x.shape[1]
    T = x.shape[0]

    # unpack training variables
    K = train_dict['K']
    mix_depth = train_dict['mix_depth']
    hidden_size = train_dict['hidden_size']
    learning_rate = train_dict['learning_rate']
    num_epochs = train_dict['num_epochs']
    subseq_len = train_dict['subseq_len']
    minib_size = train_dict['minib_size']
    decay_rate = train_dict['decay_rate']
    decay_steps = train_dict['decay_steps']

    print("Training with N={n}, T={t}, K={k}\t"
          "mix_depth={md}".format(n=N, t=T, k=K, md=mix_depth))
    # initialize parameters for mlp function approximator
    key = jrandom.PRNGKey(seed_dict['est_mlp_seed'])
    layer_sizes = [N]+[hidden_size]*(mix_depth-1)+[N]
    mlp_params = init_mlp_params(key, layer_sizes)

    # initialize parameters for estimating distribution parameters
    np.random.seed(seed_dict['est_distrib_seed'])
    mu_est = np.random.uniform(-5., 5., size=(K, N))
    var_est = np.random.uniform(1., 2., size=(K, N))
    D_est = np.zeros(shape=(K, N, N))
    for k in range(K):
        D_est[k] = np.diag(var_est[k])

    # initialize transition parameter estimates
    A_est = np.eye(K) + 0.05
    A_est = A_est / A_est.sum(1, keepdims=True)
    pi_est = A_est.sum(0)/A_est.sum()

    # set up optimizer
    schedule = optimizers.exponential_decay(learning_rate,
                                            decay_steps=decay_steps,
                                            decay_rate=decay_rate)
    opt_init, opt_update, get_params = optimizers.adam(schedule)

    # set up loss function and training step
    @jit
    def calc_loss(params, input_data, marginal_posteriors,
                  mu_est, D_est, num_subseqs):
        """Calculates the loss for gradient M-step for function estimator.
        """
        lp_x, lp_x_exc_J, lp_J, _ = mbatch_emission_likelihood(params,
                                                               input_data,
                                                               mu_est, D_est)
        expected_lp_x = jnp.sum(marginal_posteriors*lp_x, -1)
        # note correction for bias below
        return -expected_lp_x.mean()*num_subseqs

    @jit
    def training_step(iter_num, input_data, marginal_posteriors,
                      mu_est, D_est, opt_state, num_subseqs):
        """Performs gradient m-step on the function estimator
               MLP parameters.
        """
        params = get_params(opt_state)
        loss, g = value_and_grad(calc_loss, argnums=0)(
            params, input_data,
            lax.stop_gradient(marginal_posteriors),
            mu_est, D_est, num_subseqs
        )
        return loss, opt_update(iter_num, g, opt_state)

    # function to load subsequence data for minibatches
    @jit
    def get_subseq_data(orig_data, subseq_array_to_fill):
        """Collects all sub-sequences into an array.
        """
        subseq_data = subseq_array_to_fill
        num_subseqs = subseq_data.shape[0]
        subseq_len = subseq_data.shape[1]

        def body_fun(i, subseq_data):
            subseq_i = lax.dynamic_slice_in_dim(orig_data, i, subseq_len)
            subseq_data = ops.index_update(subseq_data, ops.index[i, :, :],
                                           subseq_i)
            return subseq_data
        return lax.fori_loop(0, num_subseqs, body_fun, subseq_data)

    # set up minibatch training
    num_subseqs = T-subseq_len+1
    assert num_subseqs >= minib_size
    num_full_minibs, remainder = divmod(num_subseqs, minib_size)
    num_minibs = num_full_minibs + bool(remainder)
    sub_data_holder = jnp.zeros((num_subseqs, subseq_len, N))
    sub_data = get_subseq_data(x, sub_data_holder)
    print("T: {t}\t"
          "subseq_len: {slen}\t"
          "minibatch size: {mbs}\t"
          "num minibatches: {nbs}".format(
              t=T, slen=subseq_len, mbs=minib_size, nbs=num_minibs))

    ## set up trackers
    #logl_hist = np.zeros(num_epochs*num_minibs)
    #loss_hist = np.zeros(num_epochs*num_minibs)
    #acc_hist = np.zeros(num_epochs)
    #corr_hist = np.zeros(num_epochs)

    # initialize and train
    opt_state = opt_init(mlp_params)
    all_subseqs_idx = np.arange(num_subseqs)
    for epoch in range(num_epochs):
        tic = time.time()
        # shuffle subseqs for added stochasticity
        np.random.shuffle(all_subseqs_idx)
        sub_data = sub_data.copy()[all_subseqs_idx]
        # train over minibatches
        for batch in range(num_minibs):
            # keep track of total iteration number
            iter_num = batch + epoch*num_minibs

            # select sub-sequence for current minibatch
            batch_data = sub_data[batch*minib_size:(batch+1)*minib_size]

            # calculate emission likelihood using most recent parameters
            params = get_params(opt_state)
            logp_x, logp_x_exc_J, lpj, s_est = mbatch_emission_likelihood(
                params, batch_data, mu_est, D_est
            )

            print("Avg. lpx_exc_J: {lpx:.2f}\t"
                  "Avg. lpJ: {lpj:.2f}\t".format(
                    lpx=logp_x_exc_J.sum((1, 2)).mean(),
                    lpj=lpj.sum(1).mean()))

            # forward-backward algorithm
            marg_posteriors, pw_posteriors, scalers = mbatch_fwd_bwd_algo(
                logp_x, A_est, pi_est
            )

            # exact M-step for mean and variance
            mu_est, D_est, A_est, pi_est = mbatch_m_step(s_est,
                                                         marg_posteriors,
                                                         pw_posteriors)

            # SGD for mlp parameters
            loss, opt_state = training_step(iter_num, batch_data, marg_posteriors,
                                            mu_est, D_est, opt_state,
                                            num_subseqs)

            # calculate approximate (for subseqs) likelihood
            logl = np.log(scalers).sum(1).mean()
            #logl_hist[iter_num] = logl
            #loss_hist[iter_num] = loss
            print("Epoch: [{0}/{1}]\t"
                  "Iter: [{2}/{3}]\t"
                  "Aprox. LogL {logl:.2f}\t"
                  "Aprox. loss {loss:.2f}".format(
                      epoch, num_epochs,
                      batch, num_minibs,
                      logl=logl, loss=loss))

        # evaluate on full data at the end of epoch
        params_latest = get_params(opt_state)
        logp_x_all, _, _, s_est_all = emission_likelihood(
               params_latest, x, mu_est, D_est
        )
        _, _, scalers = forward_backward_algo(
                logp_x_all, A_est, pi_est
            )
        logl_all = np.log(scalers).sum()

        # viterbi to estimate state prediction
        est_seq = viterbi_algo(logp_x_all, A_est, pi_est)
        est_seq = np.array(est_seq.copy())
        match_counts = np.zeros((K, K))
        for k in range(K):
            for l in range(K):
                est_k_idx = (est_seq == k)
                true_l_idx = (state_seq == l)
                match_counts[k, l] = -np.sum(est_k_idx == true_l_idx)
        _, matchidx = sp.optimize.linear_sum_assignment(match_counts)
        for t in range(T):
            est_seq[t] = matchidx[est_seq[t]]
            clustering_acc = np.sum(state_seq == est_seq)/T
            acc_hist[epoch] = clustering_acc

    #    # evaluate correlation of s_est
    #    s_corr_diag, s_est_sorted, sort_idx = matching_sources_corr(
    #        s_est_all, s_data, method="pearson"
    #    )
    #    mean_abs_corr = np.mean(np.abs(s_corr_diag))
    #    corr_hist[epoch] = mean_abs_corr
    #    print("Seeds: {ds}-{km}-{em}-{ps}\t"
    #          "Epoch: [{0}/{1}]\t"
    #          "LogL: {logl:.2f}\t"
    #          "mean corr between s and s_est {corr:.2f}\t"
    #          "acc {acc:.2f}\t"
    #          "elapsed {time:.2f}".format(
    #              epoch, num_epochs, ds=data_seed, km=key_mix_mlp, em=key_est_mlp,
    #              ps=seed_est_distrib,logl=logl_all, corr=mean_abs_corr,
    #              acc=clustering_acc, time=time.time()-tic))

    ## pack data into tuples
    #est_params = (mu_est, D_est, A_est, est_seq)
    #train_trackers = (logl_hist, corr_hist, acc_hist)
    #return s_est, sort_idx, train_trackers, est_params
