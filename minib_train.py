import numpy as onp
import scipy as osp
import jax.numpy as np
from jax import grad, value_and_grad, jit, vmap, jacfwd, jacrev
from jax import random, scipy
from jax import lax, ops
from jax.config import config
from jax.experimental import optimizers
from functools import partial

from jax_sim_data import gen_source_data
from jax_sim_data import init_invertible_mlp_params
from jax_sim_data import invertible_mlp_fwd, invertible_mlp_inverse
from jax_sim_data import init_mlp_params, mlp
from jax_sim_data import matching_sources_corr

from viz import visualize_init, visualize_train

import pdb
import time

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# enable 64-bit floating point precision
config.update('jax_enable_x64', True)


# code to perform training
def train(data_gen_dict, opt_dict, seed_dict,
          viz_pre=False, viz_train=False,
          use_true_mix=False, use_true_distrib=False):

    # unpack data generation variables
    N = data_gen_dict['N']
    K = data_gen_dict['K']
    T = data_gen_dict['T']
    p_stay = data_gen_dict['p_stay']
    mix_depth = data_gen_dict['mix_depth']

    # unpack random seeds
    data_seed = seed_dict['data_seed']
    key_mix_mlp = seed_dict['key_mix_mlp']
    key_est_mlp = seed_dict['key_est_mlp']
    seed_est_distrib = seed_dict['seed_est_distrib']

    # unpack optimization variables
    hidden_size = opt_dict['hidden_size']
    learning_rate = opt_dict['learning_rate']
    num_epochs = opt_dict['num_epochs']
    subseq_len = opt_dict['subseq_len']
    minib_size = opt_dict['minib_size']
    decay_rate = opt_dict['decay_rate']
    decay_steps = opt_dict['decay_steps']

    # generate source data
    #s_data, state_seq, mu, D, A = gen_source_data(N, K, T,
    #                                              p_stay,
    #                                              random_seed=data_seed)

    # mix the sources to create observable signals
    #key = random.PRNGKey(key_mix_mlp)
    #mix_params = init_invertible_mlp_params(key, N, mix_depth)
    #x_data = invertible_mlp_fwd(mix_params, s_data)

    # initialize parameters for mlp function approximator
    key = random.PRNGKey(key_est_mlp)
    layer_sizes = [N]+[hidden_size]*(mix_depth-1)+[N]
    mlp_params = init_mlp_params(key, layer_sizes)
    func_estimator = mlp

    # Test with true mix MLP
    if use_true_mix:
        print("Running with true mix MLP forced")
        func_estimator = invertible_mlp_inverse
        mlp_params = mix_params

    # initialize parameters for estimating distribution parameters
    onp.random.seed(seed_est_distrib)
    mu_est = onp.random.uniform(-5., 5., size=(K, N))
    var_est = onp.random.uniform(1., 2., size=(K, N))
    D_est = onp.zeros(shape=(K, N, N), dtype=onp.float64)
    for k in range(K):
        D_est[k] = onp.diag(var_est[k])

    # initialize transition parameter estimates (ensure good condition number)
    A_est = np.eye(K) + 0.05
    A_est = A_est / A_est.sum(1, keepdims=True)
    pi_est = A.sum(0)/A.sum()

    # use true source distributiono parameters (for testing)
    if use_true_distrib:
        print("Running with true source distrib. parameters")
        mu_est = mu
        D_est = D
        A_est = A
        pi_est = A.sum(0)/A.sum()

    # set up optimizer
    schedule = optimizers.exponential_decay(learning_rate,
                                            decay_steps=decay_steps,
                                            decay_rate=decay_rate)
    opt_init, opt_update, get_params = optimizers.adam(schedule)

    # visualize before training
    if viz_pre and N == 2:
        s_est_pre = func_estimator(mlp_params, x_data)
        visualize_init(x_data, s_data, s_est_pre, state_seq)

    # set up functions
    @jit
    def J_loglikeli_contrib(params, input_data):
        J = jacfwd(func_estimator, argnums=1)
        batch_J = vmap(partial(J, params))(input_data)
        _, batch_logdetJ = np.linalg.slogdet(batch_J)
        return batch_logdetJ

    @jit
    def calc_likelihood(params, input_data, mu_est, D_est):
        # create output of the estimated inverse mixing function
        s_est = func_estimator(params, input_data)

        # calculate emission probabilities using current parameters
        T = input_data.shape[0]
        K = mu_est.shape[0]
        logp_x_exc_J = np.zeros(shape=(T, K), dtype=np.float64)
        for k in range(K):
            lpx_per_k = scipy.stats.multivariate_normal.logpdf(s_est,
                                                               mu_est[k],
                                                               D_est[k])
            logp_x_exc_J = ops.index_update(logp_x_exc_J, ops.index[:, k],
                                            lpx_per_k)
        logp_J = J_loglikeli_contrib(params, input_data)
        logp_x = logp_x_exc_J + logp_J.reshape(-1, 1)
        return logp_x, logp_x_exc_J, logp_J, s_est

    @jit
    def batch_calc_likelihood(params, input_data, mu_est, D_est):
        return vmap(calc_likelihood, (None, 0, None, None),
                    (0, 0, 0, 0))(params, input_data, mu_est, D_est)

    @jit
    def forward_backward_algo(logp_x, transition_matrix, init_probs):
        # set T and K
        T, K = logp_x.shape

        # transform into probabilities
        x_probs = np.exp(logp_x)

        # set up transition parameters
        A_est_ = transition_matrix
        pi_est_ = init_probs

        # define forward pass
        def forward_pass(t, fwd_msgs_and_scalers):
            scaled_fwd_msgs, scalers = fwd_msgs_and_scalers
            alpha = x_probs[t]*np.dot(A_est_.T, scaled_fwd_msgs[t-1])
            scaled_fwd_msgs = ops.index_update(scaled_fwd_msgs, t,
                                               alpha / alpha.sum())
            scalers = ops.index_update(scalers, t, alpha.sum())
            fwd_msgs_and_scalers = (scaled_fwd_msgs, scalers)
            return fwd_msgs_and_scalers

        # initialize forward pass
        scalers = np.zeros(T, dtype=np.float64)
        scaled_fwd_msgs = np.zeros(shape=(T, K), dtype=np.float64)
        alpha = x_probs[0]*pi_est_
        scaled_fwd_msgs = ops.index_update(scaled_fwd_msgs, 0,
                                           alpha/alpha.sum())
        scalers = ops.index_update(scalers, 0, alpha.sum())
        fwd_msgs_and_scalers = (scaled_fwd_msgs, scalers)

        # note loop start from 1 since 0 was initialize
        scaled_fwd_msgs, scalers = lax.fori_loop(1, T, forward_pass,
                                                 fwd_msgs_and_scalers)

        # define backward pass
        def backward_pass(t, scaled_bck_msgs):
            beta = np.dot(A_est_, x_probs[-t]
                          * scaled_bck_msgs[-t]) / scalers[-t]
            scaled_bck_msgs = ops.index_update(scaled_bck_msgs,
                                               ops.index[-(t+1)], beta)
            return scaled_bck_msgs

        # initialize backward pass
        scaled_bck_msgs = np.zeros(shape=(T, K), dtype=np.float64)
        beta = np.ones(K, dtype=np.float64)
        scaled_bck_msgs = ops.index_update(scaled_bck_msgs,
                                           ops.index[-1], beta)

        # run backward pass
        scaled_bck_msgs = lax.fori_loop(1, T, backward_pass,
                                        scaled_bck_msgs)

        # calculate posteriors i.e. e-step
        marg_posteriors = scaled_fwd_msgs*scaled_bck_msgs
        pw_posteriors = np.zeros(shape=(T-1, K, K),
                                 dtype=np.float64)

        def calc_pw_posteriors(t, pw_posteriors):
            pwm = np.dot(scaled_fwd_msgs[t].reshape(-1, 1),
                         (scaled_bck_msgs[t+1]
                         * x_probs[t+1]).reshape(1, -1))
            pwm = pwm*A_est_ / scalers[t+1]
            return ops.index_update(pw_posteriors,
                                    ops.index[t, :, :], pwm)

        pw_posteriors = lax.fori_loop(0, T-1,
                                      calc_pw_posteriors,
                                      pw_posteriors)

        # to avoid numerical precision issues
        eps = 1e-30
        marg_posteriors = np.clip(marg_posteriors, a_min=eps)
        pw_posteriors = np.clip(pw_posteriors, a_min=eps)
        return marg_posteriors, pw_posteriors, scalers

    @jit
    def batch_fwd_bwd_algo(logp_x, transition_matrix, init_probs):
        return vmap(forward_backward_algo, (0, None, None),
                    (0, 0, 0))(logp_x, transition_matrix, init_probs)

    @jit
    def viterbi_algo(logp_x, transition_matrix, init_probs):
        # set up T and K
        T, K = logp_x.shape

        # set up transition parameters
        A_est_ = transition_matrix
        pi_est_ = init_probs

        # define forward pass
        def forward_pass(t, fwd_msgs_and_paths):
            fwd_msgs, best_paths = fwd_msgs_and_paths
            msg = logp_x[t]+np.max(np.log(A_est_)
                                   + fwd_msgs[t-1].reshape(-1, 1), 0)
            max_prev_state = np.argmax(np.log(A_est_)
                                       + fwd_msgs[t-1].reshape(-1, 1), 0)
            fwd_msgs = ops.index_update(fwd_msgs, ops.index[t, :], msg)
            best_paths = ops.index_update(best_paths, ops.index[t-1],
                                          max_prev_state)
            fwd_msgs_and_paths = (fwd_msgs, best_paths)
            return fwd_msgs_and_paths

        # initialize forward pass
        fwd_msgs = np.zeros(shape=(T, K), dtype=np.float64)
        best_paths = np.zeros(shape=(T, K), dtype=np.int64)
        msg = logp_x[0] + np.log(pi_est_)
        fwd_msgs = ops.index_update(fwd_msgs, 0,
                                    msg)
        fwd_msgs_and_paths = (fwd_msgs, best_paths)

        # note loop start from 1 since 0 was initialize
        fwd_msgs, best_paths = lax.fori_loop(1, T, forward_pass,
                                             fwd_msgs_and_paths)

        # define backward pass
        def backward_pass(t, the_best_path):
            best_k = best_paths[-(t+1), the_best_path[-t]]
            the_best_path = ops.index_update(the_best_path,
                                             ops.index[-(t+1)], best_k)
            return the_best_path

        # initialize backward pass
        the_best_path = np.zeros(shape=(T,), dtype=np.int64)
        best_k = np.argmax(fwd_msgs[-1])
        the_best_path = ops.index_update(the_best_path,
                                         ops.index[-1], best_k)

        # run backward pass
        the_best_path = lax.fori_loop(1, T, backward_pass,
                                      the_best_path)
        return the_best_path

    @jit
    def batch_m_step(s_est, marg_posteriors, pw_posteriors):
        # update mean parameters
        mu_est = (np.expand_dims(s_est, -1)
                  * np.expand_dims(marg_posteriors, -2)).sum((0, 1))
        mu_est /= marg_posteriors.sum((0, 1)).reshape(1, -1)
        mu_est = mu_est.T

        # update covariance matrices for all latent states
        dist_to_mu = s_est[:, np.newaxis, :, :]-mu_est[np.newaxis, :,
                                                       np.newaxis, :]
        cov_est = np.einsum('bktn, bktm->bktnm', dist_to_mu, dist_to_mu)
        wgt_cov_est = (cov_est*np.transpose(
            marg_posteriors, (0, 2, 1))[:, :, :, np.newaxis,
                                        np.newaxis]).sum((0, 2))
        D_est = wgt_cov_est / marg_posteriors.sum((0, 1))[:, np.newaxis,
                                                          np.newaxis]

        # set lowerbound to avoid heywood cases
        eps = 0.01
        D_est = np.clip(D_est, a_min=eps)
        D_est = D_est*np.eye(N).reshape(1, N, N)

        # update latent state transitions (notice the prior)
        hyperobs = 1 #i.e. a-1 ; a=2 where a is hyperprior or dirichlet
        expected_counts = pw_posteriors.sum((0, 1))
        A_est = (expected_counts + hyperobs) / (
            K*hyperobs + marg_posteriors.sum((0, 1)).reshape(-1, 1))

        # estimate m_step (with prior)
        pi_est = marg_posteriors.mean(0)[0] + 1e-4
        pi_est = pi_est/pi_est.sum()
        return mu_est, D_est, A_est, pi_est

    @partial(jit, static_argnums=(5,))
    def calc_loss(params, input_data, marginal_posteriors,
                  mu_est, D_est, num_subseqs):
        lp_x, lp_x_exc_J, lp_J, _ = batch_calc_likelihood(params, input_data,
                                                          mu_est, D_est)
        expected_lp_x = np.sum(marginal_posteriors*lp_x, -1)
        # note correction for bias below
        return -expected_lp_x.mean()*num_subseqs

    @partial(jit, static_argnums=(6,))
    def training_step(iter_num, input_data, marginal_posteriors,
                      mu_est, D_est, opt_state, num_subseqs):
        params = get_params(opt_state)
        loss, g = value_and_grad(calc_loss, argnums=0)(
            params, input_data,
            lax.stop_gradient(marginal_posteriors),
            mu_est, D_est, num_subseqs
        )
        return loss, opt_update(iter_num, g, opt_state)

    @jit
    def get_subseq_data(orig_data, subseq_array_to_fill):
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
    sub_data_holder = np.zeros((num_subseqs, subseq_len, N))
    sub_data = get_subseq_data(x_data, sub_data_holder)
    print("T: {t}\t"
          "subseq_len: {slen}\t"
          "minibatch size: {mbs}\t"
          "num minibatches: {nbs}".format(
              t=T, slen=subseq_len, mbs=minib_size, nbs=num_minibs))

    # set up trackers
    logl_hist = onp.zeros(num_epochs*num_minibs)
    loss_hist = onp.zeros(num_epochs*num_minibs)
    acc_hist = onp.zeros(num_epochs)
    corr_hist = onp.zeros(num_epochs)

    # initialize and train
    opt_state = opt_init(mlp_params)
    all_subseqs_idx = onp.arange(num_subseqs)
    for epoch in range(num_epochs):
        tic = time.time()
        # shuffle subseqs for added stochasticity
        onp.random.shuffle(all_subseqs_idx)
        sub_data = sub_data.copy()[all_subseqs_idx]
        # train over minibatches
        for batch in range(num_minibs):
            # keep track of total iteration number
            iter_num = batch + epoch*num_minibs

            # select sub-sequence for current minibatch
            batch_data = sub_data[batch*minib_size:(batch+1)*minib_size]

            # calculate likelihood using most recent parameters
            params = get_params(opt_state)
            logp_x, logp_x_exc_J, lpj, s_est = batch_calc_likelihood(
                params, batch_data, mu_est, D_est
            )
            #print("Avg. lpx_exc_J: {lpx:.2f}\t"
            #      "Avg. lpJ: {lpj:.2f}\t".format(
            #      lpx=logp_x_exc_J.sum((1, 2)).mean(),
            #      lpj=lpj.sum(1).mean()))

            # forward-backward algorithm
            marg_posteriors, pw_posteriors, scalers = batch_fwd_bwd_algo(
                logp_x, A_est, pi_est
            )

            # exact M-step for mean and variance
            if not use_true_distrib:
                mu_est, D_est, A_est, pi_est = batch_m_step(s_est, marg_posteriors,
                                                            pw_posteriors)

            # SGD for mlp parameters
            if not use_true_mix:
                loss, opt_state = training_step(iter_num, batch_data, marg_posteriors,
                                                mu_est, D_est, opt_state,
                                                num_subseqs)
            else:
                loss = 0

            # calculate approximate (for subseqs) likelihood
            #logl = onp.log(scalers).sum(1).mean()
            #logl_hist[iter_num] = logl
            #loss_hist[iter_num] = loss
            #print("Epoch: [{0}/{1}]\t"
            #      "Iter: [{2}/{3}]\t"
            #      "Aprox. LogL {logl:.2f}\t"
            #      "Aprox. loss {loss:.2f}".format(
            #          epoch, num_epochs,
            #          batch, num_minibs,
            #          logl=logl, loss=loss))

        # evaluate on full data at the end of epoch
        params_latest = get_params(opt_state)
        logp_x_all, _, _, s_est_all = calc_likelihood(
               params_latest, x_data, mu_est, D_est
        )
        _, _, scalers = forward_backward_algo(
                logp_x_all, A_est, pi_est
            )
        logl_all = onp.log(scalers).sum()

        # viterbi to estimate state prediction
        est_seq = viterbi_algo(logp_x_all, A_est, pi_est)
        est_seq = onp.array(est_seq.copy())
        match_counts = onp.zeros((K, K), dtype=onp.int64)
        for k in range(K):
            for l in range(K):
                est_k_idx = (est_seq == k).astype(onp.int64)
                true_l_idx = (state_seq == l).astype(onp.int64)
                match_counts[k, l] = -onp.sum(est_k_idx == true_l_idx)
        _, matchidx = osp.optimize.linear_sum_assignment(match_counts)
        for t in range(T):
            est_seq[t] = matchidx[est_seq[t]]
            clustering_acc = onp.sum(state_seq == est_seq)/T
            acc_hist[epoch] = clustering_acc

        # evaluate correlation of s_est
        s_corr_diag, s_est_sorted, sort_idx = matching_sources_corr(
            s_est_all, s_data, method="pearson"
        )
        mean_abs_corr = onp.mean(onp.abs(s_corr_diag))
        corr_hist[epoch] = mean_abs_corr
        print("Seeds: {ds}-{km}-{em}-{ps}\t"
              "Epoch: [{0}/{1}]\t"
              "LogL: {logl:.2f}\t"
              "mean corr between s and s_est {corr:.2f}\t"
              "acc {acc:.2f}\t"
              "elapsed {time:.2f}".format(
                  epoch, num_epochs, ds=data_seed, km=key_mix_mlp, em=key_est_mlp,
                  ps=seed_est_distrib,logl=logl_all, corr=mean_abs_corr,
                  acc=clustering_acc, time=time.time()-tic))
        # visualize
        if viz_train:
            visualize_train(s_data, s_est_all, mu_est, D_est)

    # pack data into tuples
    true_params = (mu, D, A, state_seq)
    est_params = (mu_est, D_est, A_est, est_seq)
    train_trackers = (logl_hist, corr_hist, acc_hist)
    return s_est, sort_idx, train_trackers, true_params, est_params


if __name__ == "__main__":
    """seeds used for data generation (data_seed, key_mix_mlp):
        (0, 0)"""
    data_gen_dict = {'N': 2,
                     'K': 5,
                     'T': 100000,
                     'p_stay': 0.99,
                     'mix_depth': 5}

    seed_dict = {'data_seed': 0,
                 'key_mix_mlp': 0,
                 'key_est_mlp': 1,
                 'seed_est_distrib': 1}

    opt_dict = {'hidden_size': 10,
                'learning_rate': 3e-4,
                'num_epochs': 100,
                'subseq_len': 100,
                'minib_size': 64,
                'decay_rate': 0.1,
                'decay_steps': 60000}

    s_est, sort_idx, train_trackers, true_params, est_params = train(
        data_gen_dict, opt_dict, seed_dict,
        viz_pre=False, viz_train=False,
        use_true_mix=False, use_true_distrib=False
    )

    pdb.set_trace()
