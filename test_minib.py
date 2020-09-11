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
from jax_sim_data import visualize_distrib

from sklearn.cluster import KMeans

import pdb
from time import time

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# enable 64-bit floating point precision
config.update('jax_enable_x64', True)


# code to perform training
def train(data_gen_dict, opt_dict, seed_dict,
          visualize_pre=False, visualize_train=False):

    # unpack data generation variables
    N = data_gen_dict['N']
    K = data_gen_dict['K']
    T = data_gen_dict['T']
    p_stay = data_gen_dict['p_stay']
    mix_depth = data_gen_dict['mix_depth']
    hidden_size = data_gen_dict['hidden_size']

    # unpack random seeds
    data_seed = seed_dict['data_seed']
    key_mix_mlp = seed_dict['key_mix_mlp']
    key_est_mlp = seed_dict['key_est_mlp']
    seed_est_distrib = seed_dict['seed_est_distrib']

    # unpack optimization variables
    learning_rate = opt_dict['learning_rate']
    num_epochs = opt_dict['num_epochs']
    decay_rate = opt_dict['decay_rate']
    decay_steps = opt_dict['decay_steps']

    # generate source data
    s_data, state_seq, mu, D, A = gen_source_data(N, K, T,
                                                  p_stay,
                                                  random_seed=data_seed)

    # mix the sources to create observable signals
    key = random.PRNGKey(key_mix_mlp)
    mix_params = init_invertible_mlp_params(key, N, mix_depth)
    x_data = invertible_mlp_fwd(mix_params, s_data)

    # initialize parameters for mlp function approximator
    key = random.PRNGKey(key_est_mlp)
    layer_sizes = [N]+[hidden_size]*(mix_depth-1)+[N]
    mlp_params = init_mlp_params(key, layer_sizes)
    func_estimator = mlp

    # TEST run with true function
    #func_estimator = invertible_mlp_inverse
    #mlp_params = mix_params 

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

    # TEST train at true distrib params
    #mu_est = mu
    #D_est = D
    #A_est = A
    #pi_est = A.sum(0)/A.sum()

    # set up optimizer
    schedule = optimizers.exponential_decay(learning_rate,
                                            decay_steps=decay_steps,
                                            decay_rate=decay_rate)
    opt_init, opt_update, get_params = optimizers.adam(schedule)

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

    @partial(jit, static_argnums=(2, 3))
    def forward_backward_algo(logp_x, transition_matrix, T, K):
        # transform into probabilities
        x_probs = np.exp(logp_x)

        # set up transition parameters
        A_est_ = transition_matrix
        pi_est_ = A_est_.sum(0) / A_est_.sum()

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

    @partial(jit, static_argnums=(2, 3))
    def viterbi_algo(logp_x, transition_matrix, T, K):
        # set up transition parameters
        A_est_ = transition_matrix
        pi_est_ = A_est_.sum(0) / A_est_.sum()

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
    def calc_loss(params, input_data, marginal_posteriors, mu_est, D_est):
        lp_x, lp_x_exc_J, lp_J, _ = calc_likelihood(params, input_data,
                                                    mu_est, D_est)
        expected_lp_x = np.sum(marginal_posteriors*lp_x)
        return -expected_lp_x

    @jit
    def m_step(s_est, marg_posteriors, pw_posteriors):
        # update mean parameters
        mu_est = (np.expand_dims(s_est, 0)
                  * np.expand_dims(marg_posteriors.T, 2)).sum(1)
        mu_est /= marg_posteriors.sum(0).reshape(-1, 1)

        # update covariance matrices for all latent states
        dist_to_mu = np.expand_dims(s_est, 0) - np.expand_dims(mu_est, 1)
        cov_est = np.einsum('ktn, ktm->ktnm', dist_to_mu, dist_to_mu)
        wgt_cov_est = (cov_est*marg_posteriors.T[:, :, np.newaxis,
                                                 np.newaxis]).sum(1)
        D_est = wgt_cov_est / marg_posteriors.sum(0)[:, np.newaxis, np.newaxis]
        eps = 0.01
        D_est = np.clip(D_est, a_min=eps)
        D_est = D_est*np.eye(N).reshape(1, N, N)

        # update latent state transitions
        eps = 1e-30
        expected_counts = pw_posteriors.sum(0)
        A_est = expected_counts / expected_counts.sum(1).reshape(-1, 1)
        A_est = np.clip(A_est, a_min=eps)
        pi_est = A_est.sum(0) / A_est.sum()
        return mu_est, D_est, A_est, pi_est

    @jit
    def training_step(iter_num, input_data, marginal_posteriors,
                      mu_est, D_est, opt_state):
        params = get_params(opt_state)
        loss, g = value_and_grad(calc_loss, argnums=0)(
            params, input_data,
            lax.stop_gradient(marginal_posteriors), mu_est, D_est
        )
        return loss, opt_update(iter_num, g, opt_state)

    # run training iterations
    opt_state = opt_init(mlp_params)
    logl_hist = onp.zeros(num_epochs)
    acc_hist = onp.zeros(num_epochs)
    corr_hist = onp.zeros(num_epochs)
    for epoch in range(num_epochs):
        # measure run time of epochs
        tic = time()

        # calculate likelihood using most recent parameters
        params = get_params(opt_state)
        logp_x, logp_x_exc_J, lpj, s_est = calc_likelihood(params, x_data,
                                                           mu_est, D_est)
        print("lpx_exc_J", logp_x_exc_J.sum(), "lpJ:", lpj.sum())

        # forward-backward algorithm
        marg_posteriors, pw_posteriors, scalers = forward_backward_algo(
            logp_x, A_est, T, K)

        # exact M-step for mean and variance
        mu_est, D_est, A_est, pi_est = m_step(s_est, marg_posteriors,
                                              pw_posteriors)
        loss, opt_state = training_step(epoch, x_data, marg_posteriors,
                                        mu_est, D_est, opt_state)

        # calculate marginal likelihood
        logl = onp.log(scalers).sum()
        logl_hist[epoch] = logl

        # to save some compute, dont calc every iteration
        if epoch % 10 == 0 or epoch == num_epochs-1:
            # viterbi to estimate state prediction
            est_seq = viterbi_algo(logp_x, A_est, T, K)
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
        else:
            clustering_acc = -666

        # evaluate correlation of s_est
        s_corr_diag, s_est_sorted, sort_idx = matching_sources_corr(
            s_est, s_data, method="pearson"
        )
        mean_abs_corr = onp.mean(onp.abs(s_corr_diag))
        corr_hist[epoch] = mean_abs_corr
        print("Epoch: [{0}/{1}]\t"
              "logl {logl:.2f}\t"
              "loss {loss:.2f}\t"
              "mean corr between s and s_est {corr:.2f}\t"
              "acc {acc:.2f}".format(
                  epoch, num_epochs, logl=logl, loss=loss,
                  corr=mean_abs_corr, acc=clustering_acc))

        # more evaluation
        print(time()-tic)

    # pack data into tuples
    true_params = (mu, D, A, state_seq)
    est_params = (mu_est, D_est, A_est, est_seq)
    train_trackers = (logl_hist, corr_hist, acc_hist)
    return s_est, sort_idx, train_trackers, true_params, est_params


if __name__ == "__main__":
    data_gen_dict = {'N': 5,
                     'K': 11,
                     'T': 10000,
                     'p_stay': 0.99,
                     'mix_depth': 4,
                     'hidden_size': 10}

    seed_dict = {'data_seed': 0,
                 'key_mix_mlp': 0,
                 'key_est_mlp': 2,
                 'seed_est_distrib': 3}

    opt_dict = {'learning_rate': 1e-2,
                'num_epochs': 1000,
                'decay_rate': 1,
                'decay_steps': 10000}

    s_est, sort_idx, train_trackers, true_params, est_params = train(
        data_gen_dict, opt_dict, seed_dict,
        visualize_pre=False, visualize_train=False,
    )

    pdb.set_trace()
