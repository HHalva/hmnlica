import pdb

import jax.numpy as jnp
import jax.scipy as jscipy

from jax import jit, vmap, jacfwd
from jax import ops, lax
from models import mlp
from functools import partial

func_estimator = mlp


@jit
def J_loglikeli_contrib(params, input_data):
    """Calculate the contribution of log(det(Jacobian)) to likelihood.

    Args:
        params (list): list of MLP parameters.
        input_data (matrix): observed data in shape (T, N).

    Returns:
        Log determinant of the Jacobian evaluated at observed data points.
    """
    J = jacfwd(func_estimator, argnums=1)
    data_J = vmap(partial(J, params))(input_data)
    _, data_logdetJ = jnp.linalg.slogdet(data_J)
    return data_logdetJ


@jit
def calc_emission_likelihood(params, input_data, mu_est, D_est):
    """Calculate likelihood of the HMM emission distribution for
           each latent state.

    Args:
        params (list): list of MLP parameters.
        input_data (matrix): observed data batch in shape (T, N).
        mu_est (array): estimated means for the K latent state distribution.
        D_est (array): estimated cov. matrices for the K latent states
           distributions.

    Returns:
        A tuple that contains (log prob. of emission distribution,
            log prob. of emission distribution without Jacobian term,
            log prob. contribution of the Jacobian term, estimated
            independent components).
    """
    # estimate the inverse mixing function
    s_est = func_estimator(params, input_data)

    # calculate emission probabilities using current parameters
    T = input_data.shape[0]
    K = mu_est.shape[0]
    logp_x_exc_J = jnp.zeros(shape=(T, K))
    for k in range(K):
        lpx_per_k = jscipy.stats.multivariate_normal.logpdf(s_est,
                                                            mu_est[k],
                                                            D_est[k])
        logp_x_exc_J = ops.index_update(logp_x_exc_J, ops.index[:, k],
                                        lpx_per_k)
    logp_J = J_loglikeli_contrib(params, input_data)
    logp_x = logp_x_exc_J + logp_J.reshape(-1, 1)
    return logp_x, logp_x_exc_J, logp_J, s_est


@jit
def mbatch_calc_emission_likelihood(params, input_data, mu_est, D_est):
    """Calculates emission contribution to data likelihood
        for a mini-batch of sub-sequences.

    This is a minibatch version of 'calc_emission_likelihood' where a minibatch
    is composed of HMM sub-sequences.

    Args:
        params (list): list of MLP parameters.
        input_data (arra): three dimensional array of following form
            [minibatch size, length of sub-sequences, data dimensionality].
        mu_est (array): estimated means for the K latent state distribution.
        D_est (array): estimated cov. matrices for the K latent states
           distributions.

    Returns:
        A tuple that contains:
            (log prob. of emission distribution for all
            the possible latent states, same but without Jacobian term,
            log prob. contribution of the Jacobian term, estimated
            independent components). Each array is in minibatch format.
    """
    return vmap(calc_emission_likelihood, (None, 0, None, None),
                (0, 0, 0, 0))(params, input_data, mu_est, D_est)


@jit
def forward_backward_algo(logp_x, transition_matrix, init_probs):
    """Forward-backward algorithm for HM-nICA.

    Args:
        logp_x (array): log emission probabilities for observed data
           evaluated at all possible latent states.
        transition_matrix (array): current estimate of the transition_matrix.
        init_probs (array): estimates of the initial state probabilites.

    Returns:
        Marginal and pairwise posteriors of latent states, and the HMM
            scaler terms which can be used to compute marginal likelihood.
    """
    # set T and K
    T, K = logp_x.shape

    # transform into probabilities
    x_probs = jnp.exp(logp_x)

    # set up transition parameters
    A_est_ = transition_matrix
    pi_est_ = init_probs

    # define forward pass
    def forward_pass(t, fwd_msgs_and_scalers):
        scaled_fwd_msgs, scalers = fwd_msgs_and_scalers
        alpha = x_probs[t]*jnp.matmul(A_est_.T, scaled_fwd_msgs[t-1])
        scaled_fwd_msgs = ops.index_update(scaled_fwd_msgs, t,
                                           alpha / alpha.sum())
        scalers = ops.index_update(scalers, t, alpha.sum())
        fwd_msgs_and_scalers = (scaled_fwd_msgs, scalers)
        return fwd_msgs_and_scalers

    # initialize forward pass
    scalers = jnp.zeros(T)
    scaled_fwd_msgs = jnp.zeros(shape=(T, K))
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
        beta = jnp.matmul(A_est_, x_probs[-t]
                          * scaled_bck_msgs[-t]) / scalers[-t]
        scaled_bck_msgs = ops.index_update(scaled_bck_msgs,
                                           ops.index[-(t+1)], beta)
        return scaled_bck_msgs

    # initialize backward pass
    scaled_bck_msgs = jnp.zeros(shape=(T, K))
    beta = jnp.ones(K)
    scaled_bck_msgs = ops.index_update(scaled_bck_msgs,
                                       ops.index[-1], beta)

    # run backward pass
    scaled_bck_msgs = lax.fori_loop(1, T, backward_pass,
                                    scaled_bck_msgs)

    # calculate posteriors i.e. e-step
    marg_posteriors = scaled_fwd_msgs*scaled_bck_msgs
    pw_posteriors = jnp.zeros(shape=(T-1, K, K))

    def calc_pw_posteriors(t, pw_posteriors):
        pwm = jnp.dot(scaled_fwd_msgs[t].reshape(-1, 1),
                      (scaled_bck_msgs[t+1] * x_probs[t+1]).reshape(1, -1))
        pwm = pwm*A_est_ / scalers[t+1]
        return ops.index_update(pw_posteriors,
                                ops.index[t, :, :], pwm)

    pw_posteriors = lax.fori_loop(0, T-1,
                                  calc_pw_posteriors,
                                  pw_posteriors)

    # to avoid numerical precision issues
    eps = 1e-30
    marg_posteriors = jnp.clip(marg_posteriors, a_min=eps)
    pw_posteriors = jnp.clip(pw_posteriors, a_min=eps)
    return marg_posteriors, pw_posteriors, scalers


@jit
def mbatch_fwd_bwd_algo(logp_x, transition_matrix, init_probs):
    """Minibatch version of the 'forward_backward_algo()'.

    Args:
        logp_x (array): three-dimensional array where the first
            dimension is equal to the size of the minibatch.
        See 'forward_backward_algo' for other details.

    Returns:
        As in 'forward_backward_algo' except for a minibatch of
            sub-sequences.
    """
    return vmap(forward_backward_algo, (0, None, None),
                (0, 0, 0))(logp_x, transition_matrix, init_probs)


@jit
def mbatch_m_step(s_est, marg_posteriors, pw_posteriors):
    """Performs the m-step for a minibatch of HMM sub-sequences.

    The updates are appropriately weighted across the minibatch.

    Args:
        s_est (array): estimated independent components.
        marg_posteriors (array): marginal posteriors (from E-step).
        pw_posteriors (array): pairwise posteriors (from E-step).

    Returns:
        mu_est (array): estimated means for the latent states.
        D_est (array): estimated variances for the latent states.
        A_est (array): estimated transition matrix.
        pi_est (array): estimated inital state probabilities.
    """
    # update mean parameters
    mu_est = (jnp.expand_dims(s_est, -1)
              * jnp.expand_dims(marg_posteriors, -2)).sum((0, 1))
    mu_est /= marg_posteriors.sum((0, 1)).reshape(1, -1)
    mu_est = mu_est.T

    # update covariance matrices for all latent states
    dist_to_mu = s_est[:, jnp.newaxis, :, :]-mu_est[jnp.newaxis, :,
                                                   jnp.newaxis, :]
    cov_est = jnp.einsum('bktn, bktm->bktnm', dist_to_mu, dist_to_mu)
    wgt_cov_est = (cov_est*jnp.transpose(
        marg_posteriors, (0, 2, 1))[:, :, :, jnp.newaxis,
                                    jnp.newaxis]).sum((0, 2))
    D_est = wgt_cov_est / marg_posteriors.sum((0, 1))[:, jnp.newaxis,
                                                      jnp.newaxis]

    # set lowerbound to avoid heywood cases
    eps = 0.01
    D_est = jnp.clip(D_est, a_min=eps)
    D_est = D_est*jnp.eye(N).reshape(1, N, N)

    # update latent state transitions (notice the prior)
    hyperobs = 1 #i.e. a-1 ; a=2 where a is hyperprior or dirichlet
    expected_counts = pw_posteriors.sum((0, 1))
    A_est = (expected_counts + hyperobs) / (
        K*hyperobs + marg_posteriors.sum((0, 1)).reshape(-1, 1))

    # estimate m_step (with prior)
    pi_est = marg_posteriors.mean(0)[0] + 1e-4
    pi_est = pi_est/pi_est.sum()
    return mu_est, D_est, A_est, pi_est
