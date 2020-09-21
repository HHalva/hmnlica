import pdb
import jax.numpy as jnp
import numpy as np

from jax import random as jrandom
from jax import nn as jnn

from utils import l2normalize, find_mat_cond_thresh, SmoothLeakyRelu


def unif_invertible_layer_weights(key, in_dim, out_dim, w_cond_thresh,
                                  weight_range, bias_range):
    """Create square random weight matrix and bias with uniform
    initialization and good condition number.

    Args:
        key: JAX random key.
        in_dim (int): layer input dimension.
        out_dim (int): layer output dimension.
        w_cond_thresh (float/None): upper threshold for condition number for
            generated weight matrices. If 'None', then no threshold applied.
        weight_range (list): list of [lower_bound, upper_bound] for
            the uniform distribution initialization of weight matrix.
        bias_range (list): list of [lower_bound, upper_bound] for
            the uniform distribution initialization of bias vector.

    Returns:
        Tuple (weight matrix, bias vector).
    """
    # ensure good condition number for weight matrix
    W_key, b_key = jrandom.split(key)
    if w_cond_thresh is None:
        W = jrandom.uniform(W_key, (in_dim, out_dim), minval=weight_range[0],
                            maxval=weight_range[1])
        W = l2normalize(W, 0)
    else:
        cond_W = w_cond_thresh + 1
        while cond_W > w_cond_thresh:
            W_key, subkey = jrandom.split(W_key)
            W = jrandom.uniform(subkey, (in_dim, out_dim),
                                minval=weight_range[0],
                                maxval=weight_range[1])
            W = l2normalize(W, 0)
            cond_W = np.linalg.cond(W)
    b = jrandom.uniform(b_key, (out_dim,), minval=bias_range[0],
                        maxval=bias_range[1])
    return W, b


def init_invertible_mlp_params(key, dim, num_layers,
                               weight_range=[-1., 1.], bias_range=[0., 1.]):
    """Initialize weights and biases of an invertible MLP.

    Note that all weight matrices have equal dimensionalities.

    Args:
        key: JAX random key.
        dim (int): dimensionality of weight matrices.
        num_layers (int): number of layers.
        weight_range (list): list of [lower_bound, upper_bound] for
            the uniform distribution initialization of weight matrix.
        bias_range (list): list of [lower_bound, upper_bound] for
            the uniform distribution initialization of bias vector.

    Returns:
        Nested list where each element is a list [W, b] that contains
        weight matrix and bias for a given layer.
    """
    keys = jrandom.split(key, num_layers)
    ct = find_mat_cond_thresh(dim, weight_range)
    return [unif_invertible_layer_weights(k, d_in, d_out,
                                          ct, weight_range, bias_range)
            for k, d_in, d_out in zip(keys, [dim]*num_layers,
                                      [dim]*num_layers)]


def invertible_mlp_fwd(params, x, slope=0.1):
    """Forward pass through invertible MLP used as the mixing function.

    Args:
        params (list): list where each element is a list of layer weight
            and bias [W, b]. len(params) is the number of layers.
        x (vector): input data, here independent components at specific time.
        slope (float): slope for activation function.

    Return:
        Output of MLP, here observed data of mixed independent components.
    """
    z = x
    for W, b in params[:-1]:
        z = jnp.matmul(z, W)+b
        z = jnn.leaky_relu(z, slope)
    final_W, final_b = params[-1]
    z = jnp.dot(z, final_W) + final_b
    return z


def invertible_mlp_inverse(params, x, lrelu_slope=0.1):
    """Inverse of invertible MLP defined above.

    Args:
        params (list): list where each element is a list of layer weight
            and bias [W, b]. len(params) is the number of layers.
        x (vector): output of forward MLP, here observed data.
        slope (float): slope for activation function.

    Returns:
        Inputs into the MLP. Here the independent components.
    """
    z = x
    params_rev = params[::-1]
    final_W, final_b = params_rev[0]
    z = z - final_b
    z = jnp.dot(z, jnp.linalg.inv(final_W))
    for W, b in params_rev[1:]:
        z = jnn.leaky_relu(z, 1./lrelu_slope)
        z = z - b
        z = jnp.dot(z, jnp.linalg.inv(W))
    return z


def init_mlp_params(key, layer_sizes):
    """Initialize weight and bias parameters of an MLP.

    Args:
        key: JAX random key.
        sizes (list): list of dimensions for each layer. For example MLP with
            one 10-unit hidden layer and 3-dimensional input and output would
            be [3, 10, 3].

    Returns:
        Nested list where each element is a list of weight matrix and bias for
            that layer [W, b].
    """
    keys = jrandom.split(key, len(layer_sizes))
    return [unif_invertible_layer_weights(k, m, n, None, [-1., 1.], [0., 0.1])
            for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]


def mlp(params, inputs, slope=0.1):
    """Forward pass through an MLP with SmoothLeakyRelu activations.

    Args:
        params (list): nested list where each element is a list of weight
            matrix and bias for a given layer. e.g. [[W_0, b_0], [W_1, b_1]].
        inputs (matrix): input data.
        slope (float): slope to control the nonlinearity of the activation
            function.

    Returns:
        Output of the MLP.
    """
    activation = SmoothLeakyRelu(slope)
    z = inputs
    for W, b in params[:-1]:
        z = jnp.matmul(z, W)+b
        z = activation(z)
    final_W, final_b = params[-1]
    z = jnp.matmul(z, final_W) + final_b
    return z
