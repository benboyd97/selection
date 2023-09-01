import jax.numpy as np
import numpy as onp
from jax import nn, ops, random
from jax.example_libraries import stax
from jax.example_libraries.stax import (Dense, Tanh, Flatten, Relu, LogSoftmax, Softmax, Exp,Sigmoid,Softplus,LeakyRelu)


DEFAULT_MIN_BIN_WIDTH = 1e-6
DEFAULT_MIN_BIN_HEIGHT = 1e-6
DEFAULT_MIN_DERIVATIVE = 1e-6


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations = bin_locations.at[..., -1].add(eps)  # bin_locations[..., -1] += eps
    return np.sum(inputs[..., None] >= bin_locations, axis=-1) - 1


def unconstrained_RQS(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    #inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    #outside_interval_mask = ~inside_interval_mask


    outputs = np.zeros_like(inputs)
    logabsdet = np.ones_like(inputs)

    unnormalized_derivatives = np.pad(
        unnormalized_derivatives, [(0, 0)] * (len(unnormalized_derivatives.shape) - 1) + [(1, 1)]
    )
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives = unnormalized_derivatives.at[..., 0].set(constant)  # unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives = unnormalized_derivatives.at[..., -1].set(constant)  # unnormalized_derivatives[..., -1] = constant


    logabsdet = np.where(inputs<-tail_bound,0,logabsdet)
    logabsdet = np.where(inputs> tail_bound , 0, logabsdet)

    outputs = np.where(logabsdet==0,inputs,outputs)






    #outputs = outputs.at[outside_interval_mask].set(inputs[outside_interval_mask])
    # outputs[outside_interval_mask] = inputs[outside_interval_mask]
    #logabsdet = logabsdet.at[outside_interval_mask].set(0)
     # logabsdet[outside_interval_mask] = 0

    outs, logdets = RQS(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    outputs = np.where(logabsdet==1,outs,outputs)  # outputs[inside_interval_mask] = outs
    logabsdet = np.where(logabsdet==1,logdets,logabsdet) # logabsdet[inside_interval_mask] = logdets

    return outputs, logabsdet


def RQS(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):


    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = nn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = np.cumsum(widths, axis=-1)
    cumwidths = np.pad(
        cumwidths, [(0, 0)] * (len(cumwidths.shape) - 1) + [(1, 0)], mode="constant", constant_values=0.0
    )
    cumwidths = (right - left) * cumwidths + left
    cumwidths = cumwidths.at[..., 0].set(left)  # cumwidths[..., 0] = left
    cumwidths = cumwidths.at[..., -1].set(right)  # cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + nn.softplus(unnormalized_derivatives)

    heights = nn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = np.cumsum(heights, axis=-1)
    cumheights = np.pad(
        cumheights, [(0, 0)] * (len(cumheights.shape) - 1) + [(1, 0)], mode="constant", constant_values=0.0
    )
    cumheights = (top - bottom) * cumheights + bottom
    cumheights = cumheights.at[..., 0].set(bottom)  # cumheights[..., 0] = bottom
    cumheights = cumheights.at[..., -1].set(top)  # cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = np.take_along_axis(cumwidths, bin_idx, -1)[..., 0]  # cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = np.take_along_axis(widths, bin_idx, -1)[..., 0]  # widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = np.take_along_axis(cumheights, bin_idx, -1)[..., 0]  # cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = np.take_along_axis(delta, bin_idx, -1)[..., 0]  # delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = np.take_along_axis(derivatives, bin_idx, -1)[..., 0]  # derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = np.take_along_axis(
        derivatives[..., 1:], bin_idx, -1
    )  # derivatives[..., 1:].gather(-1, bin_idx)
    input_derivatives_plus_one = input_derivatives_plus_one[..., 0]

    input_heights = np.take_along_axis(heights, bin_idx, -1)[..., 0]  # heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = np.square(b) - 4 * a * c

        root = (2 * c) / (-b - np.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        derivative_numerator = np.square(input_delta) * (
            input_derivatives_plus_one * np.square(root)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * np.square(1 - root)
        )
        logabsdet = np.log(derivative_numerator) - 2 * np.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * np.square(theta) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = np.square(input_delta) * (
            input_derivatives_plus_one * np.square(theta)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * np.square(1 - theta)
        )
        logabsdet = np.log(derivative_numerator) - 2 * np.log(denominator)
        return outputs, logabsdet


def network(rng,conditional_dim,out_dim, hidden_dim):
    init_fun,apply_fun=stax.serial(stax.Dense(hidden_dim), Tanh, stax.Dense(hidden_dim), Tanh, stax.Dense(out_dim),)
    _, params = init_fun(rng, (conditional_dim,))
    return params,apply_fun




def NeuralSpline1D(network,K=5, B=3, hidden_dim=64):
    def init_fun(rng, input_dim, **kwargs):
        params, apply_fun = network(rng,input_dim,3*K-1,hidden_dim)

        def direct_fun(params, x):
            log_det = np.zeros(x.shape[0])
            out = apply_fun(params,x[:,1:]).reshape(-1, 1,3 * K - 1)
            W, H, D = np.array_split(out, 3, axis=2)
            W, H = nn.softmax(W, axis=2), nn.softmax(H, axis=2)
            W, H = 2 * B * W, 2 * B * H
            D = nn.softplus(D)



            out, ld = unconstrained_RQS( x[:,:1], W, H, D, inverse=False, tail_bound=B)
            log_det += np.sum(ld, axis=1)
            return np.concatenate([out,x[:,1:]], axis=1), log_det.reshape((x.shape[0],))

        def inverse_fun(params, z):
            log_det = np.zeros(z.shape[0])
            out = apply_fun(params,z[:,1:]).reshape(-1, 1,3 * K - 1)
            W, H, D = onp.array_split(out, 3, axis=2)
            W, H = nn.softmax(W, axis=2), nn.softmax(H, axis=2)
            W, H = 2 * B * W, 2 * B * H
            D = nn.softplus(D)


            out, ld = unconstrained_RQS( z[:,:1], W, H, D, inverse=True, tail_bound=B)
            log_det += np.sum(ld, axis=1)
            return np.concatenate([out,z[:,1:]], axis=1), log_det.reshape((z.shape[0],))

        return params, direct_fun, inverse_fun

    return init_fun



