import jax.numpy as np
def MADE(transform):
    """An implementation of `MADE: Masked Autoencoder for Distribution Estimation`
    (https://arxiv.org/abs/1502.03509).

    Args:
        transform: maps inputs of dimension ``num_inputs`` to ``2 * num_inputs``

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.
    """

    def init_fun(rng, input_dim,**kwargs):
        params, apply_fun = transform(rng, input_dim)

        def direct_fun(params, inputs, **kwargs):
            log_weight, bias = apply_fun(params, inputs[:,1:]).split(2, axis=1)
            inputs = inputs.at[:,:1].set((inputs[:,:1] - bias) * np.exp(-log_weight))
            log_det_jacobian = -log_weight.sum(-1)
            return inputs, log_det_jacobian

        def inverse_fun(params, inputs, **kwargs):


            log_weight, bias = apply_fun(params, inputs[:,1:]).split(2, axis=1)
            inputs = inputs.at[:,0].set(inputs[:,0] * np.exp(log_weight[:, 0]) + bias[:, 0])
            log_det_jacobian = log_weight.sum(-1)

            return inputs, log_det_jacobian

        return params, direct_fun, inverse_fun

    return init_fun