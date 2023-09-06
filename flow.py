from jax.scipy.stats import norm, multivariate_normal,uniform
from jax import random,jit
import jax.numpy as np

def Normal():
    """
    Returns:
        A function mapping ``(rng, input_dim)`` to a ``(params, log_pdf, sample)`` triplet.
    """

    def init_fun(rng, input_dim):
        def log_pdf(params, inputs):
            return norm.logpdf(inputs,loc=0,scale=1).sum(1)

        def sample(rng, params, num_samples=1):
            return random.normal(rng, (num_samples, input_dim))

        return (), log_pdf, sample

    return init_fun

def Flow(transformation, prior=Normal()):
    """
    Args:
        transformation: a function mapping ``(rng, input_dim)`` to a
            ``(params, direct_fun, inverse_fun)`` triplet
        prior: a function mapping ``(rng, input_dim)`` to a
            ``(params, log_pdf, sample)`` triplet

    Returns:
        A function mapping ``(rng, input_dim)`` to a ``(params, log_pdf, sample)`` triplet.

    Examples:
        >>> import flows
        >>> input_dim, rng = 3, random.PRNGKey(0)
        >>> transformation = flows.Serial(
        ...     flows.Reverse(),
        ...     flows.Reverse()
        ... )
        >>> init_fun = flows.Flow(transformation, Normal())
        >>> params, log_pdf, sample = init_fun(rng, input_dim)
    """
    def init_fun(rng,conditional_dim):
        transformation_rng, prior_rng = random.split(rng)
        params, direct_fun, inverse_fun = transformation(transformation_rng,conditional_dim)
        prior_params, prior_log_pdf, prior_sample = prior(prior_rng, 1)

        def log_pdf(params, inputs):
            u, log_det = direct_fun(params, inputs)
            log_probs = prior_log_pdf(prior_params, u[:,:1])
            return log_probs + log_det -0.5

        def sample(rng, params,conditionals, num_samples=1):
            prior_samples = prior_sample(rng, prior_params, num_samples)
            x=inverse_fun(params, np.column_stack((prior_samples,conditionals)))[0]
            return x

        return params, log_pdf, sample

    return init_fun

def transform(rng,conditional_dim,hidden_dim=64,hidden_layers=2):
    init_fun, apply_fun= stax.serial(*(Dense(hidden_dim),Relu,)*(hidden_layers-1),Dense(hidden_dim),Relu,Dense(2))
    _, params = init_fun(rng, (conditional_dim,))
    return params, apply_fun



def Serial(*init_funs):
    """
    Args:
        *init_funs: Multiple bijections in sequence

    Returns:
        An ``init_fun`` mapping ``(rng, input_dim)`` to a ``(params, direct_fun, inverse_fun)`` triplet.

    Examples:
        >>> num_examples, input_dim, tol = 20, 3, 1e-4
        >>> layer_rng, input_rng = random.split(random.PRNGKey(0))
        >>> inputs = random.uniform(input_rng, (num_examples, input_dim))
        >>> init_fun = Serial(Shuffle(), Shuffle())
        >>> params, direct_fun, inverse_fun = init_fun(layer_rng, input_dim)
        >>> mapped_inputs = direct_fun(params, inputs)[0]
        >>> reconstructed_inputs = inverse_fun(params, mapped_inputs)[0]
        >>> np.allclose(inputs, reconstructed_inputs).item()
        True
    """

    def init_fun(rng, conditional_dim, **kwargs):
        all_params, direct_funs, inverse_funs = [], [], []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            param, direct_fun, inverse_fun = init_fun(layer_rng,conditional_dim)
            all_params.append(param)
            direct_funs.append(direct_fun)
            inverse_funs.append(inverse_fun)


        def feed_forward(params, apply_funs, inputs):
            log_det_jacobians = np.zeros(inputs.shape[0])

            for apply_fun, param in zip(apply_funs, params):

                inputs, log_det_jacobian = apply_fun(param,inputs,**kwargs)
                log_det_jacobians += log_det_jacobian

            return inputs, log_det_jacobians


        def direct_fun(params, inputs, **kwargs):
            return feed_forward(params, direct_funs, inputs)

        def inverse_fun(params, inputs, **kwargs):
            return feed_forward(reversed(params), reversed(inverse_funs), inputs)

        return all_params, direct_fun, inverse_fun

    return init_fun
