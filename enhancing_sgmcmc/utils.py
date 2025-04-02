import jax
import jax.numpy as jnp


def generate_gaussian_noise(rng_key, param_tree):
    """Generate Gaussian noise with the same shape as param_tree."""
    treedef = jax.tree.structure(param_tree)
    keys = jax.random.split(rng_key, len(jax.tree.leaves(param_tree)))
    noise_leaves = [
        jax.random.normal(k, shape=leaf.shape) for k, leaf in zip(keys, jax.tree.leaves(param_tree))
    ]
    return jax.tree.unflatten(treedef, noise_leaves)


def gaussian_mixture_logprob(x, means, covs, weights):
    """Log probability of a Gaussian mixture model."""

    def single_gaussian_logprob(x, mean, cov):
        d = x.shape[0]
        delta = x - mean
        return -0.5 * (
            d * jnp.log(2 * jnp.pi)
            + jnp.log(jnp.linalg.det(cov))
            + delta.T @ jnp.linalg.solve(cov, delta)
        )

    logprobs = jnp.array(
        [jnp.log(w) + single_gaussian_logprob(x, m, c) for w, m, c in zip(weights, means, covs)]
    )

    return jax.nn.logsumexp(logprobs)
