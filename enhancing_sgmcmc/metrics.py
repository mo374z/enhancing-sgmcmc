from typing import List, Optional

import jax
import jax.numpy as jnp
import ot

from enhancing_sgmcmc.utils import gaussian_mixture_logprob


def wasserstein_distance_approximation(samples: jnp.ndarray, true_samples: jnp.ndarray) -> float:
    """Efficient Wasserstein distance approximation using Sinkhorn algorithm."""
    samples = jnp.array(samples)

    n_samples = samples.shape[0]
    n_true_samples = true_samples.shape[0]

    a = jnp.ones((n_samples,)) / n_samples
    b = jnp.ones((n_true_samples,)) / n_true_samples

    M = ot.dist(samples, true_samples)
    reg = 0.01  # Regularization parameter for Sinkhorn distance (smaller values yield more accurate results)

    return ot.sinkhorn2(a, b, M, reg)


def negative_log_likelihood(
    samples: jnp.ndarray,
    means: jnp.ndarray,
    covs: jnp.ndarray,
    weights: jnp.ndarray,
) -> float:
    """
    Compute the negative log-likelihood of samples under the true Gaussian mixture distribution.
    Higher values indicate worse fit between the empirical sample distribution and the true distribution.
    """

    def log_prob_for_sample(sample):
        return gaussian_mixture_logprob(sample, means, covs, weights)

    log_probs = jax.vmap(log_prob_for_sample)(samples)
    return -float(jnp.mean(log_probs))


def compute_metrics(
    samples: jnp.ndarray,
    true_samples: Optional[jnp.ndarray],
    means: Optional[jnp.ndarray] = None,
    covs: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    metrics: List[str] = ["wasserstein", "nll"],
    verbosity: int = 0,
) -> dict:
    """
    Compute multiple metrics between sampler output and true distribution.
    """
    if verbosity > 0:
        print("Computing metrics...")

    if "wasserstein" in metrics:
        w_dist = wasserstein_distance_approximation(samples, true_samples)
        if verbosity > 1:
            print(f"Wasserstein distance: {w_dist}")

    if "nll" in metrics:
        kl_div = negative_log_likelihood(samples, means=means, covs=covs, weights=weights)
        if verbosity > 1:
            print(f"KL-Divergence: {kl_div}")

    return {
        "wasserstein": float(w_dist) if "wasserstein" in metrics else None,
        "nll": float(kl_div) if "nll" in metrics else None,
    }
