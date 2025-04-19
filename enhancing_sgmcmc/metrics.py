from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import wasserstein_distance

from enhancing_sgmcmc.utils import gaussian_mixture_logprob

# QUESTION: are those valid metrics and is their implementation valid (bot only approximating)
# Especially for the wasserstein there are more accurate implementations, but they are much slower.


def wasserstein_distance_approximation(samples: jnp.ndarray, true_samples: jnp.ndarray) -> float:
    """
    Approximate Wasserstein distance by computing 1D distances along each dimension and averaging.
    """
    samples_np = np.array(samples)
    true_samples_np = np.array(true_samples)

    dim = samples_np.shape[1]

    distances = []
    for d in range(dim):
        dist = wasserstein_distance(samples_np[:, d], true_samples_np[:, d])
        distances.append(dist)

    return np.mean(distances)


def kl_divergence_approximation(
    samples: jnp.ndarray,
    means: jnp.ndarray,
    covs: jnp.ndarray,
    weights: jnp.ndarray,
) -> float:
    """
    Compute the KL divergence between the sampler output and the true distribution.

    Note: This computes the negative log-likelihood of samples under the true distribution,
    which is related to but not exactly the KL divergence. The true KL divergence would require
    knowing the density of the empirical distribution, which is not available.
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
    metrics: List[str] = ["wasserstein", "kldivergence"],
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

    if "kldivergence" in metrics:
        kl_div = kl_divergence_approximation(samples, means=means, covs=covs, weights=weights)
        if verbosity > 1:
            print(f"KL-Divergence: {kl_div}")

    return {
        "wasserstein": float(w_dist) if "wasserstein" in metrics else None,
        "kldivergence": float(kl_div) if "kldivergence" in metrics else None,
    }
