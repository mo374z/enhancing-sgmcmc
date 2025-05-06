from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import ot
from kernax.kernels import Gaussian, GetSteinFn
from kernax.utils import median_heuristic

from enhancing_sgmcmc.utils import gaussian_mixture_logprob


def wasserstein_distance_approximation(samples: jnp.ndarray, true_samples: jnp.ndarray) -> float:
    """Efficient Wasserstein distance approximation using Sinkhorn algorithm."""
    n_samples = samples.shape[0]
    n_true_samples = true_samples.shape[0]

    # weight the samples uniformly according to their number of samples
    a = jnp.ones((n_samples,)) / n_samples
    b = jnp.ones((n_true_samples,)) / n_true_samples

    M = ot.dist(samples, true_samples)
    reg = 0.01

    wasserstein_dist = ot.sinkhorn2(a, b, M, reg)
    return float(wasserstein_dist)


def negative_log_likelihood(
    samples: jnp.ndarray,
    means: jnp.ndarray,
    covs: jnp.ndarray,
    weights: jnp.ndarray,
) -> float:
    """Compute the average negative log-likelihood of samples under the true Gaussian mixture distribution."""

    def log_prob_for_sample(sample):
        return gaussian_mixture_logprob(sample, means, covs, weights)

    log_probs = jax.vmap(log_prob_for_sample)(samples)
    return -float(jnp.mean(log_probs))


def kernel_stein_discrepancy(
    samples: jnp.ndarray,
    scores: jnp.ndarray,
) -> float:
    """Compute Kernel Stein Discrepancy (KSD) between samples and target distribution."""
    lengthscale = jnp.array(median_heuristic(samples))
    kernel_fn = jax.tree_util.Partial(Gaussian, lengthscale=lengthscale)

    kp_fn = GetSteinFn(kernel_fn)
    kp = jax.vmap(lambda a, b: jax.vmap(lambda c, d: kp_fn(a, b, c, d))(samples, scores))(
        samples, scores
    )
    ksd = jnp.sqrt(jnp.sum(kp)) / samples.shape[0]
    return float(ksd)


def effective_sample_size(
    samples: jnp.ndarray,
    max_lag: Optional[int] = None,
) -> Dict[str, float]:
    """Compute Effective Sample Size (ESS) for each dimension of the samples."""
    n_samples, dim = samples.shape
    if max_lag is None:
        max_lag = min(n_samples // 4, 250)

    def compute_ess_1d(x):
        x_centered = x - jnp.mean(x)  # center the data
        var = jnp.var(x)

        fft_values = jnp.fft.fft(x_centered, n=2 * len(x))
        acf = jnp.fft.ifft(fft_values * jnp.conj(fft_values)).real[: len(x)]
        acf = acf / (var * len(x))
        acf = acf[: max_lag + 1]

        # Use a mask-based approach instead of dynamic slicing
        positive_mask = acf[1:] > 0.05
        masked_acf = jnp.where(positive_mask, acf[1:], 0.0)

        iact = 1 + 2 * jnp.sum(masked_acf)
        ess = n_samples / iact
        return ess, iact

    ess_values, iact_values = jax.vmap(compute_ess_1d, in_axes=1, out_axes=0)(samples)

    return {
        "mean_ess": float(jnp.mean(ess_values)),
        "ess_ratio": float(jnp.min(ess_values) / jnp.max(ess_values)),
        "ess_values": [float(ess) for ess in ess_values],
    }


def create_gmm_score_fn(means, covs, weights):
    """Create score function for Gaussian mixture using JAX autodiff."""

    def score_fn(x):
        return jax.grad(gaussian_mixture_logprob)(x, means, covs, weights)

    return score_fn


def compute_metrics(
    samples: jnp.ndarray,
    true_samples: Optional[jnp.ndarray],
    means: Optional[jnp.ndarray] = None,
    covs: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    metrics: List[str] = ["wasserstein", "nll", "ksd", "ess"],
    verbosity: int = 0,
    **kwargs,
) -> Dict[str, Any]:
    """Compute multiple metrics between sampler output and true distribution."""
    valid_metrics = {"wasserstein", "nll", "ksd", "ess"}
    invalid_metrics = set(metrics) - valid_metrics

    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {invalid_metrics}. Valid metrics are: {valid_metrics}")

    results = {}

    if "wasserstein" in metrics:
        if true_samples is None:
            raise ValueError("Wasserstein distance requires true_samples")
        results["wasserstein"] = wasserstein_distance_approximation(samples, true_samples)

    if "nll" in metrics:
        if any(x is None for x in [means, covs, weights]):
            raise ValueError("NLL requires means, covs, and weights")
        results["nll"] = negative_log_likelihood(samples, means=means, covs=covs, weights=weights)

    if "ksd" in metrics:
        if any(x is None for x in [means, covs, weights]):
            raise ValueError("KSD requires means, covs, and weights")
        score_fn = create_gmm_score_fn(means, covs, weights)
        scores = jax.vmap(score_fn)(samples)
        results["ksd"] = kernel_stein_discrepancy(samples, scores)

    if "ess" in metrics:
        max_lag = kwargs.get("max_lag", None)
        ess_results = effective_sample_size(samples, max_lag=max_lag)
        results.update(ess_results)

    return results
