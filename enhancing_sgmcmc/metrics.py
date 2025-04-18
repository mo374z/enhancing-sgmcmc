from typing import Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance_nd

# TODO: try to do as much as possible in jax


def wasserstein_distance(
    samples: np.ndarray,
    true_samples: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    true_weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the Wasserstein distance between sampler output and true distribution.

    Args:
        samples: Samples from the MCMC sampler, shape (n_samples, dim)
        true_samples: Samples from the true distribution, shape (n_true_samples, dim)
        sample_weights: Optional weights for the samples (default: equal weights)
        true_weights: Optional weights for the true samples (default: equal weights)

    Returns:
        The Wasserstein distance between the distributions
    """
    # Ensure the inputs are numpy arrays, not JAX arrays
    samples = np.array(samples)
    true_samples = np.array(true_samples)
    if sample_weights is not None:
        sample_weights = np.array(sample_weights)
    if true_weights is not None:
        true_weights = np.array(true_weights)

    return wasserstein_distance_nd(samples, true_samples, sample_weights, true_weights)


def jensen_shannon_divergence(
    samples: np.ndarray,
    true_samples: np.ndarray,
    n_bins: Union[int, Tuple[int, ...]] = 20,
    range_bounds: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Compute the Jensen-Shannon divergence between histograms of samples.

    Args:
        samples: Samples from the MCMC sampler, shape (n_samples, dim)
        true_samples: Samples from the true distribution, shape (n_true_samples, dim)
        n_bins: Number of bins for histogramming (per dimension)
        range_bounds: Optional bounds for histogram binning (min, max)

    Returns:
        The Jensen-Shannon divergence between the histogrammed distributions
    """
    samples = np.array(samples)
    true_samples = np.array(true_samples)

    if samples.shape[1] == 2:
        if isinstance(n_bins, int):
            n_bins = (n_bins, n_bins)

        if range_bounds is None:
            min_x = min(np.min(samples[:, 0]), np.min(true_samples[:, 0]))
            max_x = max(np.max(samples[:, 0]), np.max(true_samples[:, 0]))
            min_y = min(np.min(samples[:, 1]), np.min(true_samples[:, 1]))
            max_y = max(np.max(samples[:, 1]), np.max(true_samples[:, 1]))
            range_bounds = [[min_x, max_x], [min_y, max_y]]

        # Create 2D histograms
        hist_samples, _, _ = np.histogram2d(
            samples[:, 0], samples[:, 1], bins=n_bins, range=range_bounds, density=True
        )
        hist_true, _, _ = np.histogram2d(
            true_samples[:, 0], true_samples[:, 1], bins=n_bins, range=range_bounds, density=True
        )

        # Flatten the 2D histograms to 1D for JS calculation
        hist_samples_flat = hist_samples.flatten()
        hist_true_flat = hist_true.flatten()

        # Ensure non-zero probabilities for numerical stability
        hist_samples_flat = np.clip(hist_samples_flat, 1e-10, None)
        hist_true_flat = np.clip(hist_true_flat, 1e-10, None)

        # Normalize
        hist_samples_flat = hist_samples_flat / np.sum(hist_samples_flat)
        hist_true_flat = hist_true_flat / np.sum(hist_true_flat)

        return jensenshannon(hist_samples_flat, hist_true_flat)

    else:
        raise ValueError("Jensen-Shannon divergence calculation is only implemented for 2D data")


def avg_log_likelihood(
    samples: jnp.ndarray,
    logprob_func: Callable,
    means: jnp.ndarray,
    covs: jnp.ndarray,
    weights: jnp.ndarray,
) -> float:
    """
    Compute the average log likelihood of samples under the true distribution.

    Args:
        samples: Samples from the MCMC sampler, shape (n_samples, dim)
        logprob_func: Function to compute log probability of a sample
        means: Mean vectors of the mixture components, shape (n_components, dim)
        covs: Covariance matrices of the mixture components, shape (n_components, dim, dim)
        weights: Weights of the mixture components, shape (n_components,)

    Returns:
        The average log likelihood of the samples
    """

    def single_sample_logprob(sample):
        return logprob_func(sample, means, covs, weights)

    log_likelihoods = jax.vmap(single_sample_logprob)(samples)

    return float(jnp.mean(log_likelihoods))


def compute_metrics(
    samples: np.ndarray,
    true_samples: Optional[np.ndarray],
    logprob_func: Optional[Callable],
    n_bins: Optional[int] = 20,
    range_bounds: Optional[Tuple[float, float]] = None,
    sample_weights: Optional[np.ndarray] = None,
    true_weights: Optional[np.ndarray] = None,
    means: Optional[np.ndarray] = None,
    covs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    metrics: List[str] = ["wasserstein", "jensen_shannon", "avg_log_likelihood"],
    verbosity: int = 0,
) -> dict:
    """
    Compute all available metrics between sampler output and true distribution.

    Args:
        samples: Samples from the MCMC sampler, shape (n_samples, dim)
        true_samples: Samples from the true distribution, shape (n_true_samples, dim)
        logprob_func: Function to compute log probability of a sample
        n_bins: Number of bins for histogramming (for Jensen-Shannon)
        range_bounds: Optional bounds for histogram binning
        sample_weights: Optional weights for the samples (for Wasserstein)
        true_weights: Optional weights for the true samples (for Wasserstein)
        *logprob_args: Additional arguments for logprob_func

    Returns:
        Dictionary containing all computed metrics
    """

    if "wasserstein" in metrics:
        w_dist = wasserstein_distance(samples, true_samples, sample_weights, true_weights)
        if verbosity > 1:
            print(f"Wasserstein distance: {w_dist}")

    if "jensen_shannon" in metrics:
        js_div = jensen_shannon_divergence(samples, true_samples, n_bins, range_bounds)
        if verbosity > 1:
            print(f"Jensen-Shannon divergence: {js_div}")

    if "avg_log_likelihood" in metrics:
        avg_loglik = avg_log_likelihood(
            samples, logprob_func, means=means, covs=covs, weights=weights
        )
        if verbosity > 1:
            print(f"Average log likelihood: {avg_loglik}")

    return {
        "wasserstein": w_dist if "wasserstein" in metrics else None,
        "jensen_shannon": js_div if "jensen_shannon" in metrics else None,
        "avg_log_likelihood": avg_loglik if "avg_log_likelihood" in metrics else None,
    }
