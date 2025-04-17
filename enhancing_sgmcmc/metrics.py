from typing import Callable, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance_nd


def compute_wasserstein_distance(
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


def compute_jensen_shannon_divergence(
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

    # For 1D case
    if samples.shape[1] == 1:
        # Create histograms
        hist_samples, bin_edges = np.histogram(
            samples, bins=n_bins, range=range_bounds, density=True
        )
        hist_true, _ = np.histogram(true_samples, bins=bin_edges, density=True)

        # Ensure non-zero probabilities for numerical stability
        hist_samples = np.clip(hist_samples, 1e-10, None)
        hist_true = np.clip(hist_true, 1e-10, None)

        # Normalize
        hist_samples = hist_samples / np.sum(hist_samples)
        hist_true = hist_true / np.sum(hist_true)

        return jensenshannon(hist_samples, hist_true)

    # For 2D case (most common in toy examples)
    elif samples.shape[1] == 2:
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
        raise ValueError(
            "Jensen-Shannon divergence calculation is only implemented for 1D and 2D data"
        )


def compute_average_log_likelihood(
    samples: np.ndarray, logprob_func: Callable, *args, **kwargs
) -> float:
    """
    Compute the average log likelihood of samples under the true distribution.

    Args:
        samples: Samples from the MCMC sampler, shape (n_samples, dim)
        logprob_func: Function to compute log probability of a sample
        *args, **kwargs: Additional arguments to pass to logprob_func

    Returns:
        The average log likelihood of the samples
    """
    # Ensure samples is a numpy array (convert from JAX array if needed)
    samples = np.array(samples)

    # Compute log likelihood for each sample
    log_likelihoods = np.zeros(len(samples))

    for i, sample in enumerate(samples):
        # Convert sample to JAX array for the logprob function
        jax_sample = jnp.array(sample)
        log_likelihoods[i] = logprob_func(jax_sample, *args, **kwargs)

    # Return the average log likelihood
    return np.mean(log_likelihoods)


def compute_all_metrics(
    samples: np.ndarray,
    true_samples: np.ndarray,
    logprob_func: Callable,
    n_bins: int = 20,
    range_bounds: Optional[Tuple[float, float]] = None,
    sample_weights: Optional[np.ndarray] = None,
    true_weights: Optional[np.ndarray] = None,
    *logprob_args,
    **logprob_kwargs
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
        *logprob_args, **logprob_kwargs: Additional arguments for logprob_func

    Returns:
        Dictionary containing all computed metrics
    """
    # Remove burn-in samples if specified in kwargs
    burnin = logprob_kwargs.pop("burnin", 0)
    if burnin > 0:
        samples = samples[burnin:]

    # Compute all metrics
    w_dist = compute_wasserstein_distance(samples, true_samples, sample_weights, true_weights)

    try:
        js_div = compute_jensen_shannon_divergence(samples, true_samples, n_bins, range_bounds)
    except ValueError:
        # If JS divergence fails (e.g., for high dimensions), set to NaN
        js_div = np.nan

    avg_loglik = compute_average_log_likelihood(
        samples, logprob_func, *logprob_args, **logprob_kwargs
    )

    return {
        "wasserstein_distance": float(w_dist),
        "jensen_shannon_divergence": float(js_div),
        "average_log_likelihood": float(avg_loglik),
    }
