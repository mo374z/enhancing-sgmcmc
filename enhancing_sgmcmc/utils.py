from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


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


def plot_gaussian_mixture_sampling(
    trajectory: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
    gaussian_mixture_logprob: Callable,
    title: str = "SGHMC Sampling",
    figsize: Tuple[int, int] = (14, 6),
    burnin: int = 0,
    plot_last_n_samples: int = 500,
    xlim: Tuple[float, float] = (-10, 10),
    ylim: Tuple[float, float] = (-10, 10),
):
    """
    Plot results of SGHMC sampling from a Gaussian mixture model.

    Args:
        trajectory: Sampling trajectory as an array of shape (n_samples, 2)
        means: Means of the Gaussian components
        covs: Covariance matrices of the Gaussian components
        weights: Weights of the Gaussian components
        gaussian_mixture_logprob: Function to compute log probability
        title: Plot title
        figsize: Figure size
        burnin: Number of initial samples to mark as burn-in
        plot_last_n_samples: Number of last samples to highlight
        xlim, ylim: Plot limits
    """
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = jnp.array([X[i, j], Y[i, j]])
            Z[i, j] = np.exp(gaussian_mixture_logprob(point, means, covs, weights))

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    ax_main = axes[0]

    ax_main.contour(X, Y, Z, levels=15, cmap="viridis", alpha=0.5)

    if burnin > 0:
        ax_main.plot(
            trajectory[:burnin, 0],
            trajectory[:burnin, 1],
            "r-",
            alpha=0.5,
            linewidth=0.8,
            label=f"Burn-in ({burnin} samples)",
        )

    ax_main.plot(
        trajectory[burnin:, 0],
        trajectory[burnin:, 1],
        "b-",
        alpha=0.4,
        linewidth=0.8,
        label="Sampling trajectory",
    )

    ax_main.scatter(trajectory[0, 0], trajectory[0, 1], c="green", s=80, marker="o", label="Start")

    ax_main.scatter(trajectory[-1, 0], trajectory[-1, 1], c="purple", s=80, marker="o", label="End")

    last_n = min(plot_last_n_samples, len(trajectory) - burnin)
    ax_main.scatter(
        trajectory[-last_n:, 0],
        trajectory[-last_n:, 1],
        c="black",
        s=15,
        alpha=0.5,
        label=f"Last {last_n} samples",
    )

    for i, (mean, cov) in enumerate(zip(means, covs)):
        ax_main.scatter(mean[0], mean[1], c="red", s=100, marker="*")

    ax_main.scatter([], [], c="red", s=100, marker="*", label="Gaussian means")

    ax_main.set_title(title)
    ax_main.set_xlabel("x")
    ax_main.set_ylabel("y")
    ax_main.set_xlim(xlim)
    ax_main.set_ylim(ylim)
    ax_main.grid(True, alpha=0.3)

    ax_ts = axes[1]

    ax_ts.plot(range(len(trajectory)), trajectory[:, 0], "b-", alpha=0.7, label="x coordinate")
    ax_ts.plot(range(len(trajectory)), trajectory[:, 1], "r-", alpha=0.7, label="y coordinate")

    if burnin > 0:
        ax_ts.axvline(x=burnin, color="black", linestyle="--", alpha=0.7)
        ax_ts.text(
            burnin + 5,
            ax_ts.get_ylim()[0] + 0.1 * (ax_ts.get_ylim()[1] - ax_ts.get_ylim()[0]),
            "Burn-in ends",
            fontsize=8,
        )

    ax_ts.set_title("Coordinates over time")
    ax_ts.set_xlabel("Iteration")
    ax_ts.set_ylabel("Coordinate value")
    ax_ts.grid(True, alpha=0.3)

    handles1, labels1 = ax_main.get_legend_handles_labels()
    handles2, labels2 = ax_ts.get_legend_handles_labels()

    fig.legend(
        handles1,
        labels1,
        loc="lower center",
        bbox_to_anchor=(0.25, -0.1),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.legend(
        handles2,
        labels2,
        loc="lower center",
        bbox_to_anchor=(0.75, -0.1),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    fig.subplots_adjust(bottom=0.2)

    return fig, axes


def run_sghmc_experiment(
    sampler,
    init_position,
    num_samples: int = 3000,
    step_size: float = 0.05,
    mdecay: float = 0.05,
    num_integration_steps: int = 1,
    mresampling: float = 0.01,
    minibatch_generator: Optional[Callable] = None,
    minibatch_size: int = None,
    seed: int = 0,
) -> np.ndarray:
    """Run an SGHMC sampling experiment."""
    state = sampler.init_state(init_position)
    trajectory = np.zeros((num_samples, init_position.shape[0]))
    trajectory[0] = np.array(state.position)

    key = jax.random.PRNGKey(seed)

    for i in range(1, num_samples):
        key, subkey = jax.random.split(key)

        # Generate minibatch if provided, otherwise use dummy minibatch
        if minibatch_generator is not None:
            minibatch_key, subkey = jax.random.split(subkey)
            minibatch = minibatch_generator(minibatch_key, minibatch_size)
        else:
            # Dummy minibatch
            minibatch = (jnp.array([0.0]), jnp.array([0.0]))

        # Run one sampling step
        state = sampler.sample_step(
            state=state,
            rng_key=subkey,
            minibatch=minibatch,
            step_size=step_size,
            mdecay=mdecay,
            num_integration_steps=num_integration_steps,
            mresampling=mresampling,
        )

        trajectory[i] = np.array(state.position)

    return trajectory
