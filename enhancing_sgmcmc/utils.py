from typing import Callable, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from enhancing_sgmcmc.samplers.sghmc import SGHMC


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


def generate_gmm_data(seed, means, covs, weights, n_samples=1000):
    """Generate data from a Gaussian mixture model."""
    # Select components based on weights
    rng_key = jax.random.PRNGKey(seed)
    keys = jax.random.split(rng_key, 2)
    components = jax.random.choice(keys[0], jnp.arange(len(weights)), shape=(n_samples,), p=weights)

    # Sample from the selected components
    samples = jnp.zeros((n_samples, means.shape[1]))
    for i, (mean, cov) in enumerate(zip(means, covs)):
        mask = components == i
        n_comp_samples = mask.sum()
        if n_comp_samples > 0:
            comp_key = jax.random.fold_in(keys[1], i)
            comp_samples = jax.random.multivariate_normal(
                comp_key, mean, cov, shape=(n_comp_samples,)
            )
            samples = samples.at[mask].set(comp_samples)

    return samples


def gmm_logprob_data(position, samples, reg=1.0):
    """Log probability using generated data samples."""
    diff = samples - position
    # Simple kernel density estimate
    kernel_values = -0.5 * jnp.sum(diff**2, axis=1) / reg
    return jax.nn.logsumexp(kernel_values) - jnp.log(len(samples))


def gmm_grad_estimator(position, samples):
    """Gradient estimator using data samples (batch size agnostic)."""
    logprob = gmm_logprob_data(position, samples)
    return logprob, jax.grad(gmm_logprob_data)(position, samples)


def process_init_m(value, init_position, data):
    """Process the init_m value from the config file."""
    if value == "identity":
        return jnp.array([1.0, 1.0])
    elif value == "fisher":
        # high-level, fast approximation
        # appr_, grad = gmm_grad_estimator(init_position, data)
        # return 1 / jnp.sqrt(grad**2)
        return compute_fisher_diagonal(init_position, data)
    else:
        return jnp.array(value)


@jax.jit
def compute_fisher_diagonal(position, data):
    """Compute diagonal Fisher Information Matrix approximation with JIT."""
    batch_grad_fn = jax.vmap(lambda sample: jax.grad(gmm_logprob_data)(position, sample[None, :]))
    all_grads = batch_grad_fn(data)
    fisher_diagonal = jnp.mean(all_grads**2, axis=0)
    return 1 / jnp.sqrt(fisher_diagonal + 1e-8)


def generate_minibatch(key, minibatch_size, all_samples):
    """Generate a minibatch from all samples."""
    indices = jax.random.randint(key, shape=(minibatch_size,), minval=0, maxval=len(all_samples))
    return all_samples[indices]


def run_sequential_sghmc(
    sampler,
    init_position,
    data,
    mcmc_samples,
    batch_size,
    init_m=None,
    step_size=0.05,
    mdecay=0.05,
    num_integration_steps=1,
    mresampling=0.0,
    seed=0,
):
    """Run SGHMC with sequential control over batches."""
    init_m = process_init_m(init_m, init_position, data)
    state = sampler.init_state(init_position, init_m)
    trajectory = np.zeros((mcmc_samples, init_position.shape[0]))
    trajectory[0] = np.array(state.position)

    key = jax.random.key(seed)
    batch_key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, num=mcmc_samples - 1)
    batch_keys = jax.random.split(batch_key, num=mcmc_samples - 1)

    for i in range(1, mcmc_samples):
        if batch_size < len(data):  # Minibatch
            batch = generate_minibatch(batch_keys[i - 1], batch_size, data)
        else:  # Full batch
            batch = data

        # Run a single SGHMC step with this batch
        state = sampler.sample_step(
            state=state,
            rng_key=step_keys[i - 1],
            minibatch=(batch, None),  # None for y as we're using unsupervised data
            step_size=step_size,
            mdecay=mdecay,
            num_integration_steps=num_integration_steps,
            mresampling=mresampling,
        )

        trajectory[i] = np.array(state.position)

    return trajectory


def run_experiment(
    means: jnp.ndarray,
    covs: jnp.ndarray,
    weights: jnp.ndarray,
    data_samples: int,
    mcmc_samples: int,
    init_position: jnp.ndarray,
    batch_size: int,
    sampler: Literal["SGHMC"] = "SGHMC",
    init_m=None,
    step_size=0.05,
    mdecay=0.05,
    num_integration_steps=1,
    mresampling=0.0,
    seed=0,
):
    """ "Run a full experiment with SGHMC."""
    sampler = SGHMC(grad_estimator=gmm_grad_estimator)
    data = generate_gmm_data(seed, means, covs, weights, n_samples=data_samples)
    trajectory = run_sequential_sghmc(
        sampler=sampler,
        init_position=init_position,
        data=data,
        n_samples=mcmc_samples,
        batch_size=batch_size,
        init_m=init_m,
        step_size=step_size,
        mdecay=mdecay,
        num_integration_steps=num_integration_steps,
        mresampling=mresampling,
        seed=seed,
    )
    return trajectory


# TODO: allow to plot only one of both subplots
def plot_gmm_sampling(
    trajectory: Optional[NDArray] = None,
    samples: Optional[NDArray] = None,
    means: Optional[NDArray] = None,
    covs: Optional[NDArray] = None,
    weights: Optional[NDArray] = None,
    gaussian_mixture_logprob: Optional[Callable] = None,
    title: str = "MCMC Sampling",
    figsize: Tuple[int, int] = (14, 6),
    burnin: int = 0,
    plot_last_n_samples: int = 0,
    padding: float = 0.5,
    show_samples: bool = True,
    plot_density: Literal["log", "pdf", None] = "pdf",
    show_means: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Enhanced visualization function for MCMC sampling from Gaussian mixture models."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    ax_main = axes[0]

    cmap = plt.get_cmap("Dark2")

    # Compute plot ranges automatically if not provided
    if xlim is None or ylim is None:
        all_data = []
        if trajectory is not None:
            all_data.append(trajectory)
        if samples is not None:
            all_data.append(samples)
        if means is not None:
            all_data.append(means)

        # Compute bounds from all available data
        if all_data:
            all_points = np.vstack([d for d in all_data if len(d) > 0])
            x_min, y_min = all_points.min(axis=0) - padding
            x_max, y_max = all_points.max(axis=0) + padding
        else:
            # default limits
            x_min, x_max, y_min, y_max = -10, 10, -10, 10

        # Use custom limits if provided
        if xlim is not None:
            x_min, x_max = xlim
        if ylim is not None:
            y_min, y_max = ylim
    else:
        x_min, x_max = xlim
        y_min, y_max = ylim

    # Plot density contours if requested
    if (
        plot_density is not None
        and means is not None
        and covs is not None
        and weights is not None
        and gaussian_mixture_logprob is not None
    ):
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        Z_log = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = jnp.array([X[i, j], Y[i, j]])
                Z[i, j] = np.exp(gaussian_mixture_logprob(point, means, covs, weights))
                Z_log[i, j] = gaussian_mixture_logprob(point, means, covs, weights)

        if plot_density == "log":
            Z = Z_log
        ax_main.contour(X, Y, Z, levels=15, cmap="coolwarm_r", alpha=0.5)

    # Plot Gaussian component means if requested
    if show_means and means is not None:
        for mean in means:
            ax_main.scatter(mean[0], mean[1], c="red", s=50, marker="*")
        ax_main.scatter([], [], c="red", s=50, marker="*", label="Gaussian means")

    # Plot data samples if requested
    if show_samples and samples is not None:
        ax_main.scatter(samples[:, 0], samples[:, 1], c="gray", s=10, alpha=1, label="Data samples")

    # Plot MCMC trajectory if available
    if trajectory is not None:
        if burnin > 0:
            ax_main.plot(
                trajectory[:burnin, 0],
                trajectory[:burnin, 1],
                color=cmap(4),
                linewidth=2,
                label=f"Burn-in ({burnin} samples)",
            )

        ax_main.plot(
            trajectory[burnin:, 0],
            trajectory[burnin:, 1],
            color=cmap(0),
            linewidth=2,
            label="Sampling trajectory",
        )

        ax_main.scatter(
            trajectory[0, 0], trajectory[0, 1], color=cmap(3), s=50, marker="o", label="Start"
        )
        ax_main.scatter(
            trajectory[-1, 0], trajectory[-1, 1], color=cmap(2), s=50, marker="o", label="End"
        )

        if plot_last_n_samples > 0:
            last_n = min(plot_last_n_samples, len(trajectory) - burnin)
            ax_main.scatter(
                trajectory[-last_n:, 0],
                trajectory[-last_n:, 1],
                c="black",
                s=10,
                alpha=0.5,
                label=f"Last {last_n} samples",
            )

        # Plot trajectory in time series on the second subplot
        ax_ts = axes[1]
        ax_ts.plot(
            range(len(trajectory)), trajectory[:, 0], color=cmap(1), alpha=0.7, label="x coordinate"
        )
        ax_ts.plot(
            range(len(trajectory)), trajectory[:, 1], color=cmap(5), alpha=0.7, label="y coordinate"
        )

        if burnin > 0:
            ax_ts.axvline(x=burnin, color="black", linestyle="--", alpha=0.7, label="Burn-in End")

        ax_ts.set_title("Coordinates over time")
        ax_ts.set_xlabel("Iteration")
        ax_ts.set_ylabel("Coordinate value")
        ax_ts.grid(True, alpha=0.3)

        # Add legends
        handles2, labels2 = ax_ts.get_legend_handles_labels()
        fig.legend(
            handles2,
            labels2,
            loc="lower center",
            bbox_to_anchor=(0.75, -0.1),
            ncol=len(labels2),
            frameon=False,
            fontsize=8,
        )

    ax_main.set_title(title)
    ax_main.set_xlabel("x")
    ax_main.set_ylabel("y")
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    ax_main.grid(True, alpha=0.3)

    # Add legend for main plot
    handles1, labels1 = ax_main.get_legend_handles_labels()
    if handles1:
        fig.legend(
            handles1,
            labels1,
            loc="lower center",
            bbox_to_anchor=(0.25, -0.1),
            ncol=3 if len(labels1) > 4 else 2,
            frameon=False,
            fontsize=8,
        )

    return fig, axes
