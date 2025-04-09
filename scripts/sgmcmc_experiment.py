import argparse
from datetime import datetime
from itertools import product
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml

from enhancing_sgmcmc.samplers.sghmc import SGHMC
from enhancing_sgmcmc.utils import (
    gaussian_mixture_logprob,
    generate_gmm_data,
    gmm_grad_estimator,
    plot_gmm_sampling,
    run_sequential_sghmc,
)


def process_init_m(value):
    """Process the init_m value from the config file."""
    if value == "identity":
        return jnp.array([1.0, 1.0])
    elif value == "fisher":
        # TODO: Implement Fisher information matrix calculation
        return jnp.array([0.1, 0.1])
    else:
        return jnp.array(value)


def run_experiments(config_path):
    """Run experiments based on config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    experiment_name = config.get("experiment_name")
    seed = config.get("seed")

    results_dir = Path(f"results/{experiment_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    data_config = config.get("data")
    means = jnp.array(data_config.get("means"))
    covs = jnp.array(data_config.get("covs"))
    weights = jnp.array(data_config.get("weights"))
    n_samples = data_config.get("num_samples")

    # Generate data
    samples = generate_gmm_data(
        seed=seed, means=means, covs=covs, weights=weights, n_samples=n_samples
    )

    init_m_values = config.get("init_m")
    step_size_values = config.get("step_size")
    mdecay_values = config.get("mdecay")
    burnin_values = config.get("burnin")
    mcmc_samples = config.get("mcmc_samples")
    n_batches_values = config.get("n_batches")
    mresampling_values = config.get("mresampling")

    # Process init_m values
    processed_init_m = []
    for im in init_m_values:
        processed_init_m.append(process_init_m(im))

    param_grid = list(
        product(
            processed_init_m,
            step_size_values,
            mdecay_values,
            burnin_values,
            mcmc_samples,
            n_batches_values,
            mresampling_values,
        )
    )

    print(param_grid)

    sampler = SGHMC(gmm_grad_estimator)
    init_position = jnp.array([0.0, 0.0])

    # Run experiments for each parameter combination
    for i, (init_m, step_size, mdecay, burnin, mcmc_samples, n_batches, mresampling) in enumerate(
        param_grid
    ):
        print(f"Running experiment {i + 1}/{len(param_grid)}:")
        print(f"  init_m: {init_m}, step_size: {step_size}, mdecay: {mdecay}")

        # Run SGHMC
        start_time = datetime.now()
        trajectory = run_sequential_sghmc(
            sampler=sampler,
            data=samples,
            init_position=init_position,
            init_m=init_m,
            batch_size=len(samples) // n_batches,
            n_samples=n_samples,
            step_size=step_size,
            mdecay=mdecay,
            num_integration_steps=1,
            mresampling=mresampling,
            seed=seed,
        )
        end_time = datetime.now()

        # Create experiment ID from current timestamp
        exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        (results_dir / exp_id).mkdir(parents=True, exist_ok=True)

        # Save trajectory
        trajectory_path = results_dir / exp_id / "trajectory.npy"
        with open(trajectory_path, "wb") as f:
            np.save(f, np.array(trajectory))

        # Plot and save figure
        fig, ax = plot_gmm_sampling(
            trajectory=trajectory,
            samples=samples,
            means=means,
            covs=covs,
            weights=weights,
            gaussian_mixture_logprob=gaussian_mixture_logprob,
            title="SGHMC Sampling",
            burnin=burnin,
            figsize=(10, 5),
        )

        plot_path = results_dir / exp_id / "plot.png"
        fig.savefig(plot_path)
        plt.close(fig)

        # Save experiment metadata
        metadata = {
            "experiment_id": exp_id,
            "parameters": {
                "init_m": init_m.tolist(),
                "step_size": step_size,
                "mdecay": mdecay,
                "burnin": burnin,
                "n_batches": n_batches,
                "mresampling": mresampling,
            },
            "data": {
                "means": means.tolist(),
                "covs": covs.tolist(),
                "weights": weights.tolist(),
                "n_samples": n_samples,
            },
            "results": {
                "trajectory_path": str(trajectory_path),
                "plot_path": str(plot_path),
                "runtime_seconds": end_time - start_time,
            },
        }

        metadata_path = results_dir / exp_id / "metadata.yaml"
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f)

        print(f"Completed experiment {i + 1} in {end_time - start_time} seconds")
        print(f"Results saved to {results_dir}")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    run_experiments(args.config)


if __name__ == "__main__":
    main()
