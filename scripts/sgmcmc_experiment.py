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


def process_init_m(value, init_position, data):
    """Process the init_m value from the config file."""
    if value == "identity":
        return jnp.array([1.0, 1.0])
    elif value == "fisher":
        # diagonal approximation of the FIM using the squared gradients
        appr_, grad = gmm_grad_estimator(init_position, data)
        return 1 / jnp.sqrt(grad**2)
    else:
        return jnp.array(value)


def jax_array_representer(dumper, data):
    array_str = jnp.array2string(data, separator=", ")
    return dumper.represent_scalar("!jaxarray", array_str)


def run_experiments(config_path):
    """Run experiments based on config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    experiment_name = config.get("experiment_name")
    seed = config.get("seed")
    verbosity = config.get("verbosity", 1)  # Default to 1 if not provided

    results_dir = Path(f"results/{experiment_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    data_config = config.get("data")
    all_means = data_config.get("means", [])
    all_covs = data_config.get("covs", [])
    all_weights = data_config.get("weights", [])
    n_samples = data_config.get("num_samples")

    # Get hyperparameters for grid search
    step_size_values = config.get("step_size")
    burnin_values = config.get("burnin")
    mcmc_samples_values = config.get("mcmc_samples")
    n_batches_values = config.get("n_batches")
    init_m_values = config.get("init_m")
    mdecay_values = config.get("mdecay")
    mresampling_values = config.get("mresampling")
    init_position = jnp.array(config.get("init_position", [0.0, 0.0]))  # Default if not provided

    sampler = SGHMC(gmm_grad_estimator)

    # Outer loop over data configurations
    for data_idx, (means, covs, weights) in enumerate(zip(all_means, all_covs, all_weights)):
        means = jnp.array(means)
        covs = jnp.array(covs)
        weights = jnp.array(weights)

        # Generate data for this configuration
        samples = generate_gmm_data(
            seed=seed + data_idx,  # Vary seed slightly for each data config
            means=means,
            covs=covs,
            weights=weights,
            n_samples=n_samples,
        )

        # Print analytical FIM for this data configuration
        if verbosity > 0:
            print(f"Data configuration {data_idx + 1}/{len(all_means)}:")
            print("Analytical FIM:")
            for cov in covs:
                print(jnp.linalg.inv(cov))

        # Process init_m values for this data configuration
        processed_init_m = []
        for im in init_m_values:
            processed_init_m.append(process_init_m(im, init_position, samples))

        # Create parameter grid for this data configuration
        param_grid = list(
            product(
                processed_init_m,
                step_size_values,
                mdecay_values,
                burnin_values,
                mcmc_samples_values,
                n_batches_values,
                mresampling_values,
            )
        )

        if verbosity > 1:
            print(f"Parameter grid for data config {data_idx + 1}:")
            print(param_grid)

        # Run experiments for each parameter combination
        for i, (
            init_m,
            step_size,
            mdecay,
            burnin,
            mcmc_samples,
            n_batches,
            mresampling,
        ) in enumerate(param_grid):
            if verbosity > 0:
                print(
                    f"Running experiment {i + 1}/{len(param_grid)} for data config {data_idx + 1}:"
                )
            if verbosity > 1:
                print(f"init_m:\n {init_m} \n\n step_size: {step_size} \n\n mdecay: {mdecay}")

            # Run SGHMC
            start_time = datetime.now()
            trajectory = run_sequential_sghmc(
                sampler=sampler,
                data=samples,
                init_position=init_position,
                init_m=init_m,
                batch_size=len(samples) // n_batches,
                n_samples=mcmc_samples,  # Use mcmc_samples not n_samples here
                step_size=step_size,
                mdecay=mdecay,
                num_integration_steps=1,
                mresampling=mresampling,
                seed=seed + data_idx + i,  # Ensure unique seeds
            )
            end_time = datetime.now()

            # Create experiment ID from current timestamp
            exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = results_dir / f"data_{data_idx}_{exp_id}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            # Save trajectory
            trajectory_path = exp_dir / "trajectory.npy"
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
                title=f"SGHMC Sampling - Data Config {data_idx + 1}",
                burnin=burnin,
                figsize=(10, 5),
            )

            plot_path = exp_dir / "plot.png"
            fig.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)

            yaml.add_representer(jnp.ndarray, jax_array_representer)

            # Save experiment metadata
            metadata = {
                "experiment_id": exp_id,
                "data_config_id": data_idx,
                "parameters": {
                    "init_m": init_m.tolist(),
                    "step_size": step_size,
                    "mdecay": mdecay,
                    "burnin": burnin,
                    "mcmc_samples": mcmc_samples,
                    "n_batches": n_batches,
                    "mresampling": mresampling,
                },
                "data": {
                    "means": [m.tolist() for m in means],
                    "covs": [cov.tolist() for cov in covs],
                    "weights": weights.tolist(),
                    "n_samples": n_samples,
                },
                "results": {
                    "trajectory_path": str(trajectory_path),
                    "plot_path": str(plot_path),
                    "runtime_seconds": (end_time - start_time).total_seconds(),
                },
            }

            def represent_list_compactly(dumper, data):
                if all(isinstance(item, list) for item in data):
                    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
                return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)

            yaml.add_representer(list, represent_list_compactly)

            metadata_path = exp_dir / "metadata.yaml"
            with open(metadata_path, "w") as f:
                yaml.dump(metadata, f)

            if verbosity > 0:
                print(f"Completed experiment {i + 1} in {end_time - start_time} seconds")
                print(f"Results saved to {exp_dir}")
                print("-" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    run_experiments(args.config)


if __name__ == "__main__":
    main()
