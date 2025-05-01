from dataclasses import dataclass
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

PRNGKey = jnp.ndarray
ParamTree = Any
DataSet = Tuple[jnp.ndarray, jnp.ndarray]


def generate_gaussian_noise(rng_key, param_tree):
    """Generate Gaussian noise with the same shape as param_tree."""
    treedef = jax.tree.structure(param_tree)
    keys = jax.random.split(rng_key, len(jax.tree.leaves(param_tree)))
    noise_leaves = [
        jax.random.normal(k, shape=leaf.shape) for k, leaf in zip(keys, jax.tree.leaves(param_tree))
    ]
    return jax.tree.unflatten(treedef, noise_leaves)


class SGHMC:
    """Simplified SGHMC algorithm implementation."""

    @dataclass
    class State:
        """State of the SGHMC sampler."""

        position: ParamTree
        momentum: ParamTree
        elementwise_sd: ParamTree
        logdensity_grad: ParamTree

        @property
        def zeros(self):
            """Get zeros with the shape of position."""
            return jax.tree.map(jnp.zeros_like, self.position)

        @property
        def ones(self):
            """Get ones with the shape of position."""
            return jax.tree.map(jnp.ones_like, self.position)

    def __init__(self, grad_estimator: Callable):
        """Initialize the SGHMC sampler.

        Args:
            grad_estimator: Function to compute gradient of log density
        """
        self._grad_estimator = grad_estimator

    def init_state(self, position, elementwise_sd=None) -> State:
        """Initialize the sampler state."""
        if elementwise_sd is None:
            elementwise_sd = jax.tree_map(jnp.ones_like, position)

        return self.State(
            position=position,
            momentum=jax.tree.map(jnp.zeros_like, position),
            elementwise_sd=elementwise_sd,
            logdensity_grad=jax.tree.map(jnp.zeros_like, position),
        )

    def sample_step(
        self,
        state: State,
        rng_key: PRNGKey,
        minibatch: DataSet,
        step_size: float = 0.001,
        mresampling: float = 0.0,
        num_integration_steps: int = 1,
        mdecay: float = 0.05,
    ) -> State:
        """Generate a new sample."""
        key_resampling, key_steps = jax.random.split(rng_key)
        state = self._resample_momentum(
            rng_key=key_resampling, state=state, mresampling=mresampling
        )

        for i in range(num_integration_steps):
            step_key = jax.random.fold_in(key_steps, i)
            state = self._integration_step(
                state=state,
                rng_key=step_key,
                minibatch=minibatch,
                step_size=step_size,
                mdecay=mdecay,
            )

        return state

    def _integration_step(
        self,
        state: State,
        rng_key: PRNGKey,
        minibatch: DataSet,
        step_size: float,
        mdecay: float,
    ) -> State:
        """SGHMC's modified Euler's step."""
        state = self._update_logdensity_grad(state=state, minibatch=minibatch)
        state = self._update_momentum(
            state=state,
            rng_key=rng_key,
            mdecay=mdecay,
            step_size=step_size,
        )
        state = self._update_position(state=state, step_size=step_size)
        return state

    def _update_logdensity_grad(
        self,
        state: State,
        minibatch: DataSet,
    ) -> State:
        """Update gradient values."""
        max_grad = 1e6  # Gradient clipping threshold
        x, _ = minibatch
        _, grad = self._grad_estimator(state.position, x)
        state.logdensity_grad = jax.tree.map(
            lambda g: jnp.where(jnp.isnan(g), 0.0, jnp.clip(g, -max_grad, max_grad)),
            grad,
        )
        return state

    @staticmethod
    def _resample_momentum(
        state: State,
        rng_key: PRNGKey,
        mresampling: float,
    ) -> State:
        """Optionally resample the momentum."""
        if jax.random.bernoulli(rng_key, mresampling):
            state.momentum = jax.tree.map(jnp.zeros_like, state.momentum)
        return state

    @staticmethod
    def _update_momentum(
        state: State,
        rng_key: PRNGKey,
        mdecay: float,
        step_size: float,
    ) -> State:
        """Update momentum."""
        noise = generate_gaussian_noise(rng_key, state.position)
        state.momentum = jax.tree.map(
            lambda mom, grad, sd, n: (1 - mdecay) * mom
            + step_size * grad
            + n * jnp.sqrt(jnp.clip(2 * mdecay * sd - jnp.square(step_size * sd), a_min=1e-7)),
            state.momentum,
            state.logdensity_grad,
            state.elementwise_sd,
            noise,
        )
        return state

    @staticmethod
    def _update_position(state: State, step_size: float) -> State:
        """Update position."""
        state.position = jax.tree.map(
            lambda pos, mom, sd: pos + step_size * mom / sd,
            state.position,
            state.momentum,
            state.elementwise_sd,
        )
        return state
