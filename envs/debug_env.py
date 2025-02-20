from typing import Tuple

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp


class Debug(PipelineEnv):

    def __init__(self, backend="generalized", **kwargs):
        path = epath.resource_path("brax") / "envs/assets/reacher.xml"
        sys = mjcf.load(path)

        n_frames = 2

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            sys = sys.replace(
                actuator=sys.actuator.replace(gear=jp.array([25.0, 25.0]))
            )
            n_frames = 4

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)
        
        self.state_dim = 3
        self.goal_indices = jp.array([1, 2])

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=-0.1, maxval=0.1
        )
        qd = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=-0.005, maxval=0.005
        )

        # set the target q, qd
        _, target = self._random_target(rng)
        q = q.at[2:].set(target)
        qd = qd.at[2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)

        seed=0
        env=rng3[0]%1000
        info = {"seed": seed, "env": jp.array(env, dtype=jp.float32)}
        obs = jp.array([env, seed, 0, seed, 0], dtype=jp.float32)

        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_dist": zero,
            "reward_ctrl": zero,
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        env_reseted = jp.where(state.info["steps"], 0, 1)
        seed = state.info["seed"] + env_reseted

        info = {"seed": seed}
        state.info.update(info)

        step_now = state.info["steps"]
        obs = jp.array([state.info["env"] ,seed, step_now, seed, step_now], dtype=jp.float32)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=0.0)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns egocentric observation of target and arm body."""
        return jp.concatenate(
            [
                jp.array(
                    [
                        0,
                    ]
                ),
                jp.array(
                    [
                        0,
                    ]
                ),
                jp.array(
                    [
                        0,
                    ]
                ),
                jp.array(
                    [
                        0,
                    ]
                ),
                jp.array(
                    [
                        0,
                    ]
                ),
            ]
        )

    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a target location in a random circle slightly above xy plane."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        dist = 0.2 * jax.random.uniform(rng1)
        ang = jp.pi * 2.0 * jax.random.uniform(rng2)
        target_x = dist * jp.cos(ang)
        target_y = dist * jp.sin(ang)
        return rng, jp.array([target_x, target_y])
