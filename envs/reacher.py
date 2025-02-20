from typing import Tuple
from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp

# This is based on original Reacher environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/reacher.py

class Reacher(PipelineEnv):
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
        
        self.state_dim = 10
        self.goal_indices = jp.array([4, 5, 6])

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

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

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_dist": zero,
            "reward_ctrl": zero,
            "success": zero,
            "dist": zero
        }
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}


        target_pos = pipeline_state.x.pos[2]
        tip_pos = (
            pipeline_state.x.take(1)
            .do(base.Transform.create(pos=jp.array([0.11, 0, 0])))
            .pos
        )
        tip_to_target = target_pos - tip_pos
        dist = jp.linalg.norm(tip_to_target)
        reward_dist = -math.safe_norm(tip_to_target)
        reward = reward_dist

        state.metrics.update(
            reward_dist=reward_dist,
            success=jp.array(dist < 0.05, dtype=float),
            dist=dist
        )
        state.info.update(info)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns egocentric observation of target and arm body."""
        theta = pipeline_state.q[:2]
        target_pos = pipeline_state.x.pos[2]
        tip_pos = (
            pipeline_state.x.take(1)
            .do(base.Transform.create(pos=jp.array([0.11, 0, 0])))
            .pos
        )
        tip_vel = (
            base.Transform.create(pos=jp.array([0.11, 0, 0]))
            .do(pipeline_state.xd.take(1))
            .vel
        )
        return jp.concatenate(
            [
                # state
                jp.cos(theta),
                jp.sin(theta),
                tip_pos,
                tip_vel,
                # target/goal
                target_pos,
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
