from brax import actuator
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco
import os

# This is based on original Humanoid environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/humanoid.py


# This is chosen to be very close to the z coordinate of the humanoid torso, when it is standing straight
TARGET_Z_COORD = 1.25

class Humanoid(PipelineEnv):
  def __init__(
      self,
      forward_reward_weight=1.25,
      ctrl_cost_weight=0.1,
      healthy_reward=5.0,
      terminate_when_unhealthy=True,
      healthy_z_range=(1.0, 2.0),
      reset_noise_scale=0.0,
      exclude_current_positions_from_observation=False,
      backend='generalized',
      min_goal_dist = 1.0,
      max_goal_dist = 5.0,
      **kwargs,
  ):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', "humanoid.xml")
    sys = mjcf.load(path)

    n_frames = 5

    if backend in ['spring', 'positional']:
      sys = sys.tree_replace({'opt.timestep': 0.0015})
      n_frames = 10
      gear = jp.array([
          350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0,
          350.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])  # pyformat: disable
      sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

    if backend == 'mjx':
      sys = sys.tree_replace({
          'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
          'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
          'opt.iterations': 1,
          'opt.ls_iterations': 4,
      })

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )
    self._target_ind = self.sys.link_names.index('target')
    self._min_goal_dist = min_goal_dist
    self._max_goal_dist = max_goal_dist
    
    self.state_dim = 268
    self.goal_indices = jp.array([0, 1, 2])

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=low, maxval=hi
    )
    qvel = jax.random.uniform(
        rng2, (self.sys.qd_size(),), minval=low, maxval=hi
    )

    _, target = self._random_target(rng)
    qpos = qpos.at[-2:].set(target)

    pipeline_state = self.pipeline_init(qpos, qvel)

    obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'forward_reward': zero,
        'reward_linvel': zero,
        'reward_quadctrl': zero,
        'reward_alive': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'dist': zero,
        'x_velocity': zero,
        'y_velocity': zero,
        "success": zero,
        "success_easy": zero,
    }
    
    info = {"seed":0}
    state = State(pipeline_state, obs, reward, done, metrics)
    state.info.update(info)

    return state

  def step(self, state: State, action: jax.Array) -> State:
    """Runs one timestep of the environment's dynamics."""

    if "steps" in state.info.keys():
        seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
    else:
        seed = state.info["seed"]
    info = {"seed": seed}

    # Scale action from [-1,1] to actuator limits
    action_min = self.sys.actuator.ctrl_range[:, 0]
    action_max = self.sys.actuator.ctrl_range[:, 1]
    action = (action + 1) * (action_max - action_min) * 0.5 + action_min

    pipeline_state0 = state.pipeline_state
    pipeline_state = self.pipeline_step(pipeline_state0, action)

    com_before, *_ = self._com(pipeline_state0)
    com_after, *_ = self._com(pipeline_state)
    velocity = (com_after - com_before) / self.dt
    forward_reward = self._forward_reward_weight * velocity[0]

    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy

    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    obs = self._get_obs(pipeline_state, action)
    distance_to_target = jp.linalg.norm(obs[:3] - obs[-3:])

    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    reward = -distance_to_target + healthy_reward - ctrl_cost
    success = jp.array(distance_to_target < 0.5, dtype=float)
    success_easy = jp.array(distance_to_target < 2., dtype=float)
    state.metrics.update(
        forward_reward=forward_reward,
        reward_linvel=forward_reward,
        reward_quadctrl=-ctrl_cost,
        reward_alive=healthy_reward,
        x_position=com_after[0],
        y_position=com_after[1],
        distance_from_origin=jp.linalg.norm(com_after),
        dist=distance_to_target,
        x_velocity=velocity[0],
        y_velocity=velocity[1],
        success=success,
        success_easy=success_easy,
    )
    state.info.update(info)
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

  def _get_obs(
      self, pipeline_state: base.State, action: jax.Array
  ) -> jax.Array:
    """Observes humanoid body position, velocities, and angles."""
    position = pipeline_state.q
    velocity = pipeline_state.qd

    if self._exclude_current_positions_from_observation:
      position = position[2:]

    com, inertia, mass_sum, x_i = self._com(pipeline_state)
    cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
    com_inertia = jp.hstack(
        [cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]]
    )

    xd_i = (
        base.Transform.create(pos=x_i.pos - pipeline_state.x.pos)
        .vmap()
        .do(pipeline_state.xd)
    )
    com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
    com_ang = xd_i.ang
    com_velocity = jp.hstack([com_vel, com_ang])

    qfrc_actuator = actuator.to_tau(
        self.sys, action, pipeline_state.q, pipeline_state.qd)


    target_pos = pipeline_state.x.pos[-1][:2]
    # external_contact_forces are excluded
    return jp.concatenate([
        position,
        velocity,
        com_inertia.ravel(),
        com_velocity.ravel(),
        qfrc_actuator,
        target_pos,
        jp.array([TARGET_Z_COORD]), # Height of the target is fixed
    ])

  def _com(self, pipeline_state: base.State) -> jax.Array:
    inertia = self.sys.link.inertia
    if self.backend in ['spring', 'positional']:
      inertia = inertia.replace(
          i=jax.vmap(jp.diag)(
              jax.vmap(jp.diagonal)(inertia.i)
              ** (1 - self.sys.spring_inertia_scale)
          ),
          mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
      )
    mass_sum = jp.sum(inertia.mass)
    x_i = pipeline_state.x.vmap().do(inertia.transform)
    com = (
        jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
    )
    return com, inertia, mass_sum, x_i  # pytype: disable=bad-return-type  # jax-ndarray
  
  def _random_target(self, rng: jax.Array):
    rng, rng1, rng2 = jax.random.split(rng, 3)

    # NOTE: this is NOT uniform sampling from 2d torus, it favors closer targets
    dist = jax.random.uniform(rng1, minval=self._min_goal_dist, maxval=self._max_goal_dist)
    ang = jp.pi * 2.0 * jax.random.uniform(rng2)
    target_x = dist * jp.cos(ang)
    target_y = dist * jp.sin(ang)
    return rng, jp.array([target_x, target_y])