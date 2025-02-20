
from brax import base
from brax.envs.base import State
from brax.io import mjcf
import mujoco
import jax
from jax import numpy as jnp
from envs.manipulation.arm_envs import ArmEnvs

"""
Binpick-Easy (EEF): Move a cube from a random location in the blue bin to the center of the red bin, controlling the EEF position directly. There are
                    invisible walls (currently renderer is broken so they're visible) to prevent the cube from going too far, which don't collide with the hand.
- Observation space: 11-dim obs + 3-dim goal.
- Action space:      4-dim, each element in [-1, 1], corresponding to target delta position for EEF, and finger closedness.

See _get_obs() and ArmEnvs._convert_action() for details.
"""
class ArmBinpickEasyEEF(ArmEnvs):
    def _get_xml_path(self):
        return "envs/assets/panda_binpick_easy_EEF.xml"
    
    @property
    def action_size(self) -> int:
        return 4 # Override default (actuator count)
    
    # See ArmEnvs._set_environment_attributes for descriptions of attributes
    def _set_environment_attributes(self):
        self.env_name = "arm_binpick_easy_EEF"
        self.episode_length = 150

        self.goal_indices = jnp.array([0, 1, 2]) # Cube position
        self.completion_goal_indices = jnp.array([0, 1, 2]) # Identical
        self.state_dim = 11
        self.goal_dist = 0.1

        self.eef_noise_scale = 0
        self.cube_noise_scale = 0.07
        self.goal_noise_scale = 0.005
    
    def _get_initial_state(self, rng):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        cube_q_xy = self.sys.init_q[:2] + self.cube_noise_scale * jax.random.uniform(subkey1, [2], minval=-1)
        cube_q_remaining = self.sys.init_q[2:7]
        target_q = self.sys.init_q[7:14]
        eef_q_default = jnp.array([0, 0.6, 0.2, 0.04, 0.04]) # Start closer to the relevant area
        eef_q = eef_q_default + self.eef_noise_scale * jax.random.uniform(subkey2, [self.sys.q_size() - 14], minval=-1)
        
        q = jnp.concatenate([cube_q_xy] + [cube_q_remaining] + [target_q] + [eef_q])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd
        
    def _get_initial_goal(self, pipeline_state: base.State, rng):
        rng, subkey = jax.random.split(rng)
        cube_goal_pos = jnp.array([0.17, 0.6, 0.03]) + jnp.array([self.goal_noise_scale, self.goal_noise_scale, 0]) * jax.random.uniform(subkey, [3], minval=-1)
        return cube_goal_pos
        
    def _compute_goal_completion(self, obs, goal):
        # Goal occupancy: is the cube close enough to the goal? 
        current_cube_pos = obs[self.completion_goal_indices]
        goal_pos = goal[:3]
        dist = jnp.linalg.norm(current_cube_pos - goal_pos)

        success = jnp.array(dist < self.goal_dist, dtype=float)
        success_easy = jnp.array(dist < 0.3, dtype=float)
        success_hard = jnp.array(dist < 0.03, dtype=float)
        return success, success_easy, success_hard
    
    def _update_goal_visualization(self, pipeline_state: base.State, goal: jax.Array) -> base.State:
        updated_q = pipeline_state.q.at[7:10].set(goal[:3]) # Only set the position, not orientation
        updated_pipeline_state = pipeline_state.replace(qpos=updated_q)
        return updated_pipeline_state

    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        """
        Observation space (11-dim)
         - q_subset (3-dim): cube position
         - End-effector (6-dim): position and velocity
         - Fingers (2-dim): finger distance, gripper force
        Note q is 19-dim: 7-dim cube position/angle, 7-dim goal marker position/angle, 3-dim EEF position, 2-dim finger offset
         
        Goal space (3-dim): position of cube
        """
        q_indices = jnp.array([0, 1, 2])
        q_subset = pipeline_state.q[q_indices]
        
        eef_index = 2 # Cube is 0, goal marker is 1, then EEF is 2
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]

        left_finger_index = 3
        left_finger_x_pos = pipeline_state.x.pos[left_finger_index]
        right_finger_index = 4
        right_finger_x_pos = pipeline_state.x.pos[right_finger_index]
        finger_distance = jnp.linalg.norm(right_finger_x_pos - left_finger_x_pos, keepdims=True)
        gripper_force = (pipeline_state.qfrc_actuator[:-2]).mean(keepdims=True) * 0.1 # Normalize it from range [-20, 20] to [-2, 2]
        
        return jnp.concatenate([q_subset] + [eef_x_pos] + [eef_xd_vel] + [finger_distance] + [gripper_force] + [goal])
