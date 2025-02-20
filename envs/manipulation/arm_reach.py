from brax import base
from brax.envs.base import State
from brax.io import mjcf
import mujoco
import jax
from jax import numpy as jnp
from envs.manipulation.arm_envs import ArmEnvs

"""
Reach: Move end of arm to random goal.
- Observation space: 13-dim obs + 3-dim goal.
- Action space:      4-dim, each element in [-1, 1], corresponding to target angles for joints 1, 2, 4, 6.

See _get_obs() and ArmEnvs._convert_action() for details.
"""
class ArmReach(ArmEnvs):
    def _get_xml_path(self):
        return "envs/assets/panda_reach.xml"
    
    @property
    def action_size(self) -> int:
        return 4 # Override default (actuator count)
    
    # See ArmEnvs._set_environment_attributes for descriptions of attributes
    def _set_environment_attributes(self):
        self.env_name = "arm_reach"
        self.episode_length = 50

        self.goal_indices = jnp.array([7, 8, 9]) # End-effector position
        self.completion_goal_indices = jnp.array([7, 8, 9]) # Identical
        self.state_dim = 13

        self.arm_noise_scale = 0
        self.goal_noise_scale = 0.2
        
    def _get_initial_state(self, rng):
        target_q = self.sys.init_q[:7]
        arm_q_default = jnp.array([1.571, 0.742, 0, -1.571, 0, 3.054, 1.449]) # Start closer to the relevant area
        arm_q = arm_q_default + self.arm_noise_scale * jax.random.uniform(rng, [self.sys.q_size() - 7], minval=-1)
        
        q = jnp.concatenate([target_q] + [arm_q])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd
        
    def _get_initial_goal(self, pipeline_state: base.State, rng):
        """
        Generate goals in a box. x: [-0.2, 0.2], y: [0.3, 0.7], z: [0.1, 0.5]
        """
        goal = jnp.array([0, 0.5, 0.3]) + self.goal_noise_scale * jax.random.uniform(rng, [3], minval=-1)
        return goal
        
    def _compute_goal_completion(self, obs, goal):
        # Goal occupancy: is the end of the arm close enough to the goal?
        eef_pos = obs[self.completion_goal_indices]
        goal_eef_pos = goal[:3]
        dist = jnp.linalg.norm(eef_pos - goal_eef_pos)

        success = jnp.array(dist < 0.1, dtype=float)
        success_easy = jnp.array(dist < 0.3, dtype=float)
        success_hard = jnp.array(dist < 0.03, dtype=float)
        
        return success, success_easy, success_hard
    
    def _update_goal_visualization(self, pipeline_state: base.State, goal: jax.Array) -> base.State:
        updated_q = pipeline_state.q.at[:3].set(goal) # Only set the position, not orientation
        updated_pipeline_state = pipeline_state.replace(qpos=updated_q)
        return updated_pipeline_state
        
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        """
        Observation space (13-dim)
         - q_subset (7-dim): joint angles
         - End of arm (6-dim): position and velocity
        Note q is 14-dim: 7-dim cube position/angle, 7-dim joint angles
         
        Goal space (3-dim): position of end of arm
        """
        
        q_subset = pipeline_state.q[7:14]
        eef_index = 7 # Cube is 0, then links 1-7 are indices 1-7. The end-effector (eef) base is merged with link 7, so we say link 7 index = eef index.
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]
        
        return jnp.concatenate([q_subset] + [eef_x_pos] + [eef_xd_vel] + [goal])
    
    def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
        q_indices = jnp.arange(7, 14)
        return pipeline_state.q[q_indices]