from brax import base
from brax.envs.base import State
from brax.io import mjcf
import mujoco
import jax
from jax import numpy as jnp
from envs.manipulation.arm_envs import ArmEnvs

"""
Grasp: Close fingers on opposite sides of a cube.
- Observation space: 23-dim obs + 7-dim goal.
- Action space:      5-dim, each element in [-1, 1], corresponding to target angles for joints 1, 2, 4, 6, and finger closedness.

See _get_obs() and ArmEnvs._convert_action() for details.
"""
class ArmGrasp(ArmEnvs):
    def __init__(self, cube_noise_scale=0.3, **kwargs):
        super().__init__(**kwargs)
        self.cube_noise_scale = cube_noise_scale
    
    def _get_xml_path(self):
        return "envs/assets/panda_grasp.xml"
    
    @property
    def action_size(self) -> int:
        return 5 # Override default (actuator count)
    
    # See ArmEnvs._set_environment_attributes for descriptions of attributes
    def _set_environment_attributes(self):
        self.env_name = "arm_grasp"
        self.episode_length = 50

        self.goal_indices = jnp.array([16, 17, 18, 19, 20, 21, 22]) # Left and right fingertip positions, and fingertip distance
        self.completion_goal_indices = jnp.array([16, 17, 18, 19, 20, 21, 22]) # Identical
        self.state_dim = 23

        self.arm_noise_scale = 0
        # self.cube_noise_scale = 0.3
        
    def _get_initial_state(self, rng):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        cube_q_xy = self.sys.init_q[:2] + self.cube_noise_scale * jax.random.uniform(subkey1, [2], minval=-1)
        cube_q_remaining = self.sys.init_q[2:7]
        target_q = self.sys.init_q[7:14]
        arm_q_default = jnp.array([1.571, 0.742, 0, -1.571, 0, 3.054, 1.449, 0.04, 0.04, 0, 0]) # Start closer to the relevant area
        arm_q = arm_q_default + self.arm_noise_scale * jax.random.uniform(subkey2, [self.sys.q_size() - 14], minval=-1)
        
        q = jnp.concatenate([cube_q_xy] + [cube_q_remaining] + [target_q] + [arm_q])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd
        
    def _get_initial_goal(self, pipeline_state: base.State, rng):
        # The fingertip goal positions will constantly adjust to be next to the cube (logic handled in arm_envs:step)
        cube_pos = pipeline_state.q[:3]
        left_fingertip_goal_pos = cube_pos + jnp.array([0.0375, 0, 0])
        right_fingertip_goal_pos = cube_pos + jnp.array([-0.0375, 0, 0])
        gripper_openness_goal = jnp.array([0.075]) # The cube itself is 0.06 wide, but we want the centers of the fingertips to be slightly farther apart
        
        goal = jnp.concatenate([left_fingertip_goal_pos] + [right_fingertip_goal_pos] + [gripper_openness_goal])
        return goal
        
    def _compute_goal_completion(self, obs, goal):
        # Goal occupancy: is the midpoint of the fingertips close enough to the cube, and is the gripper closed enough?
        # Technically, only success_hard is properly gripping the cube, but success/success_easy are for signs of life.
        cube_pos = obs[:3]
        left_fingertip_pos = obs[16:19]
        right_fingertip_pos = obs[19:22]
        fingertip_midpoint = (left_fingertip_pos + right_fingertip_pos) / 2
        cube_to_fingertip_midpoint_dist = jnp.linalg.norm(cube_pos - fingertip_midpoint)

        gripper_openness = obs[22]
        goal_gripper_openness = goal[9]
        gripper_openness_difference = jnp.linalg.norm(gripper_openness - goal_gripper_openness)

        success = jnp.array(
            jnp.all(jnp.array([
                cube_to_fingertip_midpoint_dist < 0.05,
                gripper_openness_difference < 0.02
            ])), 
            dtype=float
        )
        success_easy = jnp.array(
            jnp.all(jnp.array([
                cube_to_fingertip_midpoint_dist < 0.15,
                gripper_openness_difference < 0.05
            ])), 
            dtype=float
        )
        success_hard = jnp.array(
            jnp.all(jnp.array([
                cube_to_fingertip_midpoint_dist < 0.02,
                gripper_openness_difference < 0.005
            ])), 
            dtype=float
        )
        return success, success_easy, success_hard
    

    def _update_goal_visualization(self, pipeline_state: base.State, goal: jax.Array) -> base.State:
        updated_q = pipeline_state.q.at[7:10].set(goal[:3]) # Only set the position, not orientation (we set the left marker and the right marker is automatically offset)
        updated_pipeline_state = pipeline_state.replace(qpos=updated_q)
        return updated_pipeline_state
        
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        """
        Observation space (23-dim)
         - q_subset (10-dim): 3-dim cube position, 7-dim joint angles
         - End-effector (6-dim): position and velocity
         - Fingertips (7-dim): 6-dim positions of fingertips, 1-dim distance between fingertips
        Note q is 25-dim: 7-dim cube position/angle, 7-dim goal marker position/angle, 7-dim joint angles, 4-dim finger offsets and dummy fingertip angles
         
        Goal space (7-dim): position of left fingertip, position of right fingertip, distance between fingertips
        """
        q_indices = jnp.array([0, 1, 2, 14, 15, 16, 17, 18, 19, 20])
        q_subset = pipeline_state.q[q_indices]
        
        eef_index = 8 # Cube is 0, goal marker is 1, then links 1-7 are indices 2-8. The end-effector (eef) base is merged with link 7, so we say link 7 index = eef index.
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]

        left_fingertip_index = 10 # Left finger is 9, fingertip is 10
        left_fingertip_x_pos = pipeline_state.x.pos[left_fingertip_index]
        right_fingertip_index = 12 # Right finger is 11, fingertip is 12
        right_fingertip_x_pos = pipeline_state.x.pos[right_fingertip_index]
        fingertip_distance = jnp.linalg.norm(right_fingertip_x_pos - left_fingertip_x_pos)[None] # [None] expands dims from 0 to 1
        
        return jnp.concatenate([q_subset] + [eef_x_pos] + [eef_xd_vel] + [left_fingertip_x_pos] + [right_fingertip_x_pos] + [fingertip_distance] + [goal])
    
    def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
        q_indices = jnp.arange(14, 21)
        return pipeline_state.q[q_indices]