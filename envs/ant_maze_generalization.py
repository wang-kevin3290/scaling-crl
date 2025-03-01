# Same as ant maze but add extra parameters, and reset + make_maze function is changed to enable different goal/start pairs
# We remove the _eval mazes, since they are duplicates (those only used for overall structure now)
import os
from typing import Tuple

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco
import xml.etree.ElementTree as ET

# This is based on original Ant environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/ant.py
# Maze creation dapted from: https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/locomotion/maze_env.py

RESET = R = 'r'
GOAL = G = 'g'


U_MAZE = [[1, 1, 1, 1, 1],
          [1, R, 0, 0, 1],
          [1, 1, 1, 0, 1],
          [1, G, 0, 0, 1],
          [1, 1, 1, 1, 1]]

U2_MAZE = [[1, 1, 1, 1, 1, 1],
           [1, R, 0, 0, 0, 1],
           [1, 1, 1, 1, 0, 1],
           [1, G, 0, 0, 0, 1],
           [1, 1, 1, 1, 1, 1]]

U3_MAZE = [[1, 1, 1, 1, 1, 1, 1],
           [1, R, 0, 0, 0, 0, 1],
           [1, 1, 1, 1, 1, 0, 1],
           [1, G, 0, 0, 0, 0, 1],
           [1, 1, 1, 1, 1, 1, 1]]

U4_MAZE = [[1, 1, 1, 1, 1],
           [1, 0, 0, 0, 1],
           [1, R, 1, 0, 1],
           [1, 1, 1, 0, 1],
           [1, G, 1, 0, 1],
           [1, 0, 0, 0, 1],
           [1, 1, 1, 1, 1]]

U5_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
           [1, 0, 0, 0, 0, 0, 0, 1],
           [1, R, 1, 1, 1, 1, 0, 1],
           [1, 1, 1, 1, 1, 1, 0, 1],
           [1, G, 1, 1, 1, 1, 0, 1],
           [1, 0, 0, 0, 0, 0, 0, 1],
           [1, 1, 1, 1, 1, 1, 1, 1]]

#CURRENTLY BIG_MAZE, HARDEST_MAZE CANNOT BE USED FOR GENERALIZATION
BIG_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, G, 1, 1, G, G, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, 1, G, G, G, 1, 1, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, G, 1, G, G, 1, G, 1],
            [1, G, G, G, 1, G, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, G, G, G, 1, G, G, G, G, G, 1],
                [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
                [1, G, G, G, G, G, G, 1, G, G, G, 1],
                [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
                [1, G, G, 1, G, 1, G, G, G, G, G, 1],
                [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
                [1, G, G, 1, G, G, G, 1, G, G, G, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

MAZE_HEIGHT = 0.5

#dfs from R to G to get the list of coordinates in the forward direction
def get_forward_path(maze_layout):
    start, end = None, None
    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            if maze_layout[i][j] == RESET:
                start = (i, j)
            elif maze_layout[i][j] == GOAL:
                end = (i, j)
    return dfs(maze_layout, start, end)

def dfs(maze_layout, start, end):
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    
    prev_x, prev_y = None, None
    curr_x, curr_y = start
    
    path = []
    
    while not (curr_x == end[0] and curr_y == end[1]):
        path.append((curr_x, curr_y))
        for direction in range(4):
            next_x, next_y = curr_x + dx[direction], curr_y + dy[direction]
            assert not (next_x < 0 or next_x >= len(maze_layout) or next_y < 0 or next_y >= len(maze_layout[0])) #should be fully surrounded by walls
            if maze_layout[next_x][next_y] == 1:
                continue
            if next_x == prev_x and next_y == prev_y:
                continue
            prev_x, prev_y = curr_x, curr_y
            curr_x, curr_y = next_x, next_y
            break
    
    path.append(end)
    return path


def get_start_goal(maze_layout, generalization_config, rng):
    sg_pairs = [] #valid start goal pairs
    forward_path = get_forward_path(maze_layout)
    num_valid_pairs = sum([len(forward_path) - i for i in range(1, 6) if f"{i}f" in generalization_config])
    num_distances = len(generalization_config.split("f")[:-1])
    weights = []
    
    for config in generalization_config.split("f")[:-1]: #gets rid of empty string at the end
        config = int(config)
        pairs = []
        for i in range(len(forward_path) - config):
            pairs.append((forward_path[i], forward_path[i + config]))
            weight = num_valid_pairs / num_distances / (len(forward_path) - config)
            weights.append(weight)
        sg_pairs.extend(pairs)
    
    print(f"num_valid_pairs: {num_valid_pairs}, sg_pairs: {sg_pairs}, weights: {weights}", flush=True)

    sg_pairs = jp.array(sg_pairs)
    weights = jp.array(weights)
    # idx = jax.random.randint(rng, (1,), 0, len(sg_pairs))
    idx = jax.random.choice(rng, len(sg_pairs), p=weights)
    random_pair = jp.array(sg_pairs[idx])
    
    return random_pair

def get_maze_layout(maze_layout_name):
    if maze_layout_name == "u_maze":
        maze_layout = U_MAZE
    elif maze_layout_name == "u2_maze":
        maze_layout = U2_MAZE
    elif maze_layout_name == "u3_maze":
        maze_layout = U3_MAZE
    elif maze_layout_name == "u4_maze":
        maze_layout = U4_MAZE
    elif maze_layout_name == "u5_maze":
        maze_layout = U5_MAZE

    elif maze_layout_name == "big_maze":
        maze_layout = BIG_MAZE
    elif maze_layout_name == "hardest_maze":
        maze_layout = HARDEST_MAZE
    else:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")

    return maze_layout

# Create a xml with maze with the starting position
def make_maze(maze_layout, maze_size_scaling):    
    xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', "ant_maze.xml")

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            struct = maze_layout[i][j]
            if struct == 1:
                ET.SubElement(
                    worldbody, "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f" % (i * maze_size_scaling,
                                    j * maze_size_scaling,
                                    MAZE_HEIGHT / 2 * maze_size_scaling),
                    size="%f %f %f" % (0.5 * maze_size_scaling,
                                        0.5 * maze_size_scaling,
                                        MAZE_HEIGHT / 2 * maze_size_scaling),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    
    torso = tree.find(".//numeric[@name='init_qpos']")
    data = torso.get("data")
    torso.set("data", f"{0} {0} " + data) 

    tree = tree.getroot()
    xml_string = ET.tostring(tree)
    
    return xml_string

class AntMazeGeneralization(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        backend="generalized",
        maze_layout_name="u_maze",
        maze_size_scaling=4.0,
        generalization_config="1f", #1f means one forward
        **kwargs,
    ):
        self.maze_layout = get_maze_layout(maze_layout_name)
        self.maze_size_scaling = maze_size_scaling
        self.generalization_config = generalization_config
        # start, goal = get_start_goal(maze_layout, generalization_config)
        
        xml_string = make_maze(self.maze_layout, self.maze_size_scaling)
        sys = mjcf.loads(xml_string)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.replace(dt=0.005)
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        if backend == "positional":
            # TODO: does the same actuator strength work as in spring
            sys = sys.replace(
                actuator=sys.actuator.replace(
                    gear=200 * jp.ones_like(sys.actuator.gear)
                )
            )

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        start, goal = get_start_goal(self.maze_layout, self.generalization_config, rng3)
        
        print(f"start: {start}, goal: {goal}", flush=True)
        
        start_pos = jp.array([start[0] * self.maze_size_scaling, start[1] * self.maze_size_scaling])
        goal_pos = jp.array([goal[0] * self.maze_size_scaling, goal[1] * self.maze_size_scaling])


        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # set the target q, qd
        q = q.at[:2].set(start_pos)
        q = q.at[-2:].set(goal_pos)
        qd = qd.at[-2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero
        }
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state

    # Todo rename seed to traj_id
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        contact_cost = 0.0

        obs = self._get_obs(pipeline_state)
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        dist = jp.linalg.norm(obs[:2] - obs[-2:])
        success = jp.array(dist < 0.5, dtype=float)
        success_easy = jp.array(dist < 2., dtype=float)
        reward = -dist + healthy_reward - ctrl_cost - contact_cost
        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy
        )
        state.info.update(info)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]

        target_pos = pipeline_state.x.pos[-1][:2]

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        return jp.concatenate([qpos] + [qvel] + [target_pos])