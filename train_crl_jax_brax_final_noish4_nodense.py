import os
import jax
import flax
import tyro
import time
import optax
import wandb
import pickle
import random
import wandb_osh
import numpy as np
import flax.linen as nn
import jax.numpy as jnp

from brax import envs
from etils import epath
from dataclasses import dataclass
from collections import namedtuple
from typing import NamedTuple, Any
from wandb_osh.hooks import TriggerWandbSyncHook
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling
from brax.io import html

from evaluator import CrlEvaluator
from buffer import TrajectoryUniformSamplingQueue
from memory_bank import MemoryBank, MemoryBankState

@dataclass
class Args:
    exp_name: str = "train" # os.path.basename(__file__)[: -len(".py")]
    seed: int = random.randint(1, 1000) # 16
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "clean_JaxGCRL_test"
    wandb_entity: str = 'wang-kevin3290-princeton-university'
    wandb_mode: str = 'offline'
    wandb_dir: str = '.'
    wandb_group: str = '.'
    capture_vis: bool = True
    vis_length: int = 1000
    checkpoint: bool = True

    #environment specific arguments
    env_id: str = "humanoid" # "ant_push" "ant_hardest_maze" "ant_big_maze" "humanoid" "ant"
    episode_length: int = 1000
    # to be filled in runtime
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0

    # Algorithm specific arguments
    total_env_steps: int = 100000000 # 50000000
    num_epochs: int = 100 # 50
    num_envs: int = 512
    eval_env_id: str = ""
    num_eval_envs: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    
    #adding in a batch_size_multiplier argument for critic vs. actor batch size
    critic_batch_size_multiplier: float = 1.0 #this has to be less than or equal to 1
    actor_batch_size_multiplier: float = 1.0 #this has to be less than 1

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    
    unroll_length: int  = 62
    
    # ADDING IN A NETWORK WIDTH ARGUMENT
    same_network_width: int = 0
    network_width: int = 256
    critic_network_width: int = 256
    actor_network_width: int = 256
    actor_depth: int = 4
    critic_depth: int = 4
    actor_skip_connections: int = 0 # 0 for no skip connections, >= 0 means the frequency of skip connections (every X layers)
    critic_skip_connections: int = 0 # 0 for no skip connections, >= 0 means the frequency of skip connections (every X layers)
    
    num_episodes_per_env: int = 1 #the number of episodes to sample from each env when sampling data 
    #(to ensure number of batches is consistent as increase batch_size; for now, just a bandaid fix)
    # should be something like batch_size / 256
    training_steps_multiplier: int = 1 #should have the same effect as num_episodes_per_env, hmmm
    use_all_batches: int = 0 # if 1, use all batches; if 0, use a random subset of batches
    num_sgd_batches_per_training_step: int = 800 # this parameter so as to hold the number of batches constant (no matter batch_size, etc)
    
    mrn: int = 0
    memory_bank: int = 0
    memory_bank_size: int = batch_size # this can be modified too
    
    batchdiv2: int = 0 
    # if 1, freeze gradients for second half of batch
    # if 2, split in half along sa and freeze second half of g (Eysenbach ablation, remember it's forward loss)
    #
    # use batch_size * 2 and split in half and freeze gradients and all that (Eysenbach ablation), does not 
    # TODO: if 2, modifies actor such that it uses the half batch size (isolate for critic ablation)
    # can add 3, 4, etc (if diff between 1 and 2, maybe for batch_size ablation we need to have separate for actor and critic)
    # add more for instead of discarding second half, just freeze gradients for second half so symmetric with first
    
    eval_actor: int = 0
    # if 0, use deterministic actor for evaluation
    # if 1, use stochastic actor for evaluation
    # if 2, sample two actions and take the one with the higher Q value
    # if K >= 2, sample K actions and take the one with the highest Q value
    expl_actor: int = 1
    # if 0, use deterministic actor for exploration/collecting data
    # if 1, use stochastic actor for exploration/collecting data
    # if 2, sample two actions and take the one with the higher Q value
    # if K >= 2, sample K actions and take the one with the highest Q value
    
    entropy_param: float = 0.5
    disable_entropy: int = 0
    
    use_relu: int = 0
    
    resnet: str = "noishmistake4_nodense"
    
    num_render: int = 10
    
    save_buffer: int = 0
    
    
    
    # to be filled in runtime
    env_steps_per_actor_step : int = 0
    """number of env steps per actor step (computed in runtime)"""
    num_prefill_env_steps : int = 0
    """number of env steps to fill the buffer before starting training (computed in runtime)"""
    num_prefill_actor_steps : int = 0
    """number of actor steps to fill the buffer before starting training (computed in runtime)"""
    num_training_steps_per_epoch : int = 0
    """the number of training steps per epoch(computed in runtime)"""

lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
bias_init = nn.initializers.zeros
def residual_block(x, width, normalize, activation):
    identity = x
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = x + identity
    return x

class SA_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        
        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish
            
        x = jnp.concatenate([s, a], axis=-1)
        #Initial layer
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        #Residual blocks
        for i in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)
        #Final layer
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x
    
class G_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    @nn.compact
    def __call__(self, g: jnp.ndarray):

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        
        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish
        
        x = g
        #Initial layer
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        #Residual blocks
        for i in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)
        #Final layer
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class Sym(nn.Module):
    dim_hidden: int = 176  # First hidden layer dimension, 176 based off the paper
    dim_embed: int = 64    # Final output size

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.dim_hidden)(x)  # First hidden layer (176 units)
        x = nn.relu(x)  # ReLU activation
        x = nn.Dense(self.dim_embed)(x)  # Final embedding layer (64 units)
        return x

class Asym(nn.Module):
    dim_hidden: int = 176  # First hidden layer dimension
    dim_embed: int = 64    # Final output size

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.dim_hidden)(x)  # First hidden layer (176 units)
        x = nn.relu(x)  # ReLU activation
        x = nn.Dense(self.dim_embed)(x)  # Final embedding layer (64 units)
        return x
  
class Actor(nn.Module):
    action_size: int
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0 # 0 for no skip connections, >= 0 means the frequency of skip connections (every X layers)
    use_relu: int = 0
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
            
        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        print(f"x.shape: {x.shape}", flush=True)

        #Initial layer
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        #Residual blocks
        for i in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)
        #Final layer
        # x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)

        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner"""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState
    memory_bank_state: MemoryBankState

class Transition(NamedTuple):
    """Container for a transition"""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()

def load_params(path: str):
    with epath.Path(path).open('rb') as fin:
        buf = fin.read()
    return pickle.loads(buf)

def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open('wb') as fout:
        fout.write(pickle.dumps(params))

if __name__ == "__main__":

    args = tyro.cli(Args)
    
    # Print every arg
    print("Arguments:", flush=True)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}", flush=True)
    print("\n", flush=True)

    args.env_steps_per_actor_step = args.num_envs * args.unroll_length
    print(f"env_steps_per_actor_step: {args.env_steps_per_actor_step}", flush=True)

    args.num_prefill_env_steps = args.min_replay_size * args.num_envs
    print(f"num_prefill_env_steps: {args.num_prefill_env_steps}", flush=True)

    args.num_prefill_actor_steps = np.ceil(args.min_replay_size / args.unroll_length)
    print(f"num_prefill_actor_steps: {args.num_prefill_actor_steps}", flush=True)

    args.num_training_steps_per_epoch = (args.total_env_steps - args.num_prefill_env_steps) // (args.num_epochs * args.env_steps_per_actor_step)
    print(f"num_training_steps_per_epoch: {args.num_training_steps_per_epoch}", flush=True)

    if args.same_network_width:
        args.critic_network_width = args.network_width
        args.actor_network_width = args.network_width
    
    run_name = f"{args.env_id}{'_' + args.eval_env_id if args.eval_env_id else ''}_{args.batch_size}_critbx:{args.critic_batch_size_multiplier}_actbx:{args.actor_batch_size_multiplier}_batchdiv2:{args.batchdiv2}_{args.total_env_steps}_nenvs:{args.num_envs}_criticwidth:{args.critic_network_width}_actorwidth:{args.actor_network_width}_criticdepth:{args.critic_depth}_actordepth:{args.actor_depth}_actorskip:{args.actor_skip_connections}_criticskip:{args.critic_skip_connections}_epspenv:{args.num_episodes_per_env}_trainmult:{args.training_steps_multiplier}_mrn:{args.mrn}_memorybank:{args.memory_bank}_sgdbatchesptrainstep:{args.num_sgd_batches_per_training_step}_useallbatches:{args.use_all_batches}_eplen:{args.episode_length}_maxbuffersize:{args.max_replay_size}_evalactor:{args.eval_actor}_explactor:{args.expl_actor}_vislen:{args.vis_length}_critlr:{args.critic_lr}_actlr:{args.actor_lr}_alplr:{args.alpha_lr}_entropy:{args.entropy_param}_disable_entropy:{args.disable_entropy}_relu:{args.use_relu}_resnet:{args.resnet}_logsumexppenalty:{args.logsumexp_penalty_coeff}_{args.seed}"
    print(f"run_name: {run_name}", flush=True)
    
    if args.track:

        if args.wandb_group ==  '.':
            args.wandb_group = None
            
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            group=args.wandb_group,
            dir=args.wandb_dir,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

        if args.wandb_mode == 'offline':
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()
        
    if args.checkpoint:
        from pathlib import Path
        from datetime import datetime
        short_run_name = f"runs/{args.env_id}_{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        save_path = Path(args.wandb_dir) / Path(short_run_name)
        os.mkdir(path=save_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key, sym_key, asym_key, memory_bank_key = jax.random.split(key, 10)


    def make_env(env_id=args.env_id):
        print(f"making env with env_id: {env_id}", flush=True)
        if env_id == "reacher":
            from envs.reacher import Reacher
            env = Reacher(
                backend="spring",
            )
            args.obs_dim = 10
            args.goal_start_idx = 4
            args.goal_end_idx = 7
        elif env_id == "pusher":
            from envs.pusher import Pusher
            env = Pusher(
                backend="spring",
            )
            args.obs_dim = 20
            args.goal_start_idx = 10
            args.goal_end_idx = 13
        elif env_id == "ant":
            from envs.ant import Ant
            env = Ant(
                backend="spring",
                exclude_current_positions_from_observation=False,
                terminate_when_unhealthy=True,
            )

            args.obs_dim = 29
            args.goal_start_idx = 0
            args.goal_end_idx = 2

        elif "ant" in env_id and "maze" in env_id: #needed the add the ant check to differentiate with humanoid maze
            if "gen" not in env_id:
                from envs.ant_maze import AntMaze
                env = AntMaze(
                    backend="spring",
                    exclude_current_positions_from_observation=False,
                    terminate_when_unhealthy=True,
                    maze_layout_name=env_id[4:]
                )

                args.obs_dim = 29
                args.goal_start_idx = 0
                args.goal_end_idx = 2
            else:
                from envs.ant_maze_generalization import AntMazeGeneralization
                gen_idx = env_id.find("gen")
                maze_layout_name = env_id[4:gen_idx-1]
                generalization_config = env_id[gen_idx+4:]
                print(f"maze_layout_name: {maze_layout_name}, generalization_config: {generalization_config}", flush=True)
                env = AntMazeGeneralization(
                    backend="spring",
                    exclude_current_positions_from_observation=False,
                    terminate_when_unhealthy=True,
                    maze_layout_name=maze_layout_name,
                    generalization_config=generalization_config
                )

                args.obs_dim = 29
                args.goal_start_idx = 0
                args.goal_end_idx = 2
        
        elif env_id == "ant_ball":
            from envs.ant_ball import AntBall
            env = AntBall(
                backend="spring",
                exclude_current_positions_from_observation=False,
                terminate_when_unhealthy=True,
            )

            args.obs_dim = 31
            args.goal_start_idx = 28
            args.goal_end_idx = 30

        elif env_id == "ant_push":
            from envs.ant_push import AntPush
            env = AntPush(
                backend="mjx",
            )

            args.obs_dim = 31
            args.goal_start_idx = 0
            args.goal_end_idx = 2
            
        elif env_id == "humanoid":
            from envs.humanoid import Humanoid
            env = Humanoid(
                backend="spring",
                exclude_current_positions_from_observation=False,
                terminate_when_unhealthy=True,
            )

            args.obs_dim = 268
            args.goal_start_idx = 0
            args.goal_end_idx = 3
            
        elif "humanoid" in env_id and "maze" in env_id:
            from envs.humanoid_maze import HumanoidMaze
            env = HumanoidMaze(
                backend="spring",
                maze_layout_name=env_id[9:]
            )

            args.obs_dim = 268
            args.goal_start_idx = 0
            args.goal_end_idx = 3

            
        elif env_id == "arm_reach":
            from envs.manipulation.arm_reach import ArmReach
            env = ArmReach(
                backend="mjx",
            )

            args.obs_dim = 13
            args.goal_start_idx = 7
            args.goal_end_idx = 10
            
        elif env_id == "arm_binpick_easy":
            from envs.manipulation.arm_binpick_easy import ArmBinpickEasy
            env = ArmBinpickEasy(
                backend="mjx",
            )

            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3
            
        elif env_id == "arm_binpick_hard":
            from envs.manipulation.arm_binpick_hard import ArmBinpickHard
            env = ArmBinpickHard(
                backend="mjx",
            )

            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3
            
        elif env_id == "arm_binpick_easy_EEF":
            from envs.manipulation.arm_binpick_easy_EEF import ArmBinpickEasyEEF
            env = ArmBinpickEasyEEF(
                backend="mjx",
            )

            args.obs_dim = 11
            args.goal_start_idx = 0
            args.goal_end_idx = 3
        
        elif "arm_grasp" in env_id: # either arm_grasp or arm_grasp_0.5, etc
            from envs.manipulation.arm_grasp import ArmGrasp
            cube_noise_scale = float(env_id[10:]) if len(env_id) > 9 else 0.3
            env = ArmGrasp(
                cube_noise_scale=cube_noise_scale,
                backend="mjx",
            )

            args.obs_dim = 23
            args.goal_start_idx = 16
            args.goal_end_idx = 23
        
        elif env_id == "arm_push_easy":
            from envs.manipulation.arm_push_easy import ArmPushEasy
            env = ArmPushEasy(
                backend="mjx",
            )

            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3
        
        elif env_id == "arm_push_hard":
            from envs.manipulation.arm_push_hard import ArmPushHard
            env = ArmPushHard(
                backend="mjx",
            )

            args.obs_dim = 17
            args.goal_start_idx = 0
            args.goal_end_idx = 3

        else:
            raise NotImplementedError
        
        return env
        
    env = make_env()
    env = envs.training.wrap(
        env,
        episode_length=args.episode_length,
    )

    obs_size = env.observation_size
    action_size = env.action_size
    env_keys = jax.random.split(env_key, args.num_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step = jax.jit(env.step)
    
    print(f"obs_size: {obs_size}, action_size: {action_size}", flush=True)
    
    
    if not args.eval_env_id:
        args.eval_env_id = args.env_id
        
    # make eval env
    eval_env = make_env(args.eval_env_id)
    eval_env = envs.training.wrap(
        eval_env,
        episode_length=args.episode_length,
    )
    eval_env_keys = jax.random.split(eval_env_key, args.num_envs)
    eval_env_state = jax.jit(eval_env.reset)(eval_env_keys)
    eval_env.step = jax.jit(eval_env.step)
        
    

    # Network setup
    # Actor
    actor = Actor(action_size=action_size, network_width=args.actor_network_width, network_depth=args.actor_depth, skip_connections=args.actor_skip_connections, use_relu=args.use_relu)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, obs_size])),
        tx=optax.adam(learning_rate=args.actor_lr)
    )

    # Critic
    sa_encoder = SA_encoder(network_width=args.critic_network_width, network_depth=args.critic_depth, skip_connections=args.critic_skip_connections, use_relu=args.use_relu)
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, args.obs_dim]), np.ones([1, action_size]))
    g_encoder = G_encoder(network_width=args.critic_network_width, network_depth=args.critic_depth, skip_connections=args.critic_skip_connections, use_relu=args.use_relu)
    g_encoder_params = g_encoder.init(g_key, np.ones([1, args.goal_end_idx - args.goal_start_idx]))
    # c = jnp.asarray(0.0, dtype=jnp.float32) (NOT USED IN CODE, WHATS THIS)
    
    sym = Sym()
    sym_params = sym.init(sym_key, np.ones([1, 64]))
    asym = Asym()
    asym_params = asym.init(asym_key, np.ones([1, 64]))
    
    
    if not args.mrn:
        critic_state = TrainState.create(
            apply_fn=None,
            params={
                "sa_encoder": sa_encoder_params, 
                "g_encoder": g_encoder_params
                },
            tx=optax.adam(learning_rate=args.critic_lr),
        )
    else:
        critic_state = TrainState.create(
            apply_fn=None,
            params={
                "sa_encoder": sa_encoder_params, 
                "g_encoder": g_encoder_params,
                "sym": sym_params,
                "asym": asym_params},
            tx=optax.adam(learning_rate=args.critic_lr),
        )

    # Entropy coefficient
    target_entropy = -args.entropy_param * action_size # action_size = 8 for ant, 17 for humanoid, etc # USEED TO BE -0.5 * action_size
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": log_alpha},
        tx=optax.adam(learning_rate=args.alpha_lr),
    )
    
    def jit_wrap(memory_bank):
        memory_bank.insert = jax.jit(memory_bank.insert)
        memory_bank.sample = jax.jit(memory_bank.sample)
        return memory_bank
    
    if args.memory_bank:
        memory_bank = jit_wrap(MemoryBank(memory_bank_size=args.memory_bank_size, feature_dim=64, batch_size=args.batch_size))
        memory_bank_state = jax.jit(memory_bank.init)(memory_bank_key)
    else:
        memory_bank_state = None
    
    # Trainstate
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
        memory_bank_state=memory_bank_state,
    )

    #Replay Buffer
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))

    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        extras={
            "state_extras": {
                "truncation": 0.0,
                "seed": 0.0,
            }
        },
    )

    def jit_wrap(buffer):
        buffer.insert_internal = jax.jit(buffer.insert_internal)
        buffer.sample_internal = jax.jit(buffer.sample_internal)
        return buffer
    
    replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=args.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=args.batch_size,
                num_envs=args.num_envs,
                episode_length=args.episode_length,
            )
        )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
        actions = nn.tanh( means )

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1-nstate.done,
            extras={"state_extras": state_extras},
        )
    
    def actor_step(training_state, env, env_state, key, extra_fields):
        means, log_stds = actor.apply(training_state.actor_state.params, env_state.obs)
        stds = jnp.exp(log_stds)
        actions = nn.tanh( means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype) )

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1-nstate.done,
            extras={"state_extras": state_extras},
        )
        
    def multi_sample_actor_step(training_state, env, env_state, key, K, extra_fields):
        # Get K sets of actions from the actor
        keys = jax.random.split(key, K)
        means, log_stds = actor.apply(training_state.actor_state.params, env_state.obs)
        stds = jnp.exp(log_stds)
        
        # Sample K actions
        actions = jnp.stack([
            nn.tanh(means + stds * jax.random.normal(k, shape=means.shape, dtype=means.dtype))
            for k in keys
        ])  # Shape: (K, batch_size, action_dim)
        
        # Compute Q values for each action
        state = env_state.obs[:, :args.obs_dim]
        goal = env_state.obs[:, args.obs_dim:]
        
        # Compute SA and G representations for each action
        sa_reprs = jax.vmap(
            lambda a: sa_encoder.apply(
                training_state.critic_state.params["sa_encoder"], 
                state, 
                a
            )
        )(actions)  # Shape: (K, batch_size, repr_dim)
        
        g_repr = g_encoder.apply(
            training_state.critic_state.params["g_encoder"], 
            goal
        )  # Shape: (batch_size, repr_dim)
        
        # Compute Q values as negative distances
        q_values = -jnp.sqrt(
            jnp.sum((sa_reprs - g_repr) ** 2, axis=-1)
        )  # Shape: (K, batch_size)
        
        # Select actions with highest Q values
        best_action_idx = jnp.argmax(q_values, axis=0)  # Shape: (batch_size,)
        best_actions = jnp.take_along_axis(
            actions,
            best_action_idx[None, :, None],
            axis=0
        )[0]  # Shape: (batch_size, action_dim)
        
        # Step environment with best actions
        nstate = env.step(env_state, best_actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        
        return nstate, Transition(
            observation=env_state.obs,
            action=best_actions,
            reward=nstate.reward,
            discount=1-nstate.done,
            extras={"state_extras": state_extras},
        )
    
    

    @jax.jit
    def get_experience(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t): #conducts a single actor step in environment
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            if args.expl_actor == 1:
                env_state, transition = actor_step(training_state, env, env_state, current_key, extra_fields=("truncation", "seed"))
            elif args.expl_actor == 0:
                env_state, transition = deterministic_actor_step(training_state, env, env_state, extra_fields=("truncation", "seed"))
            else:
                env_state, transition = multi_sample_actor_step(training_state, env, env_state, current_key, args.expl_actor, extra_fields=("truncation", "seed"))
            return (env_state, next_key), transition

        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=args.unroll_length)

        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(
                training_state,
                env_state,
                buffer_state,
                key,
            
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + args.env_steps_per_actor_step,
            )
            return (training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=args.num_prefill_actor_steps)[0]

    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        actor_batch_size = int(args.batch_size * args.actor_batch_size_multiplier)
        transitions = jax.tree_util.tree_map(
            lambda x: x[:actor_batch_size], 
            transitions
        )
        def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
            obs = transitions.observation           # expected_shape = batch_size, obs_size + goal_size
            state = obs[:, :args.obs_dim]
            future_state = transitions.extras["future_state"]
            goal = future_state[:, args.goal_start_idx : args.goal_end_idx]
            observation = jnp.concatenate([state, goal], axis=1)

            means, log_stds = actor.apply(actor_params, observation)
            stds = jnp.exp(log_stds)
            x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            action = nn.tanh(x_ts)
            log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
            log_prob = log_prob.sum(-1)           # dimension = B

            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
            sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
            g_repr = g_encoder.apply(g_encoder_params, goal)

            qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

            if args.disable_entropy:
                actor_loss = -jnp.mean(qf_pi)
            else:
                actor_loss = jnp.mean( jnp.exp(log_alpha) * log_prob - (qf_pi) )

            return actor_loss, log_prob

        def alpha_loss(alpha_params, log_prob):
            alpha = jnp.exp(alpha_params["log_alpha"])
            alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - target_entropy))
            return jnp.mean(alpha_loss)
        
        (actorloss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(training_state.actor_state.params, training_state.critic_state.params, training_state.alpha_state.params['log_alpha'], transitions, key)
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        alphaloss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
        new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

        training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss": actorloss,
            "alph_aloss": alphaloss,   
            "log_alpha": training_state.alpha_state.params["log_alpha"],
        }

        return training_state, metrics

    @jax.jit
    def update_critic(transitions, training_state, key):
        critic_batch_size = int(args.batch_size * args.critic_batch_size_multiplier)
        transitions = jax.tree_util.tree_map(
            lambda x: x[:critic_batch_size], 
            transitions
        )
        def critic_loss(critic_params, transitions, key):
            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
            
            obs = transitions.observation[:, :args.obs_dim]
            action = transitions.action
            
            sa_repr = sa_encoder.apply(sa_encoder_params, obs, action)
            g_repr = g_encoder.apply(g_encoder_params, transitions.observation[:, args.obs_dim:])
                
            if args.memory_bank:
                new_memory_bank_state, (sa_bank, g_bank) = memory_bank.sample(training_state.memory_bank_state) #currently just sampling another batch_size, can modify in memory_bank.py later
                sa_repr = jnp.concatenate([sa_repr, sa_bank], axis=0)
                g_repr = jnp.concatenate([g_repr, g_bank], axis=0)
                new_memory_bank_state = memory_bank.insert(new_memory_bank_state, sa_repr[:args.batch_size], g_repr[:args.batch_size])
            else:
                new_memory_bank_state = training_state.memory_bank_state
                
            if args.batchdiv2 == 1:
                sa_repr = jnp.concatenate([sa_repr[:args.batch_size//2], jax.lax.stop_gradient(sa_repr[args.batch_size//2:])])
                g_repr = jnp.concatenate([g_repr[:args.batch_size//2], jax.lax.stop_gradient(g_repr[args.batch_size//2:])])
            elif args.batchdiv2 == 2:
                sa_repr = sa_repr[:args.batch_size//2]
                g_repr = jnp.concatenate([g_repr[:args.batch_size//2], jax.lax.stop_gradient(g_repr[args.batch_size//2:])])
                
            if args.mrn:
                sym1 = sym.apply(critic_params['sym'], sa_repr)                    # (B, 64)
                sym2 = sym.apply(critic_params['sym'], g_repr)                     # (B, 64)
                dist_s = jnp.sum((sym1[:, None, :] - sym2[None, :, :]) ** 2, axis=-1) + 1e-6 # (B, B)

                # Asymmetric path
                asym1 = asym.apply(critic_params['asym'], sa_repr)                # (B, 64)
                asym2 = asym.apply(critic_params['asym'], g_repr)                 # (B, 64)
                res = jax.nn.relu(asym1[:, None, :] - asym2[None, :, :])         # (B, B, 64)
                dist_a = jnp.max(res, axis=-1) + 1e-6                              # (B, B)

                # Combining distances
                logits = -(dist_s + dist_a)                                       # (B, B)
                critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))  # scalar
                
                # logsumexp regularisation
                logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
                critic_loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

            else:
                # InfoNCE
                logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))       # shape = BxB
                critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

                # logsumexp regularisation
                logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
                critic_loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

            if 0:
                I = jnp.eye(logits.shape[0])
                correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
                logits_pos = jnp.sum(logits * I) / jnp.sum(I)
                logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
            else:
                I, correct, logits_pos, logits_neg = jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), jnp.zeros(1)
                

            return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg, new_memory_bank_state)
            
        (loss, (logsumexp, I, correct, logits_pos, logits_neg, new_memory_bank_state)), grad = jax.value_and_grad(critic_loss, has_aux=True)(training_state.critic_state.params, transitions, key)
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state = new_critic_state, memory_bank_state=new_memory_bank_state)

        metrics = {
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logsumexp": logsumexp.mean(),
            "critic_loss": loss,
        }

        return training_state, metrics
    
    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key, = jax.random.split(key, 3)

        training_state, actor_metrics = update_actor_and_alpha(transitions, training_state, actor_key)

        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)

        training_state = training_state.replace(gradient_steps = training_state.gradient_steps + 1)

        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        
        return (training_state, key,), metrics

    @jax.jit
    def training_step(training_state, env_state, buffer_state, key, t):
        experience_key1, experience_key2, sampling_key, training_key, sgd_batches_key = jax.random.split(key, 5)

        # print(f"Current training step: {t}")
        # if t % args.training_steps_multiplier == 0:
        
        # update buffer
        env_state, buffer_state = get_experience(
            training_state,
            env_state,
            buffer_state,
            experience_key1,
        )

        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.env_steps_per_actor_step,
        )
            
        # def collect_data():
        #     new_env_state, new_buffer_state = get_experience(
        #         training_state.actor_state,
        #         env_state,
        #         buffer_state,
        #         experience_key1,
        #     )
        #     new_training_state = training_state.replace(
        #         env_steps=training_state.env_steps + args.env_steps_per_actor_step
        #     )
        #     return new_training_state, new_env_state, new_buffer_state

        # def skip_data_collection():
        #     return training_state, env_state, buffer_state

        # training_state, env_state, buffer_state = jax.lax.cond(
        #     t % args.training_steps_multiplier == 0,
        #     collect_data,
        #     skip_data_collection
        # )

        # # sample actor-step worth of transitions
        # buffer_state, transitions = replay_buffer.sample(buffer_state)
        # print(f"transitions.observation.shape: {transitions.observation.shape}", flush=True)
        
        # Sample actor-step worth of transitions N times and concatenate them (NOTE: just a bandaid fix right now, currently can sample repeat data)
        
        transitions_list = []
        for _ in range(args.num_episodes_per_env):
            buffer_state, new_transitions = replay_buffer.sample(buffer_state)
            transitions_list.append(new_transitions)

        # Concatenate all sampled transitions
        transitions = jax.tree_util.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=0),
            *transitions_list
        )

        print(f"transitions.observation.shape (after {args.num_episodes_per_env} episodes per env): {transitions.observation.shape}", flush=True)   

        # process transitions for training
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx), transitions, batch_keys
        )
        print(f"transitions.observation.shape (after flatten_crl_fn): {transitions.observation.shape}", flush=True)

        
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        print(f"transitions.observation.shape (after first reshape): {transitions.observation.shape}", flush=True)
        
              
        permutation = jax.random.permutation(experience_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        
        # I added this code, so as to ensure len(transitions.observation) is divisible by batch_size
        num_full_batches = len(transitions.observation) // args.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:num_full_batches * args.batch_size], transitions)
        print(f"transitions.observation.shape (after ensuring divisibility by batch_size): {transitions.observation.shape}", flush=True)
        
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
            transitions,
        )

        print(f"transitions.observation.shape (after processing): {transitions.observation.shape}", flush=True)
        
        if args.use_all_batches == 0:
            num_total_batches = transitions.observation.shape[0]
            selected_indices = jax.random.permutation(
                sgd_batches_key, 
                num_total_batches
            )[:args.num_sgd_batches_per_training_step]
            transitions = jax.tree_util.tree_map(
                lambda x: x[selected_indices], 
                transitions
            )
        print(f"transitions.observation.shape (after {args.use_all_batches}, selecting {args.num_sgd_batches_per_training_step} batches): {transitions.observation.shape}", flush=True)
        
        
        # take actor-step worth of training-step
        (training_state, _,), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

        return (training_state, env_state, buffer_state,), metrics

    @jax.jit
    def training_epoch(
        training_state,
        env_state,
        buffer_state,
        key,
    ):  
        @jax.jit
        def f(carry, t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k, 2)
            (ts, es, bs,), metrics = training_step(ts, es, bs, train_key, t)
            return (ts, es, bs, k), metrics

        #(training_state, env_state, buffer_state, key), metrics = jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=args.num_training_steps_per_epoch * args.training_steps_multiplier)
        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(f, (training_state, env_state, buffer_state, key), jnp.arange(args.num_training_steps_per_epoch * args.training_steps_multiplier))

        
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    key, prefill_key = jax.random.split(key, 2)

    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )
    

    if args.eval_actor == 0:
        '''Setting up evaluator'''
        evaluator = CrlEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
        
    elif args.eval_actor == 1:
        key, eval_actor_key = jax.random.split(key)
        evaluator = CrlEvaluator(
            lambda training_state, env, env_state, extra_fields: actor_step(
                training_state,
                env,
                env_state,
                eval_actor_key,
                extra_fields
            ),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
    
    elif args.eval_actor > 1:
        key, eval_actor_key = jax.random.split(key)
        evaluator = CrlEvaluator(
            # Replace deterministic_actor_step with a partial function of multi_sample_actor_step
            lambda training_state, env, env_state, extra_fields: multi_sample_actor_step(
                training_state, 
                env, 
                env_state, 
                eval_actor_key,  # Use dedicated key for action sampling
                args.eval_actor,  # Use eval_actor as K parameter
                extra_fields
            ),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
    

    training_walltime = 0
    print('starting training....', flush=True)
    start_time = time.time()  # Add this line before the training loop
    for ne in range(args.num_epochs):
        
        t = time.time()

        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(training_state, env_state, buffer_state, epoch_key)
        
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time

        sps = (args.env_steps_per_actor_step * args.num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/envsteps": training_state.env_steps.item(),
            **{f"training/{name}": value for name, value in metrics.items()},
        }

        metrics = evaluator.run_evaluation(training_state, metrics)

        print(f"epoch {ne} out of {args.num_epochs} complete. metrics: {metrics}", flush=True)

        if args.checkpoint:
            if ne < 5 or ne >= args.num_epochs - 5 or ne % 10 == 0:
                # Save current policy and critic params.
                params = (training_state.alpha_state.params, training_state.actor_state.params, training_state.critic_state.params)
                path = f"{save_path}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)
        
        if args.track:
            wandb.log(metrics, step=ne)

            if args.wandb_mode == 'offline':
                trigger_sync()
        
        hours_passed = (time.time() - start_time) / 3600
        print(f"Time elapsed: {hours_passed:.3f} hours", flush=True)

    
    if args.checkpoint:
        # Save current policy and critic params.
        params = (training_state.alpha_state.params, training_state.actor_state.params, training_state.critic_state.params)
        path = f"{save_path}/final.pkl"
        save_params(path, params)
        
    # After training is complete, render the final policy
    if args.capture_vis:
        def render_policy(training_state, save_path):
            """Renders the policy and saves it as an HTML file."""
            # JIT compile the rollout function
            @jax.jit
            def policy_step(env_state, actor_params):
                means, _ = actor.apply(actor_params, env_state.obs)
                actions = nn.tanh(means)
                next_state = env.step(env_state, actions)
                return next_state, env_state  # Return current state for visualization
            
            rollout_states = []
            for i in range(args.num_render):
                # env = Humanoid(backend=None or "spring")
                # if args.eval_env_id:
                #     env = make_env(args.eval_env_id)
                # else:
                #     env = make_env()
                env = make_env(args.eval_env_id)
                
                # Initialize environment
                rng = jax.random.PRNGKey(seed=i+1)
                env_state = jax.jit(env.reset)(rng)
                
                # Collect rollout using jitted function
                for _ in range(args.vis_length):
                    env_state, current_state = policy_step(env_state, training_state.actor_state.params)
                    rollout_states.append(current_state.pipeline_state)
            
            # Render and save
            html_string = html.render(env.sys, rollout_states)
            render_path = f"{save_path}/vis.html"
            with open(render_path, "w") as f:
                f.write(html_string)
            wandb.log({"vis": wandb.Html(html_string)})
            
        print("Rendering final policy...", flush=True)
        try:
            render_policy(training_state, save_path)
        except Exception as e:
            print(f"Error rendering final policy: {e}", flush=True)
        
    #After training is complete, save the Args
    if args.checkpoint:
        with open(f"{save_path}/args.pkl", 'wb') as f:
            pickle.dump(args, f)
        print(f"Saved args to {save_path}/args.pkl", flush=True)
        
    #After training is complete, save the replay buffer (if save_buffer is 1, this takes a lot of memory)
    if args.checkpoint:
        if args.save_buffer:
            print("Saving final buffer_state and buffer data (everything needed to recreate replay_buffer)...", flush=True)
            try:
                buffer_path = f"{save_path}/final_buffer.pkl"
                buffer_data = {
                    'buffer_state': buffer_state,
                    'max_replay_size': args.max_replay_size,
                    'batch_size': args.batch_size,
                    'num_envs': args.num_envs,
                    'episode_length': args.episode_length,
                }
                with open(buffer_path, 'wb') as f:
                    pickle.dump(buffer_data, f)
                print(f"Saved replay_buffer to {buffer_path}", flush=True)
            except Exception as e:
                print(f"Error saving final replay buffer: {e}", flush=True)
        
        
        
        
# (50000000 - 1024 x 1000) / 50 x 1024 x 62 = 15        #number of actor steps per epoch (which is equal to the number of training steps)
# 1024 x 999 / 256 = 4000                               #number of gradient steps per actor step 
# 1024 x 62 / 4000 = 16                                 #ratio of env steps per gradient step