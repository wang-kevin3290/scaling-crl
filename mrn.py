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
from brax.io import html, model
from etils import epath
from dataclasses import dataclass
from collections import namedtuple
from typing import NamedTuple, Any
from wandb_osh.hooks import TriggerWandbSyncHook
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling

from src import utils
from evaluator import CrlEvaluator
from buffer import TrajectoryUniformSamplingQueue

@dataclass
class Args:
    exp_name: str = "train" # os.path.basename(__file__)[: -len(".py")]
    seed: int = 14
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "Ant Push Different Widths"
    # wandb_entity: str = 'ishaanjav-princeton'
    wandb_entity: str = 'ij9461-princeton-university'
    wandb_mode: str = 'offline'
    wandb_dir: str = '.'
    wandb_group: str = '.'
    capture_video: bool = True
    checkpoint: bool = False

    #environment specific arguments
    env_id: str = "ant" #"ant_hardest_maze" #"ant_big_maze" #"ant_ball" #"humanoid" #"ant"
    episode_length: int = 1000
    # to be filled in runtime
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0

    # Algorithm specific arguments
    total_env_steps: int = 10000000 # 50000000
    num_epochs: int = 10 # 50
    num_envs: int = 512
    num_eval_envs: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    logsumexp_penalty_coeff: float = 0.1

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    
    unroll_length: int  = 62
    mrn: bool = False
    
    # ADDING IN A NETWORK WIDTH ARGUMENT
    network_width: int = 1024
    
    
    # to be filled in runtime
    env_steps_per_actor_step : int = 0
    """number of env steps per actor step (computed in runtime)"""
    num_prefill_env_steps : int = 0
    """number of env steps to fill the buffer before starting training (computed in runtime)"""
    num_prefill_actor_steps : int = 0
    """number of actor steps to fill the buffer before starting training (computed in runtime)"""
    num_training_steps_per_epoch : int = 0
    """the number of training steps per epoch(computed in runtime)"""

class SA_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = jnp.concatenate([s, a], axis=-1)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
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
    
    
class G_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    @nn.compact
    def __call__(self, g: jnp.ndarray):

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(g)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x

class Actor(nn.Module):
    action_size: int
    norm_type = "layer_norm"
    network_width: int = 1024
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        # print(f"x.shape: {x.shape}", flush=True)

        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)

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

    run_name = f"{args.env_id}_{args.batch_size}_{args.total_env_steps}_nenvs:{args.num_envs}_width:{args.network_width}_{int(time.time())}_{args.seed}"

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
        save_path = Path(args.wandb_dir) / Path(run_name)
        os.mkdir(path=save_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

    # Environment setup    
    if args.env_id == "ant":
        from envs.ant import Ant
        env = Ant(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )

        args.obs_dim = 29
        args.goal_start_idx = 0
        args.goal_end_idx = 2

    elif "maze" in args.env_id:
        from envs.ant_maze import AntMaze
        env = AntMaze(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
            maze_layout_name=args.env_id[4:]
        )

        args.obs_dim = 29
        args.goal_start_idx = 0
        args.goal_end_idx = 2
    
    elif args.env_id == "ant_ball":
        from envs.ant_ball import AntBall
        env = AntBall(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )

        args.obs_dim = 31
        args.goal_start_idx = -4
        args.goal_end_idx = -2

    elif args.env_id == "ant_push":
        from envs.ant_push import AntPush
        env = AntPush(
            backend="mjx",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )

        args.obs_dim = 31
        args.goal_start_idx = 0
        args.goal_end_idx = 2
    
    elif args.env_id == "humanoid":
        from envs.humanoid import Humanoid
        env = Humanoid(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )

        args.obs_dim = 268
        args.goal_start_idx = 0
        args.goal_end_idx = 3

    else:
        raise NotImplementedError

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

    # Network setup
    # Actor
    actor = Actor(action_size=action_size, network_width=args.network_width)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, obs_size])),
        tx=optax.adam(learning_rate=args.actor_lr)
    )

    # Critic
    sa_encoder = SA_encoder(network_width=args.network_width)
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, args.obs_dim]), np.ones([1, action_size]))
    g_encoder = G_encoder(network_width=args.network_width)
    g_encoder_params = g_encoder.init(g_key, np.ones([1, args.goal_end_idx - args.goal_start_idx]))
    # c = jnp.asarray(0.0, dtype=jnp.float32) (NOT USED IN CODE, WHATS THIS)
    # Sym and Asym networks
    sym = Sym()
    # sym_params = sym.init(key, jnp.ones([1, args.network_width]))
    # Update `Sym` initialization to pass an input of shape (32, args.network_width)
    sym_params = sym.init(key, jnp.ones([1, 64]))
    asym = Asym()
    asym_params = asym.init(key, jnp.ones([1, 64]))

    # Add Sym and Asym to critic parameters
    critic_state = TrainState.create(
        apply_fn=None,
        params={
            "sa_encoder": sa_encoder_params,
            "g_encoder": g_encoder_params,
            "sym": sym_params,
            "asym": asym_params
        },
        tx=optax.adam(learning_rate=args.critic_lr),
    )
    # critic_state = TrainState.create(
    #     apply_fn=None,
    #     params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
    #     tx=optax.adam(learning_rate=args.critic_lr),
    # )

    # Entropy coefficient
    target_entropy = -0.5 * action_size
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": log_alpha},
        tx=optax.adam(learning_rate=args.alpha_lr),
    )
    
    # Trainstate
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
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
    
    def actor_step(actor_state, env, env_state, key, extra_fields):        
        means, log_stds = actor.apply(actor_state.params, env_state.obs)
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

    @jax.jit
    def get_experience(actor_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            env_state, transition = actor_step(actor_state, env, env_state, current_key, extra_fields=("truncation", "seed"))
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
                training_state.actor_state,
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
        def critic_loss(critic_params, transitions, key):
            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]

            obs = transitions.observation[:, :args.obs_dim]
            action = transitions.action

            sa_repr = sa_encoder.apply(sa_encoder_params, obs, action)
            g_repr = g_encoder.apply(g_encoder_params, transitions.observation[:, args.obs_dim:])

            if args.mrn == "func":
                # MRN distance loss
                logits = -utils.mrn_distance(sa_repr[:, None, :], g_repr[None, :])
                critic_loss_value = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
                # Returning dummy aux values
                return critic_loss_value, (None, None, None, None, None)

            elif args.mrn == "arch": #THIS HAS SHAPE ERROR
                # Pass through the Sym network
                sym1 = sym.apply(critic_params['sym'], sa_repr)
                sym2 = sym.apply(critic_params['sym'], g_repr)
                dist_s = jnp.mean((sym1 - sym2) ** 2, axis=-1, keepdims=True)  # Symmetric distance

                # Pass through the Asym network
                asym1 = asym.apply(critic_params['asym'], sa_repr)
                asym2 = asym.apply(critic_params['asym'], g_repr)
                res = jax.nn.relu(asym1 - asym2)
                dist_a = jnp.max(res, axis=-1, keepdims=True)  # Asymmetric distance

                # Total distance (combine symmetric and asymmetric distances)
                logits = -(dist_s + dist_a)
                critic_loss_value = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

            else:
                # InfoNCE
                logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))  # shape = BxB
                critic_loss_value = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

            # logsumexp regularization
            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            critic_loss_value += args.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

            I = jnp.eye(logits.shape[0])
            correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
            logits_pos = jnp.sum(logits * I) / jnp.sum(I)
            logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

            return critic_loss_value, (logsumexp, I, correct, logits_pos, logits_neg)
            
        (loss, (logsumexp, I, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(critic_loss, has_aux=True)(training_state.critic_state.params, transitions, key)
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state = new_critic_state)

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
    def training_step(training_state, env_state, buffer_state, key):
        experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)

        # update buffer
        env_state, buffer_state = get_experience(
            training_state.actor_state,
            env_state,
            buffer_state,
            experience_key1,
        )

        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.env_steps_per_actor_step,
        )

        # sample actor-step worth of transitions
        buffer_state, transitions = replay_buffer.sample(buffer_state)

        # process transitions for training
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx), transitions, batch_keys
        )

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        permutation = jax.random.permutation(experience_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)

        # Padding to ensure the total size is divisible by batch size
        total_samples = transitions.observation.shape[0]
        remainder = total_samples % args.batch_size

        if remainder != 0:
            padding_size = args.batch_size - remainder
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.pad(x, ((0, padding_size),) + ((0, 0),) * (x.ndim - 1)),
                transitions
            )

        # Reshape after padding
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
            transitions,
        )

        # take actor-step worth of training-step
        (training_state, _,), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

        return (training_state, env_state, buffer_state,), metrics

  
    @jax.jit
    def training_step(training_state, env_state, buffer_state, key):
        experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)

        # update buffer
        env_state, buffer_state = get_experience(
            training_state.actor_state,
            env_state,
            buffer_state,
            experience_key1,
        )

        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.env_steps_per_actor_step,
        )

        # sample actor-step worth of transitions
        buffer_state, transitions = replay_buffer.sample(buffer_state)

        # process transitions for training
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx), 
            transitions, 
            batch_keys
        )  
        
        # Flatten the transitions
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )

        # Permute the transitions
        permutation = jax.random.permutation(experience_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)

        # Truncate to nearest multiple of batch_size
        num_samples = len(transitions.observation)
        num_complete_batches = num_samples // args.batch_size
        truncated_size = num_complete_batches * args.batch_size

        transitions = jax.tree_util.tree_map(
            lambda x: x[:truncated_size], 
            transitions
        )

        # Reshape into batches
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
            transitions,
        )

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
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k, 2)
            (ts, es, bs,), metrics = training_step(ts, es, bs, train_key)
            return (ts, es, bs, k), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=args.num_training_steps_per_epoch)
        
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    key, prefill_key = jax.random.split(key, 2)

    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )

    '''Setting up evaluator'''
    evaluator = CrlEvaluator(
        deterministic_actor_step,
        env,
        num_eval_envs=args.num_eval_envs,
        episode_length=args.episode_length,
        key=eval_env_key,
    )

    training_walltime = 0
    print('starting training....')
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

        print(metrics)

        if args.checkpoint:
            # Save current policy and critic params.
            params = (training_state.alpha_state.params, training_state.actor_state.params, training_state.critic_state.params)
            path = f"{save_path}/step_{int(training_state.env_steps)}.pkl"
            save_params(path, params)
        
        if args.track:
            wandb.log(metrics, step=ne)
            # Define the render function
            def render(exp_dir, exp_name):
                # Initialize rng correctly and reset for each usage in the loop
                rng = jax.random.PRNGKey(seed=1)
                inference_fn = lambda x: actor.apply(training_state.actor_state.params, x)
                jit_env_reset = jax.jit(env.reset)
                jit_env_step = jax.jit(env.step)
                jit_inference_fn = jax.jit(inference_fn)

                rollout = []
                state = jit_env_reset(rng=rng)
                for i in range(5000):
                    rollout.append(state.pipeline_state)
                    
                    # Split rng to get a new key each time
                    rng, act_rng = jax.random.split(rng)
                    act, _ = jit_inference_fn(state.obs, act_rng)
                    state = jit_env_step(state, act)
                    
                    if i % 1000 == 0:
                        rng, reset_rng = jax.random.split(rng)
                        state = jit_env_reset(rng=reset_rng)

                # Render HTML and log to WandB
                url = html.render(env.sys.replace(dt=env.dt), rollout, height=1024)
                html_path = os.path.join(exp_dir, f"{exp_name}.html")
                with open(html_path, "w") as file:
                    file.write(url)

                # Log to WandB
                wandb.log({"render": wandb.Html(html_path)})

            # Call render before syncing
            # render("/home/ij9461/Documents/IW/T/JaxGCRL/clean_JaxGCRL/renders", args.exp_name)

            if args.wandb_mode == 'offline':
                trigger_sync()
    
    if args.checkpoint:
        # Save current policy and critic params.
        params = (training_state.alpha_state.params, training_state.actor_state.params, training_state.critic_state.params)
        path = f"{save_path}/final.pkl"
        save_params(path, params)
        
# (50000000 - 1024 x 1000) / 50 x 1024 x 62 = 15        #number of actor steps per epoch (which is equal to the number of training steps)
# 1024 x 999 / 256 = 4000                               #number of gradient steps per actor step 
# 1024 x 62 / 4000 = 16                                 #ratio of env steps per gradient step