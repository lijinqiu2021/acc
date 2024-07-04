import argparse
import random
import sys
import time
from distutils.util import strtobool
from functools import partial
from typing import Sequence

import envpool
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def get_parse():
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-name", type=str, default="acc",
                        help="the name of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ACC",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--wandb-mode", type=str, default='online', choices=["offline", "online"],
                        help="the sync mode of wandb's project")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to save model into the `runs/{run_name}` folder")

    # environment specific arguments
    parser.add_argument("--env-id", type=str, default="Pong-v5",
                        choices=["Pong-v5", "Tennis-v5", "DoubleDunk-v5", "IceHockey-v5", "FishingDerby-v5", ],
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000,
                        help="total timesteps of the experiments")
    parser.add_argument("--seed", type=int, default=1,
                        help="the seed of the experiment")
    parser.add_argument("--mask-prob", type=float, default=0.08,
                        help="the probability of pixels being masked")

    # algorithm specific arguments
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    # fmt: on

    return parser


def make_env(env_id, seed, num_envs):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
            episodic_life=True,
            reward_clip=True,
            seed=seed,
        )
        envs.num_envs = num_envs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        envs.is_vector_env = True
        return envs

    return thunk


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)

        logits = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)
        return logits


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)

        value = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        return value


@flax.struct.dataclass
class AgentParams:
    actor_params: flax.core.FrozenDict
    executor_critic_params: flax.core.FrozenDict
    oracle_critic_params: flax.core.FrozenDict


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    executor_values: jnp.array
    oracle_values: jnp.array
    executor_returns: jnp.array
    oracle_returns: jnp.array
    advantages: jnp.array
    rewards: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


def run(args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size

    # track the experiment
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.mask_prob}__{args.total_timesteps}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            mode=args.wandb_mode,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, executor_critic_key, oracle_critic_key = jax.random.split(key, 4)

    # env setup
    envs = make_env(args.env_id, args.seed, args.num_envs)()
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )
    handle, recv, send, step_env = envs.xla()

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    @jax.jit
    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    # init network
    actor = Actor(action_dim=envs.single_action_space.n)
    executor_critic = Critic()
    oracle_critic = Critic()
    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            actor.init(actor_key, np.array([envs.single_observation_space.sample()])),
            executor_critic.init(executor_critic_key, np.array([envs.single_observation_space.sample()])),
            oracle_critic.init(oracle_critic_key, np.array([envs.single_observation_space.sample()])),
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
    )
    actor.apply = jax.jit(actor.apply)
    executor_critic.apply = jax.jit(executor_critic.apply)
    oracle_critic.apply = jax.jit(oracle_critic.apply)

    @jax.jit
    def get_action_and_value_for_sample(
        params: flax.core.FrozenDict,
        obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""
        executor_obs, oracle_obs = obs
        logits = actor.apply(params.actor_params, executor_obs)
        # sample action: Gumbel-softmax trick
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        executor_value = executor_critic.apply(params.executor_critic_params, executor_obs)
        oracle_value = oracle_critic.apply(params.oracle_critic_params, oracle_obs)
        return action, logprob, executor_value.squeeze(1), oracle_value.squeeze(1), key

    @jax.jit
    def get_action_and_value_for_train(
        params: flax.core.FrozenDict,
        obs: np.ndarray,
        action: np.ndarray,
    ):
        """calculate value, logprob of supplied `action`, and entropy"""
        executor_obs, oracle_obs = obs
        logits = actor.apply(params.actor_params, executor_obs)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        # normalize the logits
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        executor_value = executor_critic.apply(params.executor_critic_params, executor_obs).squeeze()
        oracle_value = oracle_critic.apply(params.oracle_critic_params, oracle_obs).squeeze()
        return logprob, entropy, executor_value, oracle_value

    def mask_observation(key, obs, dones, mask_maps, mask_prob):
        """mask obs to get executor_obs and oracle_obs"""

        def mask_single_observation(mask_key, ob, done, mask_map):
            new_mask_map = jax.random.choice(
                key=mask_key, a=jnp.array([True, False]), shape=ob.shape[1:], p=jnp.array([mask_prob, 1 - mask_prob])
            )
            new_mask_map = jnp.tile(new_mask_map, [ob.shape[0], 1, 1])
            now_mask_map = jnp.where(done, new_mask_map, mask_map)
            masked_ob = jnp.where(now_mask_map, 0, ob)
            return masked_ob, now_mask_map

        key, sub_key = jax.random.split(key, 2)
        mask_keys = jax.random.split(sub_key, obs.shape[0])
        executor_obs, mask_maps = jax.vmap(mask_single_observation, in_axes=(0, 0, 0, 0))(
            mask_keys,
            obs,
            dones,
            mask_maps,
        )
        return key, (executor_obs, obs), mask_maps

    mask_observation = partial(mask_observation, mask_prob=args.mask_prob)
    mask_observation = jax.jit(mask_observation)

    # interact with the environment and store data
    def step_env_wrappeed(episode_stats, handle, action, key, mask_maps):
        handle, (next_obs, reward, next_done, info) = step_env(handle, action)
        key, next_obs, mask_maps = mask_observation(key, next_obs, next_done, mask_maps)
        new_episode_return = episode_stats.episode_returns + info["reward"]
        new_episode_length = episode_stats.episode_lengths + 1
        episode_stats = episode_stats.replace(
            episode_returns=(new_episode_return) * (1 - info["terminated"]) * (1 - info["TimeLimit.truncated"]),
            episode_lengths=(new_episode_length) * (1 - info["terminated"]) * (1 - info["TimeLimit.truncated"]),
            # only update the `returned_episode_returns` if the episode is done
            returned_episode_returns=jnp.where(
                info["terminated"] + info["TimeLimit.truncated"], new_episode_return, episode_stats.returned_episode_returns
            ),
            returned_episode_lengths=jnp.where(
                info["terminated"] + info["TimeLimit.truncated"], new_episode_length, episode_stats.returned_episode_lengths
            ),
        )
        return episode_stats, handle, (next_obs, reward, next_done, info), key, mask_maps

    def step_once(carry, step, env_step_fn):
        agent_state, episode_stats, obs, done, key, handle, mask_maps = carry
        action, logprob, executor_value, oracle_value, key = get_action_and_value_for_sample(agent_state.params, obs, key)

        episode_stats, handle, (next_obs, reward, next_done, _), key, mask_maps = env_step_fn(
            episode_stats, handle, action, key, mask_maps
        )
        storage = Storage(
            obs=obs,
            actions=action,
            logprobs=logprob,
            dones=done,
            executor_values=executor_value,
            oracle_values=oracle_value,
            rewards=reward,
            executor_returns=jnp.zeros_like(reward),
            oracle_returns=jnp.zeros_like(reward),
            advantages=jnp.zeros_like(reward),
        )
        return (agent_state, episode_stats, next_obs, next_done, key, handle, mask_maps), storage

    def rollout(agent_state, episode_stats, next_obs, next_done, key, handle, mask_maps, step_once_fn, max_steps):
        (agent_state, episode_stats, next_obs, next_done, key, handle, mask_maps), storage = jax.lax.scan(
            step_once_fn, (agent_state, episode_stats, next_obs, next_done, key, handle, mask_maps), (), max_steps
        )
        return agent_state, episode_stats, next_obs, next_done, storage, key, handle, mask_maps

    rollout = partial(rollout, step_once_fn=partial(step_once, env_step_fn=step_env_wrappeed), max_steps=args.num_steps)
    rollout = jax.jit(rollout)

    # for train
    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        executor_next_obs, oracle_next_obs = next_obs
        executor_next_value = executor_critic.apply(agent_state.params.executor_critic_params, executor_next_obs).squeeze()
        oracle_next_value = oracle_critic.apply(agent_state.params.oracle_critic_params, oracle_next_obs).squeeze()

        executor_advantages = jnp.zeros((args.num_envs,))
        oracle_advantages = jnp.zeros((args.num_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        executor_values = jnp.concatenate([storage.executor_values, executor_next_value[None, :]], axis=0)
        oracle_values = jnp.concatenate([storage.oracle_values, oracle_next_value[None, :]], axis=0)

        _, executor_advantages = jax.lax.scan(
            compute_gae_once,
            executor_advantages,
            (dones[1:], executor_values[1:], executor_values[:-1], storage.rewards),
            reverse=True,
        )
        _, oracle_advantages = jax.lax.scan(
            compute_gae_once,
            oracle_advantages,
            (dones[1:], oracle_values[1:], oracle_values[:-1], storage.rewards),
            reverse=True,
        )
        advantages = jnp.maximum(executor_advantages, oracle_advantages)

        storage = storage.replace(
            advantages=advantages,
            executor_returns=executor_advantages + storage.executor_values,
            oracle_returns=oracle_advantages + storage.oracle_values,
        )
        return storage

    def ppo_loss(params, x, a, logp, mb_advantages, mb_executor_returns, mb_oracle_returns):
        newlogprob, entropy, executor_newvalue, oracle_newvalue = get_action_and_value_for_train(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # value loss
        executor_v_loss = 0.5 * ((executor_newvalue - mb_executor_returns) ** 2).mean()
        oracle_v_loss = 0.5 * ((oracle_newvalue - mb_oracle_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + executor_v_loss * args.vf_coef + oracle_v_loss * args.vf_coef
        return loss, (pg_loss, executor_v_loss, oracle_v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        def update_epoch(carry, unused_inp):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_map(flatten, storage)
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)

            def update_minibatch(agent_state, minibatch):
                (loss, (pg_loss, executor_v_loss, oracle_v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.executor_returns,
                    minibatch.oracle_returns,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, executor_v_loss, oracle_v_loss, entropy_loss, approx_kl, grads)

            agent_state, (loss, pg_loss, executor_v_loss, oracle_v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
                update_minibatch, agent_state, shuffled_storage
            )
            return (agent_state, key), (loss, pg_loss, executor_v_loss, oracle_v_loss, entropy_loss, approx_kl, grads)

        (agent_state, key), (loss, pg_loss, executor_v_loss, oracle_v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        return agent_state, loss, pg_loss, executor_v_loss, oracle_v_loss, entropy_loss, approx_kl, key

    # start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)
    key, next_obs, mask_maps = mask_observation(key, next_obs, jnp.ones(args.num_envs, dtype=jax.numpy.bool_), next_obs)

    tbar = tqdm(range(1, args.num_updates + 1), desc="PPO", file=sys.stdout)
    for update in tbar:
        update_time_start = time.time()

        agent_state, episode_stats, next_obs, next_done, storage, key, handle, mask_maps = rollout(
            agent_state, episode_stats, next_obs, next_done, key, handle, mask_maps
        )
        global_step += args.num_steps * args.num_envs
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, executor_v_loss, oracle_v_loss, entropy_loss, approx_kl, key = update_ppo(
            agent_state,
            storage,
            key,
        )

        # record training metrics
        writer.add_scalar("losses/executor_value_loss", executor_v_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/oracle_value_loss", oracle_v_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl[-1, -1].item(), global_step)
        writer.add_scalar("losses/loss", loss[-1, -1].item(), global_step)

        avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
        avg_episodic_length = np.mean(jax.device_get(episode_stats.returned_episode_lengths))
        cur_learning_rate = agent_state.opt_state[1].hyperparams["learning_rate"].item()
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        writer.add_scalar("charts/avg_episodic_length", avg_episodic_length, global_step)
        writer.add_scalar("charts/learning_rate", cur_learning_rate, global_step)

        fps = int(global_step / (time.time() - start_time))
        fps_update = int(args.batch_size / (time.time() - update_time_start))
        writer.add_scalar("train/FPS", fps, global_step)
        writer.add_scalar("train/FPS_update", fps_update, global_step)

        show_dict = {
            "update": update,
            "global_step": global_step,
            "FPS": fps,
            "FPS_update": fps_update,
            "[train]avg_episodic_return": avg_episodic_return,
            "[train]avg_episodic_length": avg_episodic_length,
        }
        tbar.set_postfix(show_dict, refresh=True)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        vars(args),
                        [
                            agent_state.params.actor_params,
                            agent_state.params.executor_critic_params,
                            agent_state.params.oracle_critic_params,
                        ],
                    ]
                )
            )
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    parser = get_parse()
    args = parser.parse_args()
    run(args)
