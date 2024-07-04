import itertools
from collections import deque

import torch

from utils.format_utils import DictList, batch_tensor_obs_squeeze, vec_obs_as_tensor


class A2CAlgo:
    """The Advantage Actor-Critic algorithm."""

    def __init__(
        self,
        envs,
        actor_model,
        critic_model_list,
        device=None,
        num_frames=None,
        gamma=0.99,
        lr=0.01,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        reshape_reward_fn=None,
        reshape_adv_fn=None,
        rmsprop_alpha=0.99,
        rmsprop_eps=1e-8,
    ):
        # Store parameters
        self.vec_env = envs
        self.actor_model = actor_model
        self.critic_model_list = critic_model_list
        self.num_critic = len(self.critic_model_list)
        self.device = device
        self.num_frames = num_frames or 8
        self.gamma = gamma
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        if reshape_reward_fn is None:
            reshape_reward_fn = lambda reward: reward
        self.reshape_reward_fn = reshape_reward_fn
        if reshape_adv_fn is None:
            reshape_adv_fn = lambda adv: adv[..., 0]
        self.reshape_adv_fn = reshape_adv_fn

        # Control parameters
        self.model_recurrent = self.actor_model.recurrent
        critic_memory_size = None
        for critic_model in self.critic_model_list:
            self.model_recurrent &= critic_model.recurrent
            if critic_memory_size is None:
                critic_memory_size = critic_model.memory_size
            else:
                assert critic_memory_size == critic_model.memory_size

        # Configure acmodel
        self.actor_model.to(self.device)
        self.actor_model.train()
        for critic_model in self.critic_model_list:
            critic_model.to(self.device)
            critic_model.train()

        # Store helpers values
        self.num_envs = envs.num_envs
        self.batch_size = self.num_frames * self.num_envs

        # Initialize experience values
        shape = (self.num_frames, self.num_envs)
        self.obs, _ = self.vec_env.reset()
        self.obss = [None] * self.num_frames
        self.last_not_done = torch.ones(self.num_envs, device=self.device)
        self.not_dones = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)
        self.values = torch.zeros((*shape, self.num_critic), device=self.device)
        self.advantages = torch.zeros((*shape, self.num_critic), device=self.device)
        # memory
        self.last_actor_memory = torch.zeros((self.num_envs, self.actor_model.memory_size), device=self.device)
        self.actor_memories = torch.zeros((*shape, self.actor_model.memory_size), device=self.device)
        self.last_critic_memory = torch.zeros((self.num_envs, critic_memory_size, self.num_critic), device=self.device)
        self.critic_memories = torch.zeros((*shape, critic_memory_size, self.num_critic), device=self.device)

        # Initialize log values
        self.log_episode_return = torch.zeros(self.num_envs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_envs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_envs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_envs
        self.log_reshaped_return = [0] * self.num_envs
        self.log_num_frames = [0] * self.num_envs
        self.log_avg_return = deque([0] * 100, maxlen=100)

        self.parameters = itertools.chain(*(model.parameters() for model in self.critic_model_list + [self.actor_model]))
        self.optimizer = torch.optim.RMSprop(self.parameters, lr, alpha=rmsprop_alpha, eps=rmsprop_eps)

    def collect_experiences(self):
        for i in range(self.num_frames):
            # Do one agent-environment interaction

            tensor_obs = vec_obs_as_tensor(self.obs, device=self.device)
            with torch.no_grad():
                dist, actor_memory = self.actor_model(
                    tensor_obs,
                    self.last_actor_memory,
                )
                value, critic_memory = [], []
                for cm_i, critic_model in enumerate(self.critic_model_list):
                    value_i, critic_memory_i = critic_model(
                        tensor_obs,
                        self.last_critic_memory[..., cm_i],
                    )
                    value.append(value_i)
                    critic_memory.append(critic_memory_i)
                value = torch.stack(value, dim=-1)
                critic_memory = torch.stack(critic_memory, dim=-1)
            action = dist.sample()

            obs, reward, terminated, truncated, _ = self.vec_env.step(action.cpu().numpy())
            done = terminated | truncated
            not_done = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            reward = torch.tensor(reward, device=self.device, dtype=torch.float)

            # Update experiences values
            self.obss[i] = tensor_obs
            self.actor_memories[i] = self.last_actor_memory
            self.critic_memories[i] = self.last_critic_memory
            self.not_dones[i] = self.last_not_done
            self.actions[i] = action
            self.values[i] = value
            self.rewards[i] = self.reshape_reward_fn(reward)
            self.log_probs[i] = dist.log_prob(action)

            self.obs = obs
            self.last_not_done = not_done
            # clear memory if done
            self.last_actor_memory = actor_memory * self.last_not_done.unsqueeze(-1)
            self.last_critic_memory = critic_memory * self.last_not_done.unsqueeze(-1).unsqueeze(-1)

            # Update log values
            self.log_episode_return += reward
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += 1

            for j, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[j].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[j].item())
                    self.log_num_frames.append(self.log_episode_num_frames[j].item())
                    self.log_avg_return.append(self.log_episode_return[j].item())

            self.log_episode_return *= self.last_not_done
            self.log_episode_reshaped_return *= self.last_not_done
            self.log_episode_num_frames *= self.last_not_done

        # Add advantage and return to experiences
        tensor_obs = vec_obs_as_tensor(self.obs, device=self.device)
        with torch.no_grad():
            last_value = []
            for cm_i, critic_model in enumerate(self.critic_model_list):
                last_value_i, _ = critic_model(
                    tensor_obs,
                    self.last_critic_memory[..., cm_i],
                )
                last_value.append(last_value_i)
            last_value = torch.stack(last_value, dim=-1)

        self.advantages = self.calc_advantages_gae(
            self.rewards,
            self.not_dones,
            self.values,
            self.last_not_done,
            last_value,
            self.gamma,
            self.gae_lambda,
        )
        self.returnns = self.values + self.advantages

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames,
        #   - E is self.n_envs,
        #   - D is the dimensionality.

        exps = DictList()
        # T x E x D -> E x T x D -> (E * T) x D
        exps.obs = batch_tensor_obs_squeeze(self.obss)
        # T x E x D -> E x T x D -> (E * T) x D
        exps.actor_memory = self.actor_memories.transpose(0, 1).reshape(-1, *self.actor_memories.shape[2:])
        exps.critic_memory = self.critic_memories.transpose(0, 1).reshape(-1, *self.critic_memories.shape[2:])
        # for all tensors below, T x E -> E x T -> (E * T)
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        # for all tensors below, T x E x Nc -> E x T x Nc -> (E * T) x Nc
        exps.value = self.values.transpose(0, 1).reshape(-1, self.num_critic)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1, self.num_critic)
        exps.returnn = self.returnns.transpose(0, 1).reshape(-1, self.num_critic)
        exps.advantage = self.reshape_adv_fn(exps.advantage)

        # Log some values (Select at least self.n_envs data to display)
        keep = max(self.log_done_counter, self.num_envs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.batch_size,
            "avg_return": self.log_avg_return,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_envs :]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_envs :]
        self.log_num_frames = self.log_num_frames[-self.num_envs :]

        return exps, logs

    @staticmethod
    def calc_advantages_gae(rewards, not_dones, values, last_not_done, last_value, gamma=1.0, gae_lambda=1.0):
        if len(rewards.shape) < len(values.shape):
            rewards = rewards.unsqueeze(-1)
            not_dones = not_dones.unsqueeze(-1)
            last_not_done = last_not_done.unsqueeze(-1)

        advantages = torch.zeros_like(values)
        num_frames = rewards.shape[0]
        for i in reversed(range(num_frames)):
            next_not_done = not_dones[i + 1] if i < num_frames - 1 else last_not_done
            next_value = values[i + 1] if i < num_frames - 1 else last_value
            next_advantage = advantages[i + 1] if i < num_frames - 1 else 0

            delta = rewards[i] + gamma * next_value * next_not_done - values[i]
            advantages[i] = delta + gamma * gae_lambda * next_advantage * next_not_done
        return advantages

    def update_parameters(self, exps):
        # Compute loss
        dist, _ = self.actor_model(
            exps.obs,
            exps.actor_memory,
        )
        value = []
        for cm_i, critic_model in enumerate(self.critic_model_list):
            value_i, _ = critic_model(
                exps.obs,
                exps.critic_memory[..., cm_i],
            )
            value.append(value_i)
        value = torch.stack(value, dim=-1)

        entropy = dist.entropy().mean()
        policy_loss = -(dist.log_prob(exps.action) * exps.advantage).mean()
        value_loss = (value - exps.returnn).pow(2).mean(dim=0).sum()
        loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

        # Update batch values
        update_entropy = entropy.item()
        update_value = value.mean().item()
        update_policy_loss = policy_loss.item()
        update_value_loss = value_loss.item()
        update_loss = loss

        # Update actor-critic
        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.parameters) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
        self.optimizer.step()

        # Log some values
        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "grad_norm": update_grad_norm,
        }
        return logs
