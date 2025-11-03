import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from typing import Union, Dict, Any, Optional, NamedTuple, Generator
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback


class CustomDictRolloutBufferSamples(NamedTuple):
    """
    扩展 DictRolloutBufferSamples 以支持 chosen_action_embeddings
    """
    observations: Dict[str, torch.Tensor]
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    chosen_action_embeddings: torch.Tensor


class CustomRolloutBuffer(DictRolloutBuffer):
    """
    一个健壮的、可扩展的 Rollout Buffer，它在 DictRolloutBuffer 的基础上
    增加了对 `chosen_action_embeddings` 的支持。

    它通过重写 get() 方法来确保所有自定义字段都被正确地“展平” (flatten)，
    从而与 Stable Baselines 3 的内部数据流完全兼容。
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[torch.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        action_embedding_dim: int = 128,
    ):
        self.action_embedding_dim = action_embedding_dim
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        self.chosen_action_embeddings = None
        
    def reset(self) -> None:
        """重置 buffer, 为新字段分配内存"""
        super().reset()
        self.chosen_action_embeddings = np.zeros(
            (self.buffer_size, self.n_envs, self.action_embedding_dim), 
            dtype=np.float32
        )
    
    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        chosen_action_embedding: Optional[np.ndarray] = None,
    ) -> None:
        # 调用父类的 add 方法来处理所有标准数据
        super().add(obs, action, reward, episode_start, value, log_prob)
        
        # 单独处理我们的自定义数据
        # self.pos 在父类 add 方法中已经自增，所以这里用 self.pos - 1
        pos = self.pos - 1 if self.pos > 0 else self.buffer_size - 1
        if chosen_action_embedding is not None:
            if chosen_action_embedding.shape != (self.n_envs, self.action_embedding_dim):
                chosen_action_embedding = chosen_action_embedding.reshape(self.n_envs, self.action_embedding_dim)
            self.chosen_action_embeddings[pos] = chosen_action_embedding.copy()
        else:
            self.chosen_action_embeddings[pos] = np.zeros((self.n_envs, self.action_embedding_dim), dtype=np.float32)

    def get(self, batch_size: Optional[int] = None) -> Generator[CustomDictRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling"
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        # 准备数据进行采样，确保只展平一次
        if not self.generator_ready:
            # 展平所有标准字段
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)
            
            for tensor in ["actions", "values", "log_probs", "advantages", "returns"]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            
            # 显式展平我们的自定义字段
            self.chosen_action_embeddings = self.swap_and_flatten(self.chosen_action_embeddings)

            self.generator_ready = True

        # 按批次进行采样和迭代
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds)
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> CustomDictRolloutBufferSamples:
        """
        根据索引从展平后的数据中获取一个批次。
        """
        # 使用父类方法获取所有标准数据
        base_data = super()._get_samples(batch_inds)
        
        # 额外获取 chosen_action_embeddings 数据
        chosen_embeddings_tensor = self.to_torch(self.chosen_action_embeddings[batch_inds])

        return CustomDictRolloutBufferSamples(
            observations=base_data.observations,
            actions=base_data.actions,
            old_values=base_data.old_values,
            old_log_prob=base_data.old_log_prob,
            advantages=base_data.advantages,
            returns=base_data.returns,
            chosen_action_embeddings=chosen_embeddings_tensor
        )


class PPOWithAuxLoss(PPO):
    """
    带有辅助损失的PPO算法实现
    """
    
    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        aux_coef: float = 0.001,
        action_embedding_dim: int = 128,
        **kwargs,
    ):
        self.aux_coef = aux_coef
        self.action_embedding_dim = action_embedding_dim
        super().__init__(policy, env, **kwargs)
        
    def _setup_model(self) -> None:
        """设置模型，包括自定义RolloutBuffer"""
        super()._setup_model()
        
        self.rollout_buffer = CustomRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            action_embedding_dim=self.action_embedding_dim,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: CustomRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        重写collect_rollouts以捕获chosen_action_embedding
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, gym.spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # 提取chosen_action_embedding
            chosen_action_embeddings = self._extract_chosen_embeddings_from_info(infos)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                chosen_action_embedding=chosen_action_embeddings,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def _extract_chosen_embeddings_from_info(self, infos) -> np.ndarray:
        """从info字典中提取chosen_action_embeddings"""
        embeddings = []
        
        if isinstance(infos, dict):
            if 'chosen_action_embedding' in infos:
                embeddings.append(infos['chosen_action_embedding'])
            else:
                embeddings.append(np.zeros(self.action_embedding_dim))
        elif isinstance(infos, (list, tuple)):
            for info in infos:
                if isinstance(info, dict) and 'chosen_action_embedding' in info:
                    embeddings.append(info['chosen_action_embedding'])
                else:
                    embeddings.append(np.zeros(self.action_embedding_dim))
        
        embeddings = np.array(embeddings, dtype=np.float32)
        if embeddings.shape[0] != self.n_envs:
            if embeddings.shape[0] < self.n_envs:
                padding = np.zeros((self.n_envs - embeddings.shape[0], self.action_embedding_dim))
                embeddings = np.vstack([embeddings, padding])
            else:
                embeddings = embeddings[:self.n_envs]
                
        return embeddings

    def train(self) -> None:
        """Update policy using the currently gathered rollout buffer (with auxiliary loss)."""
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        aux_losses = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                # The warning indicates a shape mismatch. Flattening the returns tensor
                # ensures it is a 1D vector, matching the shape of values_pred, which is robust against buffer issues.
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # 计算辅助损失
                aux_loss = torch.tensor(0.0, device=self.device)
                if hasattr(self.policy, 'action_net') and rollout_data.chosen_action_embeddings.numel() > 0:
                    try:
                        # Correctly obtain action mean by following the SB3 policy architecture:
                        # 1. Extract features from the observation.
                        features = self.policy.extract_features(rollout_data.observations)
                        # 2. Pass features through the policy's MLP extractor (hidden layers).
                        latent_pi = self.policy.mlp_extractor.policy_net(features)
                        # 3. The output of the MLP is the input to the final action layer.
                        action_mean = self.policy.action_net(latent_pi)

                        # Compute the MSE loss against the true action embeddings.
                        aux_loss = F.mse_loss(action_mean, rollout_data.chosen_action_embeddings)
                    except Exception as e:
                        # If computing aux loss fails, record but do not crash training.
                        print(f"Warning: Failed to compute auxiliary loss: {e}")
                        aux_loss = torch.tensor(0.0, device=self.device)

                aux_losses.append(aux_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.aux_coef * aux_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/aux_loss", np.mean(aux_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


# 需要导入的辅助函数
def obs_as_tensor(obs, device):
    """Convert observation to tensor"""
    if isinstance(obs, dict):
        return {key: torch.as_tensor(_obs, device=device) for key, _obs in obs.items()}
    return torch.as_tensor(obs, device=device)