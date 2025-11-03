from stable_baselines3.common.callbacks import BaseCallback
import logging
import numpy as np

class TensorBoardRewardCallback(BaseCallback):
    def __init__(self, verbose=0, eval_env=None):
        super().__init__(verbose)
        # 初始化episode统计追踪
        self.num_envs = self.training_env.num_envs if hasattr(self, 'training_env') else 1
        self.current_episode_norm_rewards = [0.0] * self.num_envs   # 当前episode的奖励累积
        self.current_episode_original_rewards = [0.0] * self.num_envs   # 当前episode的奖励累积
        self.current_episode_steps = [0] * self.num_envs     # 当前episode的步数
        self.rollout_count = 0
        self.eval_env = eval_env
        self.eval_env.unwrapped.set_temperature(0.1)

    def _on_training_start(self):
        self.logger.info("on_training_start")
        self.num_envs = self.training_env.num_envs if hasattr(self, 'training_env') else 1
        self.logger.info(f"num_envs: {self.num_envs}")
        self.current_episode_norm_rewards = [0.0] * self.num_envs   # 当前episode的奖励累积
        self.current_episode_original_rewards = [0.0] * self.num_envs   # 当前episode的奖励累积
        self.current_episode_steps = [0] * self.num_envs     # 当前episode的步数

    
    def _on_rollout_start(self) -> None:
        self.rollout_count += 1
        logging.info(f"🔄 开始第 {self.rollout_count} 个 rollout")

    def _on_step(self):
        # 获取当前步的原始奖励
        current_lr = self.model.lr_schedule(self.model._current_progress_remaining)
        self.logger.record("train/learning_rate", current_lr)
            
        # 获取标准化前的原始奖励
        original_rewards = self.training_env.get_original_reward()
        

        dones = self.locals.get('dones', [])
        norm_rewards = self.locals.get('rewards', [])  # 标准化后的奖励
        any_episode_ended = False
        # 累积episode统计
        dones = self.locals['dones']
        norm_rewards = self.locals['rewards']
        infos = self.locals['infos']
        for i in range(len(dones)):
            self.current_episode_norm_rewards[i] += norm_rewards[i]
            self.current_episode_original_rewards[i] += original_rewards[i]
            self.current_episode_steps[i] += 1
            if dones[i]:
                achieved_cost_ratio = infos[i]['achieved_cost_ratio']
                database_name = infos[i]['database_name']
                budget = infos[i]['budget']
                self.logger.record(f"{database_name}_B_{budget:.1f}/cost_ratio", achieved_cost_ratio)
                self.logger.record(f"{database_name}_B_{budget:.1f}/norm_reward", self.current_episode_norm_rewards[i])
                self.logger.record(f"{database_name}_B_{budget:.1f}/original_reward", self.current_episode_original_rewards[i])
                self.logger.record(f"{database_name}_B_{budget:.1f}/episode_steps", self.current_episode_steps[i])
                self.current_episode_norm_rewards[i] = 0.0
                self.current_episode_original_rewards[i] = 0.0
                self.current_episode_steps[i] = 0
                any_episode_ended = True

        if any_episode_ended:
            self.logger.dump(step=self.num_timesteps)
        return True

    def _on_rollout_end(self):
        if self.eval_env is None:
            logging.warning("Eval env not set, skipping rollout evaluation.")
            return

        logging.info(f"Evaluating model at the end of rollout {self.rollout_count}...")
        n_eval_episodes = 1
        episode_rewards = []
        episode_lengths = []
        episode_achieved_cost_ratios = []

        for _ in range(n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                db_name = info["database_name"]
                budget = info["budget"]

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if "achieved_cost_ratio" in info:
                episode_achieved_cost_ratios.append(info["achieved_cost_ratio"])
        
        mean_reward = np.mean(episode_rewards)
        mean_ep_length = np.mean(episode_lengths)
        
        self.logger.record(f"eval_{db_name}_B_{budget:.1f}/mean_reward", mean_reward)
        self.logger.record(f"eval_{db_name}_B_{budget:.1f}/mean_ep_length", mean_ep_length)
        if episode_achieved_cost_ratios:
            self.logger.record(f"eval_{db_name}_B_{budget:.1f}/mean_achieved_cost_ratio", np.mean(episode_achieved_cost_ratios))
        if hasattr(self.eval_env.unwrapped, 'temperature'):
            self.logger.record(f"eval_{db_name}_B_{budget:.1f}/temperature", self.eval_env.unwrapped.temperature)



        self.logger.dump(step=self.num_timesteps)
        logging.info(f"Evaluation complete. Mean reward: {mean_reward:.2f}")



def ent_coef_linear(max_ent_coef: float, min_ent_coef: float=0.001):
    def func(progress_remaining: float) -> float:
        return min_ent_coef + progress_remaining * (max_ent_coef - min_ent_coef)

    return func

class EntCoefScheduleCallback(BaseCallback):
    """
    一个用于在训练期间动态调整熵系数(ent_coef)的回调。
    """
    def __init__(self, schedule_function, t_max=0.1, t_min=0.05, verbose=0):
        super().__init__(verbose)
        self.schedule_function = schedule_function

        self.T_max = t_max
        self.T_min = t_min
        self.current_temperature = self.T_max

    def _on_step(self) -> bool:
        """
        在每个训练步骤后被调用。
        """
        # 计算剩余进度的比例
        progress_remaining = 1.0 - self.num_timesteps / self.model._total_timesteps
        # 根据衰减函数更新模型的ent_coef
        self.model.ent_coef = self.schedule_function(progress_remaining)
        # (可选) 将ent_coef的值记录到TensorBoard
        self.logger.record("train/ent_coef", self.model.ent_coef)

        progress = min(1.0, self.num_timesteps / self.model._total_timesteps)
        self.current_temperature = self.T_min + (self.T_max - self.T_min) * (1.0 - progress)**2

        if hasattr(self.training_env, 'env_method'):
            self.training_env.env_method('set_temperature', self.current_temperature)
        

        return True

