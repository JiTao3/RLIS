import copy
import logging
import sys
import cProfile

from datetime import datetime

import gym_db  # noqa: F401
from solution.experiment import Experiment
from solution.features_extractor import FeaturesExtractor
from solution.PPOWithAuxLoss import PPOWithAuxLoss  # 新增：导入自定义PPO
from solution.utils import print_model_summary
import stable_baselines3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from callback import TensorBoardRewardCallback, EntCoefScheduleCallback, ent_coef_linear
from selection.dbms import postgres_dbms
import math
import logging


def cosine_annealing(initial_lr: float, min_lr: float = 1e-7, cycles: int = 5):
    """确保最后阶段学习率最低的多周期退火"""

    def func(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        cycle_position = (progress * cycles) % 1.0
        cos_inner = math.pi * cycle_position
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(cos_inner))

    return func


def main():

    logging.basicConfig(level=logging.INFO)
    assert len(sys.argv) >= 2, "Experiment configuration file must be provided: main.py path_fo_file.json"
    CONFIGURATION_FILE = sys.argv[1]

    experiment = Experiment(CONFIGURATION_FILE)

    postgres_dbms.PORT = experiment.config["port"]
    experiment.prepare()

    envs = [experiment.make_env(env_id) for env_id in range(experiment.config["parallel_environments"])]
    training_env = SubprocVecEnv(envs)
    training_env = VecNormalize(
        training_env,
        norm_obs=False,
        norm_reward=True,
        gamma=experiment.config["rl_algorithm"]["model_params"]["args"]["gamma"],
        training=True,
    )

    # 创建非并行环境（用于测试）
    # training_env = experiment.make_env(0)()

    # 设置模型架构，添加特征提取器
    model_architecture = copy.copy(experiment.config["rl_algorithm"]["model_params"]["model_architecture"])
    model_architecture["features_extractor_class"] = FeaturesExtractor

    # 设置辅助损失权重
    aux_coef = experiment.config.get("aux_coef", 0.1)
    action_embedding_dim = experiment.config.get("action_embedding_dim", 158)

    logging.info(f"Using PPOWithAuxLoss with action_embedding_dim={action_embedding_dim}, aux_coef={aux_coef}")

    # 创建带辅助损失的PPO模型
    model = PPOWithAuxLoss(
        policy=experiment.config["rl_algorithm"]["policy"],
        env=training_env,
        aux_coef=aux_coef,
        action_embedding_dim=action_embedding_dim,
        verbose=2,
        seed=experiment.config["random_seed"],
        tensorboard_log=experiment.experiment_folder_path,
        policy_kwargs=model_architecture,
        learning_rate=cosine_annealing(experiment.config["rl_algorithm"]["model_params"]["learning_rate"]),
        ent_coef=experiment.config["rl_algorithm"]["model_params"]["ent_coef"],
        **experiment.config["rl_algorithm"]["model_params"]["args"],
        device=experiment.config["cuda_device"],
    )

    # 打印模型参数统计
    print_model_summary(model, experiment)

    experiment.set_model(model)

    if experiment.config["load_pretrained_model"]:
        experiment._load_pretrained_model(
            experiment.config["load_pretrained_model"], device=experiment.config["cuda_device"]
        )


    # 创建测试环境
    test_env_1 = experiment.make_test_env(0, 0, workload_idx=0, budget=0.3)()
    test_env_2 = experiment.make_test_env(1, 0, workload_idx=0, budget=0.7)()
    test_env_3 = experiment.make_test_env(2, 1, workload_idx=0, budget=0.3)()
    test_env_4 = experiment.make_test_env(3, 1, workload_idx=0, budget=0.7)()

    # --- 设置回调函数 ---
    # 1. 创建熵系数衰减的逻辑函数
    ent_coef_schedule = ent_coef_linear(experiment.config["rl_algorithm"]["model_params"]["ent_coef"], 0.001)

    # 2. 创建回调实例列表
    callbacks = [
        TensorBoardRewardCallback(eval_env=test_env_1),
        TensorBoardRewardCallback(eval_env=test_env_2),
        TensorBoardRewardCallback(eval_env=test_env_3),
        TensorBoardRewardCallback(eval_env=test_env_4),
        EntCoefScheduleCallback(ent_coef_schedule),
    ]

    experiment.start_learning()

    current_date = datetime.now().strftime("%Y%m%d_%H%M")
    logging.info(f"@@@@@@@ Current date: {current_date} @@@@@@")

    # 开始训练
    model.learn(
        total_timesteps=experiment.config["rl_algorithm"]["total_timesteps"],
        tb_log_name="expv1.0" + str(experiment.id) + "_" + current_date,
        callback=CallbackList(callbacks),
    )

    experiment.finish_learning(training_env, 0, current_date)


if __name__ == "__main__":
    # cProfile.run('main()', sort='cumulative')
    main()
