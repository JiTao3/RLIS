import copy
import logging
import sys

from datetime import datetime

import gym_db  # noqa: F401
from solution.experiment import Experiment
from solution.features_extractor import FeaturesExtractor
from solution.PPOWithAuxLoss import PPOWithAuxLoss
from solution.utils import print_model_summary
import stable_baselines3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from callback import TensorBoardRewardCallback, EntCoefScheduleCallback, ent_coef_linear
from selection.dbms import postgres_dbms
import math
import logging


def cosine_annealing(initial_lr: float, min_lr: float = 1e-6, cycles: int = 5):
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
        norm_reward=False,
        gamma=experiment.config["rl_algorithm"]["model_params"]["args"]["gamma"],
        training=True,
    )

    model_architecture = copy.copy(experiment.config["rl_algorithm"]["model_params"]["model_architecture"])
    model_architecture["features_extractor_class"] = FeaturesExtractor

    aux_coef = experiment.config.get("aux_coef", 0.1)
    action_embedding_dim = experiment.config.get("action_embedding_dim", 158)

    logging.info(f"Using PPOWithAuxLoss with action_embedding_dim={action_embedding_dim}, aux_coef={aux_coef}")

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

    print_model_summary(model, experiment)

    experiment.set_model(model)

    if experiment.config["load_pretrained_model"]:
        experiment._load_pretrained_model(
            experiment.config["load_pretrained_model"],
            device=experiment.config["cuda_device"],
            load_best_cost_so_far=False,
        )

    experiment.start_learning()

    current_date = datetime.now().strftime("%Y%m%d_%H%M")
    logging.info(f"@@@@@@@ Current date: {current_date} @@@@@@")

    test_env_1 = experiment.make_test_env(0, 0, workload_idx=0, budget=0.3)()
    test_env_2 = experiment.make_test_env(1, 0, workload_idx=0, budget=0.7)()

    ent_coef_schedule = ent_coef_linear(experiment.config["rl_algorithm"]["model_params"]["ent_coef"], 0.001)

    callbacks = [
        TensorBoardRewardCallback(eval_env=test_env_1),
        TensorBoardRewardCallback(eval_env=test_env_2),
        EntCoefScheduleCallback(ent_coef_schedule, t_max=0.05, t_min=0.05),
    ]

    # 开始训练
    model.learn(
        total_timesteps=experiment.config["rl_algorithm"]["total_timesteps"],
        tb_log_name="exp_ft_v1.0" + str(experiment.id) + "_" + current_date,
        callback=CallbackList(callbacks),
    )

    experiment.finish_learning(training_env, 0, current_date)


if __name__ == "__main__":
    main()
