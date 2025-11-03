import datetime
import logging
from math import log
import os
import pickle
import random
from multiprocessing import Manager

import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv

from gym_db.common import EnvironmentType

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO

from . import utils
from .configuration_parser import ConfigurationParser
from .schema import Schema
from .workload_generator import TrainingWorkloadGenerator
from .workload_embedder import SQLWorkloadEmbedder
from .statistics import Statistics
from .action_manager import DenseRepresentationActionManager
from .observation_manager import EmbeddingObservationManager
from .reward_calculator import NormalizedIndexReward
from selection.database_context import DatabaseContext

from .file_lock import create_db_locks


class Experiment(object):
    """
    Single DB Version Experiment
    TODO : 暂时采用枚举所有可能1-2列索引的方法。
    TODO : Multi DB -> 以workload为单位 训练的时候切换上下文。
    """

    def __init__(self, configuration_file):
        cp = ConfigurationParser(configuration_file)
        self.config = cp.config

        self.id = self.config["id"]
        self.model = None

        self.rnd = random.Random()
        self.rnd.seed(self.config["random_seed"])

        self.number_of_features = None
        self.number_of_actions = None

        self.EXPERIMENT_RESULT_PATH = self.config["result_path"]
        self.EXPERIMENT_STATISTICS_PATH = self.config["statistics_path"]
        self._create_experiment_folder()

        self.statistic = None
        self.database_name = None
        self.database_contexts = {}

        self.schema = None
        self.workloads = None
        self.globally_indexable_columns = None
        self.globally_indexable_columns_flat = None
        self.action_storage_consumptions = None

        self.db_locks = None
        self.manager = Manager()

        self.eval_envs = self.config["workload"]["eval_benchmarks"]
        self.training_envs = [env for env in self.config["workload"]["benchmarks"] if env not in self.eval_envs]
        if len(self.training_envs) == 0:
            self.training_envs = self.config["workload"]["benchmarks"]

        shared_best_cost = self.manager.dict()
        self.reward_calculator = NormalizedIndexReward(shared_best_cost=shared_best_cost)

    def __getstate__(self):
        """
        自定义序列化方法，以处理不可序列化的Manager对象。
        """
        state = self.__dict__.copy()
        # Manager对象不可序列化，并且在子进程中也不需要它。
        # 它创建的代理对象（如共享字典）是可序列化的。
        del state["manager"]
        return state

    def __setstate__(self, state):
        """
        自定义反序列化方法。
        """
        self.__dict__.update(state)
        # 在子进程中，管理器不会被恢复。
        self.manager = None

    def prepare(self):
        # 创建statistics缓存目录
        statistics_cache_dir = os.path.join(self.EXPERIMENT_STATISTICS_PATH, "statistics_cache")

        for idx, benchmark in enumerate(self.config["workload"]["benchmarks"]):
            logging.info(f"\n #############################{benchmark}#############################")
            logging.info(f"Preparing benchmark: {benchmark}  [{idx+1}/{len(self.config['workload']['benchmarks'])}]")
            schema = Schema(benchmark, self.config["workload"]["scale_factor"])
            database_name = benchmark
            self.config["workload"]["benchmark"] = benchmark
            workload_generator = TrainingWorkloadGenerator(
                self.config["workload"],
                schema=schema,
                random_seed=self.config["random_seed"],
                eval_mode=False,
            )

            globally_indexable_columns = workload_generator.globally_indexable_columns

            # [[single column indexes], [2-column combinations], [3-column combinations]...]
            globally_indexable_columns = utils.create_column_permutation_indexes(
                globally_indexable_columns, self.config["max_index_width"]
            )

            globally_indexable_columns_flat = [item for sublist in globally_indexable_columns for item in sublist]
            logging.info(f"Feeding {len(globally_indexable_columns_flat)} candidates into the environments.")

            action_storage_consumptions = utils.predict_index_sizes(globally_indexable_columns_flat, database_name)

            # 使用Statistics类的缓存方法
            statistic = Statistics.load_from_cache_or_create(
                statistics_cache_dir, database_name, schema.tables, schema.columns
            )

            self.database_contexts[benchmark] = DatabaseContext(
                database_name,
                schema,
                workload_generator.workloads,
                globally_indexable_columns,
                globally_indexable_columns_flat,
                statistic,
                action_storage_consumptions,
            )

        # workload embedder 只需要一个实例
        self.workload_embedder = SQLWorkloadEmbedder(
            self.config["workload"]["workload_window_size"],
            self.config["workload_embedder"]["max_columns_per_query"],
            self.config["workload_embedder"]["max_edges"],
            self.config["workload_embedder"]["sql_predicates_vec_size"],
            self.config["workload_embedder"]["sql_embedding_dim"],
        )

    def _assign_budgets_to_workloads(self):
        self.workload_generator.train_workload.budget = self.rnd.choice(
            self.config["budgets"]["validation_and_testing"]
        )

    def _pickle_workloads(self):
        with open(f"{self.experiment_folder_path}/testing_workloads.pickle", "wb") as handle:
            pickle.dump(self.workload_generator.wl_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{self.experiment_folder_path}/validation_workloads.pickle", "wb") as handle:
            pickle.dump(self.workload_generator.wl_validation, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def start_learning(self):
        self.training_start_time = datetime.datetime.now()
        logging.info(f"Training started at {self.training_start_time}")

    def set_model(self, model):
        self.model = model

    def _create_experiment_folder(self):
        assert os.path.isdir(
            self.EXPERIMENT_RESULT_PATH
        ), f"Folder for experiment results should exist at: ./{self.EXPERIMENT_RESULT_PATH}"

        self.experiment_folder_path = f"{self.EXPERIMENT_RESULT_PATH}/ID_{self.id}"
        # assert os.path.isdir(self.experiment_folder_path) == False, (
        #     f"Experiment folder already exists at: ./{self.experiment_folder_path} - "
        #     "terminating here because we don't want to overwrite anything."
        # )
        if os.path.isdir(self.experiment_folder_path) == False:
            os.mkdir(self.experiment_folder_path)
        else:
            logging.warning(f"Experiment folder already exists at: ./{self.experiment_folder_path}")

    def make_env(self, env_id, environment_type=EnvironmentType.TRAINING):
        self.id = env_id

        def _init():
            random_benchmark = self.rnd.choice(self.training_envs)
            self.database_contexts[random_benchmark].switch_database(self)
            workloads = self.database_contexts[random_benchmark].workloads
            workload = workloads[self.rnd.randint(0, len(workloads) - 1)]

            action_manager = DenseRepresentationActionManager(
                max_index_width=self.config["max_index_width"],
                action_storage_consumptions=self.action_storage_consumptions,
                columns_vec_dict=self.statistic.columns_vec_dict,
                columns_vec_len=self.statistic.columns_vec_len,
                indexable_column_combinations=self.globally_indexable_columns,
            )

            if self.number_of_actions is None:
                self.number_of_actions = action_manager.action_embedding_size

            observation_manager = EmbeddingObservationManager(
                self.config["workload_embedder"]["max_columns_per_query"],
                self.config["workload_embedder"]["max_edges"],
                self.number_of_actions,
                self.workload_embedder,
                self.statistic,
            )

            reward_calculator = self.reward_calculator

            env = gym.make(
                f"DB-v1",
                environment_type=environment_type,
                config={
                    "database_name": self.database_name,
                    "globally_indexable_columns": self.globally_indexable_columns_flat,
                    "workload": workload,
                    "random_seed": self.config["random_seed"] + env_id,
                    "max_steps_per_episode": self.config["max_steps_per_episode"],
                    "action_manager": action_manager,
                    "observation_manager": observation_manager,
                    "reward_calculator": reward_calculator,
                    "env_id": env_id,
                    "database_contexts": self.database_contexts,
                    "workload_embedder": self.workload_embedder,
                    "max_index_width": self.config["max_index_width"],
                    "max_columns_per_query": self.config["workload_embedder"]["max_columns_per_query"],
                    "max_edges": self.config["workload_embedder"]["max_edges"],
                    "training_envs": self.training_envs,
                    "eval_envs": self.eval_envs,
                    "db_lock_dir": os.path.join(self.EXPERIMENT_RESULT_PATH, f"{self.config['port']}_locks"),
                },
            )
            set_random_seed(self.config["random_seed"])
            return env

        return _init

    def make_test_env(self, env_id, benchmark_id, environment_type=EnvironmentType.TESTING, workload_idx=0, budget=0.5):
        self.id = env_id

        def _init():
            eval_benchmark = self.config["workload"]["eval_benchmarks"][benchmark_id]
            self.database_contexts[eval_benchmark].switch_database(self)
            schema = Schema(eval_benchmark, self.config["workload"]["scale_factor"])
            self.config["workload"]["benchmark"] = eval_benchmark
            workload_generator = TrainingWorkloadGenerator(
                self.config["workload"],
                schema=schema,
                random_seed=self.config["random_seed"],
                eval_mode=True,
            )
            workloads = workload_generator.workloads
            workload = workloads[workload_idx]
            workload.budget = budget
            logging.info(f"Testing workload budget: {workload.budget}")

            action_manager = DenseRepresentationActionManager(
                max_index_width=self.config["max_index_width"],
                action_storage_consumptions=self.action_storage_consumptions,
                columns_vec_dict=self.statistic.columns_vec_dict,
                columns_vec_len=self.statistic.columns_vec_len,
                indexable_column_combinations=self.globally_indexable_columns,
            )

            if self.number_of_actions is None:
                self.number_of_actions = action_manager.action_embedding_size

            observation_manager = EmbeddingObservationManager(
                self.config["workload_embedder"]["max_columns_per_query"],
                self.config["workload_embedder"]["max_edges"],
                self.number_of_actions,
                self.workload_embedder,
                self.statistic,
            )

            reward_calculator = self.reward_calculator

            env = gym.make(
                f"DB-v1",
                environment_type=environment_type,
                config={
                    "database_name": self.database_name,
                    "globally_indexable_columns": self.globally_indexable_columns_flat,
                    "workload": workload,
                    "random_seed": self.config["random_seed"] + env_id,
                    "max_steps_per_episode": self.config["max_steps_per_episode"],
                    "action_manager": action_manager,
                    "observation_manager": observation_manager,
                    "reward_calculator": reward_calculator,
                    "env_id": env_id,
                    "database_contexts": self.database_contexts,
                    "workload_embedder": self.workload_embedder,
                    "max_index_width": self.config["max_index_width"],
                    "max_columns_per_query": self.config["workload_embedder"]["max_columns_per_query"],
                    "max_edges": self.config["workload_embedder"]["max_edges"],
                    "training_envs": self.training_envs,
                    "eval_envs": self.eval_envs,
                    "db_lock_dir": os.path.join(self.EXPERIMENT_RESULT_PATH, f"{self.config['port']}_locks"),
                },
            )
            set_random_seed(self.config["random_seed"])
            return env

        return _init

    def finish_learning(self, training_env, env_id, env_type):
        self.training_end_time = datetime.datetime.now()
        logging.info(f"Training ended at {self.training_end_time}")
        self.training_duration = self.training_end_time - self.training_start_time
        self.training_duration_seconds = self.training_duration.total_seconds()
        self.training_duration_minutes = self.training_duration_seconds / 60
        logging.info(f"Training duration: {self.training_duration_minutes} minutes")

        training_env.close()
        self.model.save(f"{self.experiment_folder_path}/model_{env_id}_{env_type}")
        self.reward_calculator.save_best_cost_so_far(f"{self.experiment_folder_path}/best_cost_so_far_{env_id}_{env_type}.txt")

    # Start of Selection
    def _load_pretrained_model(self, model_path, device="auto", load_best_cost_so_far=False):
        logging.info(f"Loading pretrained model from {model_path}")
        # 兼容新版 PyTorch 的加载方式, 移除 weights_only 以兼容旧版 torch
        old_model = PPO.load(model_path, device=device)
        self.model.set_parameters(old_model.get_parameters())
        logging.info(f"Pretrained model loaded 👌👌👌")
        #  /home/weixun/data/Index_Recommendation_ZS/exp_result/ID_All_Database_15/model_0_20250904_1636.zip
        # 0 as env_id, 20250904_1636 as env_type
        if load_best_cost_so_far:
            model_filename = model_path.split("/")[-1]  # model_0_20250904_1636.zip
            model_parts = model_filename.split("_")     # ['model', '0', '20250904', '1636.zip']
            env_id = model_parts[1]                     # '0'
            env_type = model_parts[2] + "_" + model_parts[3].split(".")[0]  # '20250904_1636'
            self.reward_calculator.load_best_cost_so_far(f"{self.experiment_folder_path}/best_cost_so_far_{env_id}_{env_type}.txt")
            logging.info(f"Best cost so far loaded 👌👌👌")
        return True
