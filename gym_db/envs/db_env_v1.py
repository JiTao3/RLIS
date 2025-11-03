import collections
import copy
import logging
import random
import os
import time

import numpy as np

import gymnasium as gym

from gym_db.common import EnvironmentType
from solution.action_manager import DenseRepresentationActionManager
from solution.observation_manager import EmbeddingObservationManager
from selection.cost_evaluation import CostEvaluation
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.index import Index
from selection.utils import b_to_mb, mb_to_b

from solution.file_lock import SerializableLockDict, create_db_locks


class DBEnvV1(gym.Env):
    def __init__(self, environment_type=EnvironmentType.TRAINING, config=None):
        super().__init__()

        if config is None:
            raise ValueError("Config parameter cannot be None")

        self.rnd = random.Random()
        self.rnd.seed(config["random_seed"])
        self.env_id = config["env_id"]
        self.environment_type = environment_type
        self.config = config

        self.number_of_resets = 0
        self.total_number_of_steps = 0

        self.current_database_name = None
        self.connector = None
        self.cost_evaluation = None
        self._process_id = os.getpid()

        self.globally_indexable_columns = config["globally_indexable_columns"]
        self.workload = copy.copy(config["workload"])
        self.current_workload_idx = 0
        self.max_steps_per_episode = config["max_steps_per_episode"]

        self.action_manager = config["action_manager"]
        self.action_space = self.action_manager.get_action_space()

        self.observation_manager = config["observation_manager"]
        self.observation_space = self.observation_manager.get_observation_space()

        self.reward_calculator = config["reward_calculator"]

        self.database_contexts = config["database_contexts"]
        self.workload_embedder = config["workload_embedder"]

        self.training_envs = config["training_envs"]
        self.eval_envs = config["eval_envs"]

        self.database_name = self.workload.database_name

        self.db_lock = create_db_locks(
            database_names=list(self.database_contexts.keys()), lock_dir=config["db_lock_dir"], timeout=30.0
        )

        self.temperature = 1.0

        self._init_modifiable_state(init=True)

    def set_temperature(self, temperature):
        self.temperature = temperature

    def _ensure_database_connection(self, database_name=None):
        current_pid = os.getpid()
        target_db = database_name or self.database_name

        need_reconnect = (
            self.current_database_name != target_db or self.connector is None or self._process_id != current_pid
        )

        if need_reconnect:
            try:
                if self.connector:
                    self.connector.close()
                    logging.debug(f"Closed connection to database: {self.current_database_name}")

                self.connector = PostgresDatabaseConnector(target_db, autocommit=True)
                self.connector.drop_indexes(drop_consistent=False)
                self.cost_evaluation = CostEvaluation(self.connector)
                self.current_database_name = target_db
                self._process_id = current_pid

                logging.info(f"Database connection established for {target_db} in process {current_pid}")

            except Exception as e:
                logging.error(f"Failed to create database connection for {target_db}: {e}")
                raise

    def __getstate__(self):
        state = self.__dict__.copy()
        state["connector"] = None
        state["cost_evaluation"] = None
        state["db_lock"] = None
        state["_process_id"] = os.getpid()
        logging.debug(f"Pickling env {state.get('env_id', 'unknown')} from process {os.getpid()}")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.connector = None
        self.cost_evaluation = None
        self.db_lock = None
        current_pid = os.getpid()
        logging.debug(f"Unpickling env {self.env_id} in process {current_pid}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rnd.seed(seed)

        self.number_of_resets += 1
        self.total_number_of_steps += self.steps_taken

        initial_observation = self._init_modifiable_state()
        self.reward_calculator.reset(self.db_size)

        info = {}
        return initial_observation, info


    def _step_asserts(self, action):
        assert (
            self.valid_actions[action] == self.action_manager.ALLOWED_ACTION
        ), f"Agent has chosen invalid action: {action}"
        assert (
            Index(self.globally_indexable_columns[action]) not in self.current_indexes
        ), f"{Index(self.globally_indexable_columns[action])} already in self.current_indexes"

    def step(self, action):
        self._ensure_database_connection()

        discrete_action, _, chosen_action_embedding = self.action_manager.map_continuous_action_to_discrete(
            action, self.temperature
        )

        if discrete_action == self.action_manager.STOP_ACTION_INDEX:
            environment_state = self._update_return_env_state(init=False)
            environment_state["stop_action"] = -1
            current_observation = self.observation_manager.get_observation(environment_state)
            reward = self.reward_calculator.calculate_reward(environment_state)
            terminated = True
            truncated = False
            infos = {
                "action_mask": self.valid_actions,
                "achieved_cost_ratio": self.current_costs / self.initial_costs * 100,
                "database_name": self.current_database_name,
                "budget": b_to_mb(self.current_budget),
                "is_stop_action": True,
                "chosen_action_embedding": chosen_action_embedding,
            }
            self._safe_lock_release(self.current_database_name)
            return current_observation, reward, terminated, truncated, infos

        if discrete_action is None:
            environment_state = self._update_return_env_state(init=False)
            current_observation = self.observation_manager.get_observation(environment_state)
            reward = self.reward_calculator.calculate_reward(environment_state)

            terminated = True
            truncated = False
            infos = {
                "action_mask": self.valid_actions,
                "achieved_cost_ratio": self.current_costs / self.initial_costs * 100,
                "database_name": self.current_database_name,
                "budget": b_to_mb(self.current_budget),
                "chosen_action_embedding": chosen_action_embedding,
            }
            self._safe_lock_release(self.current_database_name)
            return current_observation, reward, terminated, truncated, infos

        self._step_asserts(discrete_action)

        self.steps_taken += 1
        old_index_size = 0

        new_index = Index(self.globally_indexable_columns[discrete_action])
        self.current_indexes.add(new_index)

        available_budget = self.current_budget - self.current_storage_consumption

        self.valid_actions, is_valid_action_left = self.action_manager.update_valid_actions(
            discrete_action, available_budget, self.current_storage_consumption
        )

        environment_state = self._update_return_env_state(
            init=False, new_index=new_index, old_index_size=old_index_size
        )
        current_observation = self.observation_manager.get_observation(environment_state)

        episode_done = self.steps_taken >= self.max_steps_per_episode or not is_valid_action_left

        if episode_done:
            environment_state["stop_action"] = -1

        reward = self.reward_calculator.calculate_reward(environment_state)

        self.current_workload_idx += 1

        terminated = episode_done
        truncated = False
        infos = {
            "action_mask": self.valid_actions,
            "achieved_cost_ratio": self.current_costs / self.initial_costs * 100,
            "database_name": self.current_database_name,
            "budget": b_to_mb(self.current_budget),
            "chosen_action_embedding": chosen_action_embedding,
        }

        if terminated:
            self._safe_lock_release(self.current_database_name)

        return current_observation, reward, terminated, truncated, infos

    def _report_episode_performance(self, environment_state):
        episode_performance = {
            "achieved_cost": self.current_costs / self.initial_costs * 100,
            "memory_consumption": self.current_storage_consumption,
            "available_budget": self.current_budget - self.current_storage_consumption,
            "evaluated_workload": self.current_workload,
            "indexes": self.current_indexes,
        }

        output = (
            f"Evaluated Workload ({self.environment_type}): {self.current_workload}\n    "
            f"Initial cost: {self.initial_costs:,.2f}, now: {self.current_costs:,.2f} "
            f"({episode_performance['achieved_cost']:.2f}). Reward: {self.reward_calculator.accumulated_reward}.\n    "
            f"Size: {b_to_mb(self.current_storage_consumption):.2f} with {len(self.current_indexes)} indexes:\n    "
            f"{self.current_indexes}\n    "
        )
        logging.warning(output)


    def _init_modifiable_state(self, init=False):
        if self.environment_type == EnvironmentType.TRAINING and not init:
            self.action_manager, self.observation_manager = self._switch_database_context()

            self.action_space = self.action_manager.get_action_space()
            self.observation_space = self.observation_manager.get_observation_space()

        if self.environment_type == EnvironmentType.TESTING:
            self.database_name = self.workload.database_name
            self._ensure_database_connection(database_name=self.database_name)

        self.db_size = self.workload.db_size
        self.current_indexes = set()
        self.steps_taken = 0
        self.current_storage_consumption = 0

        self.current_workload = self.workload

        self.current_budget = self.current_workload.budget * mb_to_b(self.db_size)
        self.previous_cost = None

        self.valid_actions = self.action_manager.get_initial_valid_actions(self.current_workload, self.current_budget)
        environment_state = self._update_return_env_state(init=True)

        state_fix_for_episode = {
            "budget": self.current_budget,
            "workload": self.current_workload,
            "initial_cost": self.initial_costs,
        }
        self.observation_manager.init_episode(state_fix_for_episode)

        initial_observation = self.observation_manager.get_observation(environment_state)

        return initial_observation

    def random_select_database(self):

        wait_timeout = 30.0
        check_interval = 0.5
        start_time = time.time()

        while True:
            available_dbs = []
            locked_dbs = []

            for db_name in self.training_envs:
                db_lock = self._get_db_lock(db_name)
                if db_lock is None or not db_lock.locked():
                    available_dbs.append(db_name)
                else:
                    locked_dbs.append(db_name)

            if available_dbs:
                random_db_name = self.rnd.choice(available_dbs)
                break

            elapsed_time = time.time() - start_time
            if elapsed_time >= wait_timeout:
                raise Exception("Timeout waiting for available database")

            time.sleep(check_interval)

        return random_db_name

    def _switch_database_context(self):

        random_db_name = self.random_select_database()

        self._safe_lock_acquire(random_db_name)

        database_context = self.database_contexts[random_db_name]
        self.workload = database_context.workloads[self.rnd.randint(0, len(database_context.workloads) - 1)]
        self.database_name = database_context.database_name

        self.globally_indexable_columns = database_context.globally_indexable_columns_flat

        statistic = database_context.statistic
        action_storage_consumptions = database_context.action_storage_consumptions

        action_manager = DenseRepresentationActionManager(
            max_index_width=self.config["max_index_width"],
            action_storage_consumptions=action_storage_consumptions,
            columns_vec_dict=statistic.columns_vec_dict,
            columns_vec_len=statistic.columns_vec_len,
            indexable_column_combinations=database_context.globally_indexable_columns,
            temperature=self.temperature,
        )
        number_of_actions = action_manager.action_embedding_size

        observation_manager = EmbeddingObservationManager(
            self.config["max_columns_per_query"],
            self.config["max_edges"],
            number_of_actions,
            self.workload_embedder,
            statistic,
        )

        self._ensure_database_connection(database_name=self.database_name)

        return action_manager, observation_manager

    def _update_return_env_state(self, init, new_index=None, old_index_size=None):
        self._ensure_database_connection()

        assert self.cost_evaluation is not None, "Cost evaluation should be initialized"
        total_costs, costs_per_query = self.cost_evaluation.calculate_cost(
            self.current_workload, self.current_indexes, store_size=True
        )

        if not init:
            self.previous_cost = self.current_costs
            self.previous_storage_consumption = self.current_storage_consumption
        else:
            logging.info(f"Initial cost: {total_costs}")
            self.initial_costs = total_costs
            new_index_size = 0

        self.current_costs = total_costs

        if new_index is not None:
            self.current_storage_consumption += new_index.estimated_size
            self.current_storage_consumption -= old_index_size

            new_index_size = new_index.estimated_size - old_index_size
            if new_index_size == 0:
                new_index_size = 1
        else:
            new_index_size = 0

        environment_state = {
            "action_status": self.action_manager.current_action_embedding,
            "current_storage_consumption": self.current_storage_consumption,
            "current_cost": self.current_costs,
            "previous_cost": self.previous_cost,
            "initial_cost": self.initial_costs,
            "new_index_size": new_index_size,
            "current_workload": self.current_workload,
            "costs_per_query": costs_per_query,
            "workload_index": self.current_workload_idx,
            "db_size": self.db_size,
            "current_budget": self.current_budget,
            "stop_action": 1,
            "database_name": self.database_name,
            "number_of_reset": self.number_of_resets,
        }

        return environment_state

    def get_cost_eval_cache_info(self):
        self._ensure_database_connection()
        assert self.cost_evaluation is not None, "Cost evaluation should be initialized"
        return self.cost_evaluation.cost_requests, self.cost_evaluation.cache_hits, self.cost_evaluation.costing_time

    def get_cost_eval_cache(self):
        self._ensure_database_connection()
        assert self.cost_evaluation is not None, "Cost evaluation should be initialized"
        return self.cost_evaluation.cache

    def render(self, mode="human"):
        print("render() was called")
        pass

    def close(self):
        print("close() was called")
        if self.connector is not None:
            try:
                self.connector.close()
                logging.debug(f"Database connection closed for env {self.env_id}")
            except Exception as e:
                logging.warning(f"Error closing database connection for env {self.env_id}: {e}")
            finally:
                self.connector = None
                self.cost_evaluation = None

    def _get_db_lock(self, database_name):
        if hasattr(self, "db_lock") and self.db_lock and database_name in self.db_lock:
            return self.db_lock[database_name]
        else:
            logging.debug(f"Database lock not available for {database_name} in process {os.getpid()}")
            return None

    def _safe_lock_acquire(self, database_name):
        db_lock = self._get_db_lock(database_name)
        if db_lock:
            db_lock.acquire()
            logging.debug(f"Acquired lock for database: {database_name}")
        else:
            logging.debug(f"No lock to acquire for database: {database_name}")

    def _safe_lock_release(self, database_name):
        db_lock = self._get_db_lock(database_name)
        if db_lock:
            try:
                db_lock.release()
                logging.debug(f"Released lock for database: {database_name}")
            except Exception as e:
                logging.warning(f"Failed to release lock for database {database_name}: {e}")
        else:
            logging.debug(f"No lock to release for database: {database_name}")
