import logging

import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box, Graph, Discrete

from solution.statistics import Statistics
from solution.workload_embedder import SQLWorkloadEmbedder
from selection.utils import b_to_mb, mb_to_b, log_real


VERY_HIGH_BUDGET = 100_000_000_000


class ObservationManager(object):
    def __init__(self, number_of_actions):
        self.number_of_actions = number_of_actions

    def _init_episode(self, state_fix_for_episode):
        self.episode_budget = state_fix_for_episode["budget"]
        if self.episode_budget is None:
            self.episode_budget = VERY_HIGH_BUDGET

        self.initial_cost = state_fix_for_episode["initial_cost"]


    def get_observation_space(self):
        observation_space = spaces.Box(
            low=self._create_low_boundaries(), high=self._create_high_boundaries(), shape=self._create_shape()
        )

        logging.info(f"Creating ObservationSpace with {self.number_of_features} features.")

        return observation_space

    def _create_shape(self):
        return (self.number_of_features,)

    def _create_low_boundaries(self):
        low = [-np.inf for feature in range(self.number_of_features)]

        return np.array(low)

    def _create_high_boundaries(self):
        high = [np.inf for feature in range(self.number_of_features)]

        return np.array(high)

    def init_episode(self, state_fix_for_episode):
        raise NotImplementedError

    def get_observation(self, environment_state):
        raise NotImplementedError

    @staticmethod
    def _get_frequencies_from_workload(workload):
        frequencies = []
        for query in workload.queries:
            frequencies.append(query.frequency)
        return frequencies


class EmbeddingObservationManager(ObservationManager):

    def __init__(
        self,
        max_columns_per_query,
        max_edges,
        number_of_actions,
        workload_embedder: SQLWorkloadEmbedder,
        statistic: Statistics,
    ):
        super().__init__(number_of_actions)

        self.workload_embedder = workload_embedder
        self.statistic = statistic
        self.number_of_features = self.number_of_actions + self.workload_embedder.embedding_dim

        self.number_of_meta = workload_embedder.workload_window_size + 5
        self.number_of_action = number_of_actions

        self.UPDATE_EMBEDDING_PER_OBSERVATION = False

        self.query_window_size = workload_embedder.workload_window_size
        self.max_columns_per_query = max_columns_per_query
        self.max_edges = max_edges

    def init_episode(self, state_fix_for_episode):
        super()._init_episode(state_fix_for_episode)
        if isinstance(state_fix_for_episode["workload"], list):
            episode_workload = state_fix_for_episode["workload"][0]  # 使用第一个工作负载
        else:
            episode_workload = state_fix_for_episode["workload"]

        self.frequencies = np.array(EmbeddingObservationManager._get_frequencies_from_workload(episode_workload))

        self.workload_embedding = self.workload_embedder.get_embeddings(episode_workload, self.statistic)

    def init_observation(self, state_fix_for_episode):
        self._init_episode(state_fix_for_episode)
        self.workload_embedding = self.workload_embedder.get_embeddings(
            state_fix_for_episode["workload"], self.statistic
        )

    def get_observation_space(self):
        return spaces.Dict(
            {
                "wl_query_embeddings": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.query_window_size, self.max_columns_per_query, self.workload_embedder.embedding_dim),
                    dtype=np.float32,
                ),
                "wl_query_maskings": Box(
                    low=0, high=1 + 1e-8, shape=(self.query_window_size, self.max_columns_per_query), dtype=np.int32
                ),
                "wl_edge_indexes": Box(
                    low=0,
                    high=self.max_columns_per_query - 1 + 1e-8,
                    shape=(self.query_window_size, 2, self.max_edges),
                    dtype=np.int32,
                ),
                "wl_edge_weights": Box(
                    low=0, high=1 + 1e-8, shape=(self.query_window_size, self.max_edges), dtype=np.int32
                ),
                "wl_edge_maskings": Box(
                    low=0, high=1 + 1e-8, shape=(self.query_window_size, self.max_edges), dtype=np.int32
                ),
                "meta_info_db": Box(low=-np.inf, high=np.inf, shape=(self.number_of_meta,), dtype=np.float32),
                "action": Box(low=-np.inf, high=np.inf, shape=(self.number_of_action,), dtype=np.float32),
            }
        )

    def get_observation(self, environment_state):
        if self.UPDATE_EMBEDDING_PER_OBSERVATION:
            self.workload_embedding = self.workload_embedder.get_embeddings(
                environment_state["current_workload"], self.statistic
            )
        meta_info_db = np.array(
            [
                log_real(b_to_mb(environment_state["current_storage_consumption"])),
                log_real(b_to_mb(environment_state["db_size"])),
                log_real(b_to_mb(environment_state["current_budget"])),
                log_real(environment_state["current_cost"]),
                log_real(environment_state["initial_cost"]),
            ],
            dtype=np.float32,
        )

        cost_per_query = log_real(np.array(environment_state["costs_per_query"], dtype=np.float32))
        meta_info_db = np.concatenate([meta_info_db, cost_per_query], axis=0)

        action = np.array(environment_state["action_status"], dtype=np.float32)

        return {
            "wl_query_embeddings": self.workload_embedding["embeddings"],
            "wl_query_maskings": self.workload_embedding["query_maskings"],
            "wl_edge_indexes": self.workload_embedding["edge_indexes"],
            "wl_edge_weights": self.workload_embedding["edge_weights"],
            "wl_edge_maskings": self.workload_embedding["edge_maskings"],
            "meta_info_db": meta_info_db,
            "action": action,
        }

    @staticmethod
    def _get_frequencies_from_workload(workload):
        frequencies = []
        for query in workload.queries:
            frequencies.append(query.frequency)
        return frequencies
