import copy
import logging

import numpy as np
from gymnasium import spaces

from selection.utils import b_to_mb
from solution.utils import sub_index_of, index_have_same_columns
from .statistics import Statistics

FORBIDDEN_ACTION_SB3 = -np.inf
ALLOWED_ACTION_SB3 = 1



class ActionManager(object):
    def __init__(self, max_index_width):
        self.valid_actions = []
        self._remaining_valid_actions = []
        self.number_of_actions = None
        self.action_embedding_size = None
        self.current_action_status = None

        self.test_variable = None

        self.MAX_INDEX_WIDTH = max_index_width

        self.FORBIDDEN_ACTION = FORBIDDEN_ACTION_SB3
        self.ALLOWED_ACTION = ALLOWED_ACTION_SB3

    def get_action_space(self):
        return spaces.Discrete(self.action_embedding_size)

    def get_initial_valid_actions(self, workload, budget):
        self.current_action_status = [0 for action in range(self.number_of_columns)]

        self.valid_actions = [self.FORBIDDEN_ACTION for action in range(self.number_of_actions)]
        self._remaining_valid_actions = []

        self._valid_actions_based_on_workload(workload)
        self._valid_actions_based_on_budget(budget, current_storage_consumption=0)

        self.current_combinations = set()

        return np.array(self.valid_actions)

    def update_valid_actions(self, last_action, budget, current_storage_consumption):
        assert self.indexable_column_combinations_flat[last_action] not in self.current_combinations

        actions_index_width = len(self.indexable_column_combinations_flat[last_action])
        if actions_index_width == 1:
            self.current_action_status[last_action] += 1
        else:
            combination_to_be_extended = self.indexable_column_combinations_flat[last_action][:-1]
            assert combination_to_be_extended in self.current_combinations

            status_value = 1 / actions_index_width

            last_action_back_column = self.indexable_column_combinations_flat[last_action][-1]
            last_action_back_columns_idx = self.column_to_idx[last_action_back_column]
            self.current_action_status[last_action_back_columns_idx] += status_value

            self.current_combinations.remove(combination_to_be_extended)

        self.current_combinations.add(self.indexable_column_combinations_flat[last_action])

        self.valid_actions[last_action] = self.FORBIDDEN_ACTION
        self._remaining_valid_actions.remove(last_action)

        self._valid_actions_based_on_last_action(last_action)
        remain_budget = budget - self.action_storage_consumptions[last_action]
        self._valid_actions_based_on_budget(remain_budget, current_storage_consumption)

        is_valid_action_left = len(self._remaining_valid_actions) > 0

        return np.array(self.valid_actions), is_valid_action_left

    def _valid_actions_based_on_budget(self, budget, current_storage_consumption):
        if budget is None:
            return
        else:
            new_remaining_actions = []
            for action_idx in self._remaining_valid_actions:
                if current_storage_consumption + self.action_storage_consumptions[action_idx] > budget:
                    self.valid_actions[action_idx] = self.FORBIDDEN_ACTION
                else:
                    new_remaining_actions.append(action_idx)

            self._remaining_valid_actions = new_remaining_actions

    def _valid_actions_based_on_workload(self, workload):
        raise NotImplementedError

    def _valid_actions_based_on_last_action(self, last_action):
        raise NotImplementedError


class DenseRepresentationActionManager(ActionManager):
    def __init__(
        self, max_index_width, action_storage_consumptions, columns_vec_dict, columns_vec_len, indexable_column_combinations,
        temperature=1.0
    ):
        ActionManager.__init__(self, max_index_width)
        self.action_storage_consumptions = action_storage_consumptions
        self.columns_vec_dict = columns_vec_dict
        self.indexable_column_combinations = indexable_column_combinations
        self.number_of_columns = len(indexable_column_combinations[0])
        self.column_embeddings = [
            self.columns_vec_dict[column[0].table.name + "." + column[0].name]
            for column in self.indexable_column_combinations[0]
        ]
        self.column_to_idx = {column[0]: idx for idx, column in enumerate(self.indexable_column_combinations[0])}

        self.indexable_column_combinations_flat = [
            item for sublist in self.indexable_column_combinations for item in sublist
        ]
        self.number_of_actions = len(self.indexable_column_combinations_flat)
        self.action_embedding_size = columns_vec_len
        self.current_action_embedding = np.zeros(self.action_embedding_size)

        self.STOP_ACTION_INDEX = -1
        self.stop_action_embedding = np.zeros(self.action_embedding_size)
        self.temperature = temperature

        # 维护一个全局的 action
        self.valid_actions = [self.FORBIDDEN_ACTION for action in range(len(self.indexable_column_combinations_flat))]
        self._remaining_valid_actions = []

    def _multi_column_index_action_embedding(self, indexable_column):
        assert len(indexable_column) > 1
        index_len = len(indexable_column)
        # return average of all columns vectors
        return np.mean(
            [self.columns_vec_dict[column.table.name + "." + column.name] for column in indexable_column], axis=0
        )

    def get_action_embeddings_for_valid_actions(self):
        """获取所有有效动作的嵌入向量"""
        action_embeddings = {}
        for idx, indexable_column in enumerate(self.indexable_column_combinations_flat):
            if self.valid_actions[idx] == self.ALLOWED_ACTION:
                if len(indexable_column) == 1:
                    action_embeddings[idx] = self.columns_vec_dict[
                        indexable_column[0].table.name + "." + indexable_column[0].name
                    ]
                else:
                    action_embeddings[idx] = self._multi_column_index_action_embedding(indexable_column)
        return action_embeddings

    def map_continuous_action_to_discrete(self, continuous_action, temperature):
        self.temperature = temperature
        action_embeddings = self.get_action_embeddings_for_valid_actions()
        if not action_embeddings:
            return self.STOP_ACTION_INDEX, 1.0, self.stop_action_embedding
        # Add the stop action for similarity comparison
        action_embeddings[self.STOP_ACTION_INDEX] = self.stop_action_embedding

        target_vector = np.array(continuous_action)
        target_norm = np.linalg.norm(target_vector)

        if target_norm == 0:
            return self.STOP_ACTION_INDEX, 1.0, self.stop_action_embedding


        candidate_actions = list(action_embeddings.keys())
        candidate_embeddings = list(action_embeddings.values())

        similarities = []
        for action_vector in candidate_embeddings:
            action_norm = np.linalg.norm(action_vector)
            if action_norm == 0:
                similarity = 0.0
            else:
                similarity = np.dot(target_vector, action_vector) / (target_norm * action_norm)
            similarities.append(similarity)

        similarities = np.array(similarities)
        if self.temperature <= 0:
            probs = np.zeros_like(similarities)
            probs[np.argmax(similarities)] = 1.0
        else:
            probs = np.exp(similarities / self.temperature)
            probs /= probs.sum()

        chosen_index = np.random.choice(len(candidate_actions), p=probs)
        chosen_action_embedding = candidate_embeddings[chosen_index]
        best_action = candidate_actions[chosen_index]
        best_similarity = similarities[chosen_index]

        return best_action, best_similarity, chosen_action_embedding

    def get_action_space(self):
        return spaces.Box(low=0, high=1, shape=(self.action_embedding_size,), dtype=np.float32)

    def _valid_actions_based_on_workload(self, workload):
        remaining_valid_actions_set = set()
        for indexable_column in workload.indexable_columns(return_sorted=False):
            for column_combination_idx, indexable_column_combination in enumerate(self.indexable_column_combinations_flat):
                    if sub_index_of(indexable_column, indexable_column_combination):
                        self.valid_actions[column_combination_idx] = self.ALLOWED_ACTION
                        remaining_valid_actions_set.add(column_combination_idx)
        self._remaining_valid_actions = list(remaining_valid_actions_set)

    def _valid_actions_based_on_last_action(self, last_action):
        last_combination = self.indexable_column_combinations_flat[last_action]
        for column_combination_idx in copy.copy(self._remaining_valid_actions):
            if index_have_same_columns(last_combination, self.indexable_column_combinations_flat[column_combination_idx]):
                self._remaining_valid_actions.remove(column_combination_idx)
                self.valid_actions[column_combination_idx] = self.FORBIDDEN_ACTION


    def _get_current_action_embedding(self):
        status_array = np.array(self.current_action_status).reshape(-1, 1)
        embeddings_array = np.array(self.column_embeddings)
        self.current_action_embedding = np.sum(status_array * embeddings_array, axis=0)
        return self.current_action_embedding

    def update_valid_actions(self, last_action, avaliavle_budget, current_storage_consumption):
        last_action_index = last_action
        last_action_column = self.indexable_column_combinations_flat[last_action_index]

        for idx, column in enumerate(last_action_column):
            if idx == 0:
                self.current_action_status[self.column_to_idx[column]] += 1
            else:
                self.current_action_status[self.column_to_idx[column]] += 1/len(last_action_column)

        self.current_action_embedding = self._get_current_action_embedding()

        self.valid_actions[last_action_index] = self.FORBIDDEN_ACTION
        self._remaining_valid_actions.remove(last_action_index)

        self._valid_actions_based_on_last_action(last_action_index)
        remain_budget = avaliavle_budget - self.action_storage_consumptions[last_action_index]
        self._valid_actions_based_on_budget(remain_budget, current_storage_consumption)

        is_valid_action_left = len(self._remaining_valid_actions) > 0

        return np.array(self.valid_actions), is_valid_action_left

