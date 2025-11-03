from selection.utils import b_to_mb
import numpy as np

from selection.utils import log_real

from collections import defaultdict


class RewardCalculator(object):
    def __init__(self):
        self.reset(0)
        self.accumulated_reward = 0
        self.current_cost = 0
        self.previous_cost = 0
        self.initial_cost = 0
        self.new_index_size = 0
        self.storage_consumption = 0
        self.stop_action = -1
        self.current_budget = 0
        self.best_cost_so_far = defaultdict(lambda: float("inf"))
        self.database_name = None

    def reset(self):
        self.accumulated_reward = 0

    def calculate_reward(self, environment_state):
        self.database_name = environment_state["database_name"]
        self.current_cost = environment_state["current_cost"]
        self.previous_cost = environment_state["previous_cost"]
        self.initial_cost = environment_state["initial_cost"]
        self.new_index_size = environment_state["new_index_size"]
        self.storage_consumption = environment_state["current_storage_consumption"]
        self.stop_action = environment_state["stop_action"]
        self.current_budget = b_to_mb(environment_state["current_budget"])
        self.workload_idx = environment_state["current_workload"].workload_idx
        self.number_of_reset = environment_state["number_of_reset"]

        self.workload_alias = self.database_name + f"_{self.current_budget:.2f}_{self.workload_idx}"

        assert self.new_index_size is not None
        reward = self._calculate_reward()

        self.accumulated_reward += reward

        return reward

    def _calculate_reward(
        self,
    ):
        raise NotImplementedError


class NormalizedIndexReward(RewardCalculator):

    def __init__(
        self,
        shared_best_cost=None,
    ):
        RewardCalculator.__init__(self)

        if shared_best_cost is not None:
            self.best_cost_so_far = shared_best_cost
        else:
            self.best_cost_so_far = defaultdict(lambda: float("inf"))

    def reset(self, db_size):
        """重置奖励计算器状态"""
        super().reset()

    def _calculate_reward(self):

        if self.stop_action == -1:
            return self._calculate_log_terminal(self.current_cost, self.initial_cost)
            # return self._calculate_terminal_reward(
            #     self.current_cost, self.initial_cost, self.storage_consumption, self.current_budget
            # )

        # reward = self._calculate_improvement_component(self.current_cost, self.previous_cost, self.initial_cost)
        reward = self._calculate_log_improvement(self.current_cost, self.previous_cost)

        return reward

    def _calculate_improvement_component(self, current_cost, previous_cost, initial_cost):

        if current_cost >= previous_cost:
            return -0.1

        best_cost_so_far = self.best_cost_so_far.get(self.workload_alias, float("inf"))

        known_potential = initial_cost - best_cost_so_far

        denominator = known_potential if known_potential > (initial_cost * 0.001) else initial_cost

        improvement = (previous_cost - current_cost) / denominator

        return improvement

    def _calculate_log_improvement(self, current_cost, previous_cost):
        return log_real(previous_cost - current_cost)

    def _calculate_log_terminal(self, final_cost, initial_cost):
        return log_real(initial_cost - final_cost)

    def _calculate_terminal_reward(self, final_cost, initial_cost):

        if final_cost >= initial_cost:
            return -0.5

        best_cost_so_far = self.best_cost_so_far.get(self.workload_alias, float("inf"))

        known_potential = initial_cost - best_cost_so_far

        denominator = known_potential if known_potential > (initial_cost * 0.001) else initial_cost

        improvement_ratio = (initial_cost - final_cost) / denominator

        return improvement_ratio

    def save_best_cost_so_far(self, save_path):
        with open(save_path, "w") as f:
            for workload_alias, best_cost_so_far in dict(self.best_cost_so_far).items():
                f.write(f"{workload_alias}: {best_cost_so_far}\n")

    def load_best_cost_so_far(self, load_path):
        with open(load_path, "r") as f:
            for line in f:
                workload_alias, best_cost_so_far = line.strip().split(":")
                self.best_cost_so_far[workload_alias] = float(best_cost_so_far)


