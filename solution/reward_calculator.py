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
            # 将共享字典转换为普通字典以便安全迭代
            for workload_alias, best_cost_so_far in dict(self.best_cost_so_far).items():
                f.write(f"{workload_alias}: {best_cost_so_far}\n")

    def load_best_cost_so_far(self, load_path):
        with open(load_path, "r") as f:
            for line in f:
                workload_alias, best_cost_so_far = line.strip().split(":")
                self.best_cost_so_far[workload_alias] = float(best_cost_so_far)


# 保留原有的IndexReward类作为备选
class IndexReward(RewardCalculator):

    def __init__(
        self,
        performance_weight=0.45,
        storage_weight=0.4,
        similarity_weight=0.05,
        stop_weight=0.1,
    ):
        RewardCalculator.__init__(self)

        # 权重配置
        self.performance_weight = performance_weight
        self.storage_weight = storage_weight
        self.similarity_weight = similarity_weight
        self.stop_weight = stop_weight

        self.best_cost_so_far = float("inf")
        self.db_size = 0

    def reset(self, db_size):
        self.best_cost_so_far = float("inf")
        self.db_size = db_size
        self.accumulated_reward = 0

    def _calculate_reward_with_storage_consumption(
        self,
        current_cost,
        previous_cost,
        initial_cost,
        new_index_size,
        similarity_score,
        storage_consumption,
        stop_action,
    ):
        # 1. 性能提升奖励 (Performance Improvement Reward)
        performance_reward = self._calculate_performance_reward(current_cost, previous_cost)

        # 2. 存储效率奖励 (Storage Efficiency Reward)
        storage_reward = self._calculate_storage_reward(current_cost, previous_cost, new_index_size)

        # 3. 语义相似度奖励 (Semantic Similarity Reward)
        similarity_reward = self._calculate_similarity_reward(similarity_score)

        if stop_action == -1:
            stop_reward = log_real(200)
        else:
            stop_reward = 0.0

        # cost_ratio_reward = self._calculate_cost_ratio_reward(current_cost, initial_cost)

        # 加权组合
        total_reward = (
            self.performance_weight * performance_reward
            + self.storage_weight * storage_reward
            + self.similarity_weight * similarity_reward
            + self.stop_weight * stop_reward
            # + cost_ratio_reward * 5.0
        )

        # 更新最佳成本
        self.best_cost_so_far = min(self.best_cost_so_far, current_cost)

        return total_reward

    def _calculate_performance_reward(self, current_cost, initial_cost):
        """性能提升奖励：基于查询成本的降低"""
        if current_cost >= initial_cost:
            # 性能没有提升，给予负奖励
            cost_increase = current_cost - initial_cost
            if cost_increase < 1:
                cost_increase = 100
            return -min(log_real(cost_increase) * 2, 10.0)  # 限制负奖励的最大值

        # 性能提升，给予正奖励
        cost_reduction = initial_cost - current_cost
        return min(log_real(cost_reduction), 10.0)  # 限制正奖励的最大值

    def _calculate_storage_reward(self, current_cost, previous_cost, new_index_size):
        """存储效率奖励：考虑性能提升与存储成本的比值"""
        if current_cost >= previous_cost:
            # 没有性能提升，存储奖励为负
            return -5.0

        # 计算性能提升与存储成本的比值
        performance_gain = previous_cost - current_cost
        storage_cost_mb = b_to_mb(new_index_size)

        if storage_cost_mb == 0:
            return 0.0

        efficiency_ratio = performance_gain / storage_cost_mb

        # 使用对数函数平滑奖励，避免极端值
        return log_real(efficiency_ratio)

    def _calculate_cost_ratio_reward(self, current_cost, initial_cost):
        if initial_cost == 0:
            return 0
        return 1 - current_cost / initial_cost  # [0, 1]

    def _calculate_storage_penalty(self, current_storage, storage_budget):
        pass

    def _calculate_similarity_reward(self, similarity_score):
        """语义相似度奖励：鼓励选择语义相关的索引"""
        if similarity_score is None:
            return 0.0

        if similarity_score < 0.5:
            return -3.0
        else:
            return 1.0


# 保留原有的奖励函数作为备选
class AbsoluteDifferenceRelativeToStorageReward(RewardCalculator):
    def __init__(self):
        RewardCalculator.__init__(self)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size, similarity_score):
        if current_cost >= previous_cost:
            return -0.5

        reward = (previous_cost - current_cost) / new_index_size
        return reward


class AbsoluteDifferenceToPreviousReward(RewardCalculator):
    def __init__(self):
        RewardCalculator.__init__(self)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size, similarity_score):
        reward = previous_cost - current_cost
        return reward


class RelativeDifferenceToPreviousReward(RewardCalculator):
    def __init__(self):
        RewardCalculator.__init__(self)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size, similarity_score):
        reward = (previous_cost - current_cost) / initial_cost
        return reward


class RelativeDifferenceRelativeToStorageReward(RewardCalculator):
    def __init__(self, scaler=1):
        RewardCalculator.__init__(self)
        self.SCALER = scaler

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size, similarity_score):
        assert new_index_size > 0
        reward = ((previous_cost - current_cost) / initial_cost) / b_to_mb(new_index_size) * self.SCALER
        return reward


class DRLindaReward(RewardCalculator):
    def __init__(self):
        RewardCalculator.__init__(self)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size, similarity_score):
        reward = ((initial_cost - current_cost) / initial_cost) * 100
        return reward
