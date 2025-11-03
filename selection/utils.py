import numpy as np
# from .workload import Workload

# --- Unit conversions ---
# Storage
def b_to_mb(b):
    return b / 1000 / 1000


def mb_to_b(mb):
    return mb * 1000 * 1000


# Time
def s_to_ms(s):
    return s * 1000


# --- Index selection utilities ---


def indexes_by_table(indexes):
    indexes_by_table = {}
    for index in indexes:
        table = index.table()
        if table not in indexes_by_table:
            indexes_by_table[table] = []

        indexes_by_table[table].append(index)

    return indexes_by_table


# def get_utilized_indexes(
#     workload, indexes_per_query, cost_evaluation, detailed_query_information=False
# ):
#     utilized_indexes_workload = set()
#     query_details = {}
#     for query, indexes in zip(workload.queries, indexes_per_query):
#         (
#             utilized_indexes_query,
#             cost_with_indexes,
#         ) = cost_evaluation.which_indexes_utilized_and_cost(query, indexes)
#         utilized_indexes_workload |= utilized_indexes_query

#         if detailed_query_information:
#             cost_without_indexes = cost_evaluation.calculate_cost(
#                 Workload([query]), indexes=[]
#             )

#             query_details[query] = {
#                 "cost_without_indexes": cost_without_indexes,
#                 "cost_with_indexes": cost_with_indexes,
#                 "utilized_indexes": utilized_indexes_query,
#             }

#     return utilized_indexes_workload, query_details


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def log_real(x):
    # 实数域log函数，支持标量和np数组
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=np.float32)
    
    # 处理 x > 1 的情况
    mask_pos = x > 1
    result[mask_pos] = np.log(x[mask_pos]) + 1
    
    # 处理 x < -1 的情况
    mask_neg = x < -1
    result[mask_neg] = -np.log(-x[mask_neg]) - 1
    
    # 处理 -1 <= x <= 1 的情况
    mask_mid = (x >= -1) & (x <= 1)
    result[mask_mid] = x[mask_mid]
    
    # 如果输入是标量，返回标量
    if np.isscalar(x) or x.ndim == 0:
        return result.item()
    return result