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

def indexes_by_table(indexes):
    indexes_by_table = {}
    for index in indexes:
        table = index.table()
        if table not in indexes_by_table:
            indexes_by_table[table] = []

        indexes_by_table[table].append(index)

    return indexes_by_table



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def log_real(x):
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=np.float32)
    
    mask_pos = x > 1
    result[mask_pos] = np.log(x[mask_pos]) + 1
    
    mask_neg = x < -1
    result[mask_neg] = -np.log(-x[mask_neg]) - 1
    
    mask_mid = (x >= -1) & (x <= 1)
    result[mask_mid] = x[mask_mid]
    
    if np.isscalar(x) or x.ndim == 0:
        return result.item()
    return result