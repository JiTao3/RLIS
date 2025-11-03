import copy
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from threading import Lock

import numpy as np
import os
import multiprocessing  # 添加导入，用于多进程 Pool
import math

# from .embedding_utils import which_queries_to_remove
from selection.candidate_generation import (
    candidates_per_query,
    syntactically_relevant_indexes,
)

# from selection.cost_evaluation import CostEvaluation
from selection.dbms.postgres_dbms import PostgresDatabaseConnector

# from selection.utils import get_utilized_indexes
from selection.workload import Query, Workload

from tqdm import tqdm

TRAINING_QUERY_PATH = "/data1/weixun/db_workload/workloads/pretrain"
EVAL_QUERY_PATH = "/data1/weixun/db_workload/workloads/test"

"""
workload.db_size -> MB
workload.budget -> percentage
"""


def _create_query_worker(args):
    """工作函数：创建单个Query对象（用于多进程）"""
    query_id, query_text, random_freq, random_seed, schema = args

    if random_freq:
        # 为每个工作进程创建独立的随机数生成器
        np_rnd = np.random.default_rng(seed=random_seed + query_id)
        freq = np_rnd.uniform(0, 10)
    else:
        freq = 1

    return Query(query_id, query_text, frequency=int(freq), parse=True, schema=schema)


class TrainingWorkloadGenerator(object):
    def __init__(self, config, schema, random_seed, eval_mode=False):
        self.debug_mode = config["debug_mode"]
        assert config["benchmark"].lower() in [
            "accidents",
            "ccs",
            "ergastf1",
            "hepatitis",
            "sakila",
            "talkingdata",
            "airline",
            "chembl",
            "financial",
            "hockey",
            "sap",
            "telstra",
            "baseball",
            "consumer",
            "fnhk",
            "imdb",
            "seznam",
            "tournament",
            "basketball",
            "credit",
            "genome",
            "legalacts",
            "ssb",
            "tpc_h",
            "carcinogenesis",
            "employee",
            "grants",
            "movielens",
            "stats",
            "tubepricing",
            "tpch_3g",
        ], f"Benchmark '{config['benchmark']}' is currently not supported for training."
        # self.experiment_id = experiment_id
        self.workload_window_size = config["workload_window_size"]

        self.rnd = random.Random()
        self.rnd.seed(random_seed)
        self.np_rnd = np.random.default_rng(seed=random_seed)

        self.workload_columns = schema.columns
        self.sqlglot_schema = schema.sqlglot_schema

        self.benchmark = config["benchmark"]
        self.database_name = config["benchmark"]
        self.database_connector = PostgresDatabaseConnector(config["benchmark"], autocommit=True)
        self.db_size = self.database_connector.get_db_size_MB()

        self.query_texts = self._load_query_texts()
        self.query_classes = None


        if eval_mode:
            self.eval_query_texts = self._load_query_texts(path=EVAL_QUERY_PATH)
            self.workloads = self.workload_gen(query_texts=self.eval_query_texts)
        else:
            self.workloads = self.workload_gen()
        # 使用新的智能过滤器
        self.filter_columns = TableNumRowsFilter(self.database_connector, self.db_size)
        self.globally_indexable_columns = self._select_indexable_columns(self.filter_columns)
        self.assign_budgets_to_workloads()
        self.database_connector.close()

    def _load_query_texts(self, path=TRAINING_QUERY_PATH):
        finished_queries = []
        with open(f"{path}/{self.benchmark.lower()}/workloads.sql", "r") as f:
            workloads = f.readlines()
            for workload in workloads:
                workload = workload.split("||")[0].split("limit")[0]
                finished_queries.append(workload)

        logging.info(f"Finished load {len(finished_queries)} query texts.")
        if path != TRAINING_QUERY_PATH:
            logging.info(f"Finished load {len(finished_queries)} eval query texts.")
        return finished_queries

    def workload_gen(self, query_texts=None, random_freq=False):
        if query_texts is None:
            query_texts = self.query_texts
        if self.debug_mode:
            query_texts = query_texts[:1000]
        logging.info(f"Start to process {len(query_texts)} queries.")
        workloads = []

        # 检查是否值得使用多进程
        if len(query_texts) < 1000:  # 小于1000个查询时使用单进程
            logging.info("Using single process for small workload")
            queries_results = []
            for query_id, query_text in enumerate(query_texts):
                if random_freq:
                    freq = self.np_rnd.uniform(0, 10)
                else:
                    freq = 1
                query = Query(query_id, query_text, frequency=int(freq), parse=True, schema=self.sqlglot_schema)
                queries_results.append(query)
        else:
            # 使用多进程，但优化参数
            logging.info(f"Using multiprocessing for {len(query_texts)} queries")

            task_args = [
                (query_id, query_text, random_freq, self.rnd.randint(0, 2**31), self.sqlglot_schema)
                for query_id, query_text in enumerate(query_texts)
            ]

            num_processes = min(44, len(task_args) // 100)  # 限制进程数
            if num_processes < 2:
                num_processes = 1

            # 更大的chunksize减少通信开销
            chunksize = max(10, len(task_args) // (num_processes * 4))

            logging.info(f"Using {num_processes} processes with chunksize {chunksize}")

            if num_processes == 1:
                # 退化为单进程
                queries_results = [_create_query_worker(args) for args in task_args]
            else:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    queries_results = pool.map(_create_query_worker, task_args, chunksize=chunksize)

        current_queries = []
        workload_idx = 0
        for query in queries_results:
            if query is not None:
                current_queries.append(query)

                if len(current_queries) == self.workload_window_size:
                    workload = Workload(
                        self.database_name,
                        current_queries,
                        self.db_size,
                        description=f"workload of {self.benchmark}",
                        workload_idx=workload_idx,
                    )
                    workloads.append(workload)
                    current_queries = []  # 重置当前查询列表
                    workload_idx += 1

        logging.info(f"Finished pares {len(queries_results)} queries, assembling {len(workloads)} workloads.")

        return workloads

    def _store_indexable_columns(self, query):
        pass

    def _select_indexable_columns(self, filter_columns):
        index_columns = []
        all_indexable_columns = set()
        for workload in self.workloads:
            indexable_columns = workload.indexable_columns()
            all_indexable_columns |= set(indexable_columns)
            indexable_columns = filter_columns.apply_filter(indexable_columns)
            global_column_id = 0
            for column in self.workload_columns:
                if column.table.name + "." + column.name in indexable_columns:
                    column.global_id = global_column_id
                    global_column_id += 1
                    index_columns.append(column)
        index_columns = list(set(index_columns))
        logging.info(f"All indexable columns length: {len(all_indexable_columns)}")
        logging.info(f"Selected {len(index_columns)} indexable columns.")
        return index_columns

    def __len__(self):
        return len(self.workloads)

    def assign_budgets_to_workloads(self):
        budget_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        for workload in self.workloads:
            workload.budget = self.rnd.choice(budget_list)


class TableNumRowsFilter(object):
    def __init__(self, database_connector, db_size_mb):
        # self.database_name = database_name  # 保存数据库名称以备后用
        self.connector = database_connector
        self.db_size_mb = db_size_mb
        if self.db_size_mb < 100:
            self.threshold = 1000
        elif self.db_size_mb < 1000:
            self.threshold = 5000
        else:
            self.threshold = 10000
        self.table_rows = {}

    def apply_filter(self, columns):

        output_columns = []

        for column in columns:
            table_name = column.split(".")[0]
            if table_name in self.table_rows.keys():
                table_num_rows = self.table_rows[table_name]
            else:
                table_num_rows = self.connector.exec_fetch(
                    f"SELECT reltuples::bigint AS estimate FROM pg_class where relname='{table_name}'", one=True
                )[0]

            if table_num_rows > self.threshold:
                output_columns.append(column)

        # logging.warning(f"Reduced columns from {len(columns)} to {len(output_columns)}.")

        return output_columns


class DatabaseSizeBasedColumnFilter(object):
    """基于数据库大小的智能列过滤器"""

    def __init__(self, database_connector, db_size_mb, config=None):
        self.connector = database_connector
        self.db_size_mb = db_size_mb
        self.config = config or {}

        # 根据数据库大小确定过滤策略
        self.filter_strategy = self._determine_filter_strategy()

        # 四个缓存buffer
        self.table_stats_buffer = {}
        self.column_selectivity_buffer = {}
        self.column_type_buffer = {}
        self.storage_efficiency_buffer = {}

    def _determine_filter_strategy(self):
        """根据数据库大小确定过滤策略"""
        if self.db_size_mb < 100:  # 小型数据库 (<100MB)
            return "relaxed"
        elif self.db_size_mb < 1000:  # 中型数据库 (100MB-1GB)
            return "moderate"
        else:  # 大型数据库 (>1GB)
            return "strict"

    def apply_filter(self, columns):
        """应用多维度过滤策略"""
        output_columns = []

        # 批量预加载表统计信息
        self._preload_table_statistics(columns)

        for column in columns:
            table_name, column_name = column.split(".")

            # 1. 基本表大小过滤
            if not self._passes_table_size_filter(table_name):
                continue

            # 2. 列选择性过滤
            # if not self._passes_selectivity_filter(table_name, column_name):
            #     continue

            # # 3. 存储效率过滤
            # if not self._passes_storage_efficiency_filter(table_name, column_name):
            #     continue

            output_columns.append(column)

        # logging.info(f"Buffer stats - Tables: {len(self.table_stats_buffer)}, "
        #             f"Selectivity: {len(self.column_selectivity_buffer)}, "
        #             f"Types: {len(self.column_type_buffer)}, "
        #             f"Storage: {len(self.storage_efficiency_buffer)}")

        return output_columns

    def _preload_table_statistics(self, columns):
        """批量预加载表统计信息"""
        table_names = list(set(col.split(".")[0] for col in columns))
        missing_tables = [name for name in table_names if name not in self.table_stats_buffer]

        if not missing_tables:
            return

        try:
            # 批量查询所有缺失的表统计信息
            table_names_str = "', '".join(missing_tables)
            batch_query = f"""
            SELECT 
                relname as table_name,
                reltuples::bigint as row_count,
                pg_total_relation_size(oid)/(1024*1024) as size_mb,
                relpages as page_count
            FROM pg_class 
            WHERE relname IN ('{table_names_str}') AND relkind = 'r'
            """

            results = self.connector.exec_fetch(batch_query)

            # 缓存查询结果
            for result in results:
                table_name = result[0]
                self.table_stats_buffer[table_name] = {
                    "row_count": result[1],
                    "size_mb": result[2],
                    "page_count": result[3],
                    "size_ratio": result[2] / self.db_size_mb if self.db_size_mb > 0 else 0,
                }

            # 为没有找到的表设置默认值
            for table_name in missing_tables:
                if table_name not in self.table_stats_buffer:
                    self.table_stats_buffer[table_name] = {
                        "row_count": 0,
                        "size_mb": 0,
                        "page_count": 0,
                        "size_ratio": 0,
                    }

            logging.debug(f"Batch loaded statistics for {len(missing_tables)} tables")

        except Exception as e:
            logging.warning(f"Failed to batch load table stats: {e}")
            # 设置默认值
            for table_name in missing_tables:
                self.table_stats_buffer[table_name] = {"row_count": 0, "size_mb": 0, "page_count": 0, "size_ratio": 0}

    def _passes_table_size_filter(self, table_name):
        """表大小过滤：根据数据库大小调整表大小阈值"""
        # 从buffer获取表统计信息
        table_stat = self.table_stats_buffer.get(table_name, {})
        row_count = table_stat.get("row_count", 0)
        size_ratio = table_stat.get("size_ratio", 0)

        if self.filter_strategy == "relaxed":
            return row_count > 100 or size_ratio > 0.01
        elif self.filter_strategy == "moderate":
            return row_count > 1000 and size_ratio > 0.005
        else:  # strict
            return row_count > 10000 and size_ratio > 0.002

    def _passes_selectivity_filter(self, table_name, column_name):
        """列选择性过滤：高选择性的列更适合建索引"""
        # 使用完整的表.列名作为缓存键
        cache_key = f"{table_name}.{column_name}"

        # 先检查缓存
        if cache_key in self.column_selectivity_buffer:
            cached_result = self.column_selectivity_buffer[cache_key]
            if isinstance(cached_result, dict):
                selectivity = cached_result.get("selectivity", 0)
                distinct_values = cached_result.get("distinct_values", 0)
                return self._evaluate_selectivity(selectivity, distinct_values)
            else:
                return cached_result

        try:
            selectivity_query = f"""
            SELECT 
                COUNT(DISTINCT {column_name})::float / NULLIF(COUNT(*), 0)::float as selectivity,
                COUNT(DISTINCT {column_name}) as distinct_values,
                COUNT(*) as total_rows
            FROM {table_name}
            WHERE {column_name} IS NOT NULL
            """
            result = self.connector.exec_fetch(selectivity_query, one=True)

            if result and result[1] is not None and result[1] > 0:
                selectivity_data = {
                    "selectivity": result[0] if result[0] is not None else 0,
                    "distinct_values": result[1],
                    "total_rows": result[2],
                }
                # 缓存详细数据
                self.column_selectivity_buffer[cache_key] = selectivity_data

                return self._evaluate_selectivity(selectivity_data["selectivity"], selectivity_data["distinct_values"])
            else:
                # 缓存失败结果
                self.column_selectivity_buffer[cache_key] = False
                return False

        except Exception as e:
            logging.warning(f"Failed to calculate selectivity for {table_name}.{column_name}: {e}")
            # 缓存错误情况下的默认值
            self.column_selectivity_buffer[cache_key] = True
            return True

    def _evaluate_selectivity(self, selectivity, distinct_values):
        """评估选择性是否通过过滤"""
        if self.filter_strategy == "relaxed":
            return selectivity > 0.1 or distinct_values > 10
        elif self.filter_strategy == "moderate":
            return selectivity > 0.05 and distinct_values > 50
        else:  # strict
            return 0.01 < selectivity < 0.95 and distinct_values > 100

    def _passes_storage_efficiency_filter(self, table_name, column_name):
        """存储效率过滤：评估索引的存储成本效益"""
        cache_key = f"{table_name}.{column_name}"

        # 先检查存储效率缓存
        if cache_key in self.storage_efficiency_buffer:
            return self.storage_efficiency_buffer[cache_key]

        try:
            # 先获取列类型信息（使用type buffer）
            column_type = self._get_column_type_info(table_name, column_name)
            if not column_type:
                self.storage_efficiency_buffer[cache_key] = True
                return True

            # 获取表统计信息
            table_stat = self.table_stats_buffer.get(table_name, {})
            table_size_mb = table_stat.get("size_mb", 0)
            row_count = table_stat.get("row_count", 0)

            # 计算存储效率
            data_type = column_type.get("data_type")
            if data_type:
                estimated_index_size_ratio = self._estimate_index_size_ratio(data_type, table_size_mb, row_count)

                if self.filter_strategy == "relaxed":
                    result = estimated_index_size_ratio < 0.5
                elif self.filter_strategy == "moderate":
                    result = estimated_index_size_ratio < 0.3
                else:  # strict
                    result = estimated_index_size_ratio < 0.2
            else:
                result = True

            # 缓存结果
            self.storage_efficiency_buffer[cache_key] = result
            return result

        except Exception as e:
            logging.warning(f"Failed to evaluate storage efficiency for {table_name}.{column_name}: {e}")
            # 缓存错误情况下的默认值
            self.storage_efficiency_buffer[cache_key] = True
            return True

    def _get_column_type_info(self, table_name, column_name):
        """获取列类型信息（使用type buffer）"""
        cache_key = f"{table_name}.{column_name}"

        # 先检查类型缓存
        if cache_key in self.column_type_buffer:
            return self.column_type_buffer[cache_key]

        try:
            type_info_query = f"""
            SELECT 
                data_type,
                character_maximum_length,
                numeric_precision
            FROM information_schema.columns 
            WHERE table_name = '{table_name}' AND column_name = '{column_name}'
            """
            result = self.connector.exec_fetch(type_info_query, one=True)

            if result:
                type_info = {
                    "data_type": result[0],
                    "character_maximum_length": result[1],
                    "numeric_precision": result[2],
                }
            else:
                type_info = {"data_type": None, "character_maximum_length": None, "numeric_precision": None}

            # 缓存结果
            self.column_type_buffer[cache_key] = type_info
            return type_info

        except Exception as e:
            logging.warning(f"Failed to get type info for {table_name}.{column_name}: {e}")
            # 缓存空结果
            default_type_info = {"data_type": None, "character_maximum_length": None, "numeric_precision": None}
            self.column_type_buffer[cache_key] = default_type_info
            return default_type_info

    def _estimate_index_size_ratio(self, data_type, table_size_mb, row_count):
        """估算索引大小相对于表大小的比例"""
        if row_count <= 0 or table_size_mb <= 0:
            return 0

        # 根据数据类型估算每行索引开销（字节）
        index_overhead_per_row = 8  # 基础开销

        if data_type in ["integer", "int", "int4"]:
            index_overhead_per_row += 4
        elif data_type in ["bigint", "int8"]:
            index_overhead_per_row += 8
        elif data_type in ["varchar", "text", "char"]:
            index_overhead_per_row += 20  # 估算平均字符串长度
        elif data_type in ["timestamp", "date"]:
            index_overhead_per_row += 8
        else:
            index_overhead_per_row += 16  # 默认值

        estimated_index_size_mb = (index_overhead_per_row * row_count) / (1024 * 1024)
        return estimated_index_size_mb / table_size_mb if table_size_mb > 0 else 0

    def clear_buffers(self):
        """清空所有缓存buffer"""
        self.table_stats_buffer.clear()
        self.column_selectivity_buffer.clear()
        self.column_type_buffer.clear()
        self.storage_efficiency_buffer.clear()
        logging.info("Cleared all filter buffers")

    def get_buffer_stats(self):
        """获取buffer统计信息"""
        return {
            "table_stats": len(self.table_stats_buffer),
            "column_selectivity": len(self.column_selectivity_buffer),
            "column_type": len(self.column_type_buffer),
            "storage_efficiency": len(self.storage_efficiency_buffer),
        }


# class WorkloadGenerator(object):
#     def __init__(
#         self, config, workload_columns, random_seed, database_name, experiment_id=None, filter_utilized_columns=None
#     ):
#         assert config["benchmark"].lower() in [
#             "TPCH",
#             "TPCDS",
#             "JOB",
#             "ccidents",
#             "ccs",
#             "ergastf1",
#             "hepatitis",
#             "sakila",
#             "talkingdata",
#             "airline",
#             "chembl",
#             "financial",
#             "hockey",
#             "sap",
#             "telstra",
#             "baseball",
#             "consumer",
#             "fnhk",
#             "imdb",
#             "seznam",
#             "tournament",
#             "basketball",
#             "credit",
#             "genome",
#             "legalacts",
#             "ssb",
#             "tpc_h",
#             "carcinogenesis",
#             "employee",
#             "grants",
#             "movielens",
#             "stats",
#             "tubepricing",
#         ], f"Benchmark '{config['benchmark']}' is currently not supported."

#         # For create view statement differentiation
#         self.experiment_id = experiment_id
#         self.filter_utilized_columns = filter_utilized_columns

#         self.rnd = random.Random()
#         self.rnd.seed(random_seed)
#         self.np_rnd = np.random.default_rng(seed=random_seed)

#         self.workload_columns = workload_columns
#         self.database_name = database_name

#         self.benchmark = config["benchmark"]
#         self.number_of_query_classes = self._set_number_of_query_classes()
#         self.excluded_query_classes = set(config["excluded_query_classes"])
#         self.varying_frequencies = config["varying_frequencies"]

#         # self.query_texts is list of lists. Outer list for query classes, inner list for instances of this class.
#         self.query_texts = self._retrieve_query_texts()
#         self.query_classes = set(range(1, self.number_of_query_classes + 1))
#         self.available_query_classes = self.query_classes - self.excluded_query_classes

#         self.globally_indexable_columns = self._select_indexable_columns(self.filter_utilized_columns)

#         validation_instances = config["validation_testing"]["number_of_workloads"]
#         test_instances = config["validation_testing"]["number_of_workloads"]
#         self.wl_validation = []
#         self.wl_testing = []

#         if config["similar_workloads"] and config["unknown_queries"] == 0:
#             # Todo: this branch can probably be removed
#             assert self.varying_frequencies, "Similar workloads can only be created with varying frequencies."
#             self.wl_validation = [None]
#             self.wl_testing = [None]
#             _, self.wl_validation[0], self.wl_testing[0] = self._generate_workloads(
#                 0, validation_instances, test_instances, config["size"]
#             )
#             if config["query_class_change_frequency"] is None:
#                 self.wl_training = self._generate_similar_workloads(config["training_instances"], config["size"])
#             else:
#                 self.wl_training = self._generate_similar_workloads_qccf(
#                     config["training_instances"], config["size"], config["query_class_change_frequency"]
#                 )
#         elif config["unknown_queries"] > 0:
#             assert (
#                 config["validation_testing"]["unknown_query_probabilities"][-1] > 0
#             ), "Query unknown_query_probabilities should be larger 0."

#             embedder_connector = PostgresDatabaseConnector(self.database_name, autocommit=True)
#             embedder = WorkloadEmbedder(
#                 # Transform globally_indexable_columns to list of lists.
#                 self.query_texts,
#                 0,
#                 embedder_connector,
#                 [list(map(lambda x: [x], self.globally_indexable_columns))],
#                 retrieve_plans=True,
#             )
#             self.unknown_query_classes = which_queries_to_remove(
#                 embedder.plans, config["unknown_queries"], random_seed
#             )

#             self.unknown_query_classes = frozenset(self.unknown_query_classes) - self.excluded_query_classes
#             missing_classes = config["unknown_queries"] - len(self.unknown_query_classes)
#             self.unknown_query_classes = self.unknown_query_classes | frozenset(
#                 self.rnd.sample(self.available_query_classes - frozenset(self.unknown_query_classes), missing_classes)
#             )
#             assert len(self.unknown_query_classes) == config["unknown_queries"]

#             self.known_query_classes = self.available_query_classes - frozenset(self.unknown_query_classes)
#             embedder = None

#             for query_class in self.excluded_query_classes:
#                 assert query_class not in self.unknown_query_classes

#             logging.critical(f"Global unknown query classes: {sorted(self.unknown_query_classes)}")
#             logging.critical(f"Global known query classes: {sorted(self.known_query_classes)}")

#             for unknown_query_probability in config["validation_testing"]["unknown_query_probabilities"]:
#                 _, wl_validation, wl_testing = self._generate_workloads(
#                     0,
#                     validation_instances,
#                     test_instances,
#                     config["size"],
#                     unknown_query_probability=unknown_query_probability,
#                 )
#                 self.wl_validation.append(wl_validation)
#                 self.wl_testing.append(wl_testing)

#             assert (
#                 len(self.wl_validation)
#                 == len(config["validation_testing"]["unknown_query_probabilities"])
#                 == len(self.wl_testing)
#             ), "Validation/Testing workloads length fail"

#             # We are temporarily restricting the available query classes now to exclude certain classes for training
#             original_available_query_classes = self.available_query_classes
#             self.available_query_classes = self.known_query_classes

#             if config["similar_workloads"]:
#                 if config["query_class_change_frequency"] is not None:
#                     logging.critical(
#                         f"Similar workloads with query_class_change_frequency: {config['query_class_change_frequency']}"
#                     )
#                     self.wl_training = self._generate_similar_workloads_qccf(
#                         config["training_instances"], config["size"], config["query_class_change_frequency"]
#                     )
#                 else:
#                     self.wl_training = self._generate_similar_workloads(config["training_instances"], config["size"])
#             else:
#                 self.wl_training, _, _ = self._generate_workloads(config["training_instances"], 0, 0, config["size"])
#             # We are removing the restriction now.
#             self.available_query_classes = original_available_query_classes
#         else:
#             self.wl_validation = [None]
#             self.wl_testing = [None]
#             self.wl_training, self.wl_validation[0], self.wl_testing[0] = self._generate_workloads(
#                 config["training_instances"], validation_instances, test_instances, config["size"]
#             )

#         logging.critical(f"Sample training workloads: {self.rnd.sample(self.wl_training, 10)}")
#         logging.info("Finished generating workloads.")

#     def _set_number_of_query_classes(self):
#         if self.benchmark == "TPCH":
#             return 22
#         elif self.benchmark == "TPCDS":
#             return 99
#         elif self.benchmark == "JOB":
#             return 113
#         else:
#             logging.info(f"Benchmark {self.benchmark} get query form file directly")
#             # raise ValueError("Unsupported Benchmark type provided, only TPCH, TPCDS, and JOB supported.")

#     def _retrieve_query_texts(self):

#         finished_queries = []
#         with open(f"{QUERY_PATH}/{self.benchmark.lower()}/workloads.sql", "r") as f:
#             workloads = f.readlines()
#             for workload in workloads:
#                 workload = workload.split("||")[0]
#                 finished_queries.append(workload)

#         logging.info(f"Finished retrieving {len(finished_queries)} workloads.")
#         self.number_of_query_classes = len(finished_queries)
#         return finished_queries

#     def _preprocess_queries(self, queries):
#         processed_queries = []
#         for query in queries:
#             query = query.replace("limit 100", "")
#             query = query.replace("limit 20", "")
#             query = query.replace("limit 10", "")
#             query = query.strip()

#             if "create view revenue0" in query:
#                 query = query.replace("revenue0", f"revenue0_{self.experiment_id}")

#             processed_queries.append(query)

#         return processed_queries

#     def _store_indexable_columns(self, query):
#         if self.benchmark != "JOB":
#             for column in self.workload_columns:
#                 if column.name in query.text:
#                     query.columns.append(column)
#         else:
#             query_text = query.text
#             assert "WHERE" in query_text, f"Query without WHERE clause encountered: {query_text} in {query.nr}"

#             split = query_text.split("WHERE")
#             assert len(split) == 2, "Query split for JOB query contains subquery"
#             query_text_before_where = split[0]
#             query_text_after_where = split[1]

#             for column in self.workload_columns:
#                 if column.name in query_text_after_where and f"{column.table.name} " in query_text_before_where:
#                     query.columns.append(column)

#     def _workloads_from_tuples(self, tuples, unknown_query_probability=None):
#         workloads = []
#         unknown_query_probability = "" if unknown_query_probability is None else unknown_query_probability

#         for tupl in tuples:
#             query_classes, query_class_frequencies = tupl
#             queries = []

#             for query_class, frequency in zip(query_classes, query_class_frequencies):
#                 query_text = self.rnd.choice(self.query_texts[query_class - 1])

#                 query = Query(query_class, query_text, frequency=frequency, parse=True)

#                 self._store_indexable_columns(query)
#                 assert len(query.columns) > 0, f"Query columns should have length > 0: {query.text}"

#                 queries.append(query)

#             assert isinstance(queries, list), f"Queries is not of type list but of {type(queries)}"
#             previously_unseen_queries = (
#                 round(unknown_query_probability * len(queries)) if unknown_query_probability != "" else 0
#             )
#             workloads.append(
#                 Workload(queries, description=f"Contains {previously_unseen_queries} previously unseen queries.")
#             )

#         return workloads

#     def _generate_workloads(
#         self, train_instances, validation_instances, test_instances, size, unknown_query_probability=None
#     ):
#         required_unique_workloads = train_instances + validation_instances + test_instances

#         unique_workload_tuples = set()

#         while required_unique_workloads > len(unique_workload_tuples):
#             workload_tuple = self._generate_random_workload(size, unknown_query_probability)
#             unique_workload_tuples.add(workload_tuple)

#         validation_tuples = self.rnd.sample(unique_workload_tuples, validation_instances)
#         unique_workload_tuples = unique_workload_tuples - set(validation_tuples)

#         test_workload_tuples = self.rnd.sample(unique_workload_tuples, test_instances)
#         unique_workload_tuples = unique_workload_tuples - set(test_workload_tuples)

#         assert len(unique_workload_tuples) == train_instances
#         train_workload_tuples = unique_workload_tuples

#         assert (
#             len(train_workload_tuples) + len(test_workload_tuples) + len(validation_tuples) == required_unique_workloads
#         )

#         validation_workloads = self._workloads_from_tuples(validation_tuples, unknown_query_probability)
#         test_workloads = self._workloads_from_tuples(test_workload_tuples, unknown_query_probability)
#         train_workloads = self._workloads_from_tuples(train_workload_tuples, unknown_query_probability)

#         return train_workloads, validation_workloads, test_workloads

#     # The core idea is to create workloads that are similar and only change slightly from one to another.
#     # For the following workload, we remove one random element, add another random one with frequency, and
#     # randomly change the frequency of one element (including the new one).
#     def _generate_similar_workloads(self, instances, size):
#         assert size <= len(
#             self.available_query_classes
#         ), "Cannot generate workload with more queries than query classes"

#         workload_tuples = []

#         query_classes = self.rnd.sample(self.available_query_classes, size)
#         available_query_classes = self.available_query_classes - frozenset(query_classes)
#         frequencies = list(self.np_rnd.zipf(1.5, size))

#         workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

#         for workload_idx in range(instances - 1):
#             # Remove a random element
#             idx_to_remove = self.rnd.randrange(len(query_classes))
#             query_classes.pop(idx_to_remove)
#             frequencies.pop(idx_to_remove)

#             # Draw a new random element, the removed one is excluded
#             query_classes.append(self.rnd.sample(available_query_classes, 1)[0])
#             frequencies.append(self.np_rnd.zipf(1.5, 1)[0])

#             frequencies[self.rnd.randrange(len(query_classes))] = self.np_rnd.zipf(1.5, 1)[0]

#             available_query_classes = self.available_query_classes - frozenset(query_classes)
#             workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

#         workloads = self._workloads_from_tuples(workload_tuples)

#         return workloads

#     # This version uses the same query id selction for query_class_change_frequency workloads
#     def _generate_similar_workloads_qccf(self, instances, size, query_class_change_frequency):
#         assert size <= len(
#             self.available_query_classes
#         ), "Cannot generate workload with more queries than query classes"

#         workload_tuples = []

#         while len(workload_tuples) < instances:
#             if len(workload_tuples) % query_class_change_frequency == 0:
#                 query_classes = self.rnd.sample(self.available_query_classes, size)

#             frequencies = list(self.np_rnd.integers(1, 10000, size))
#             workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

#         workloads = self._workloads_from_tuples(workload_tuples)

#         return workloads

#     def _generate_random_workload(self, size, unknown_query_probability=None):
#         assert size <= self.number_of_query_classes, "Cannot generate workload with more queries than query classes"

#         workload_query_classes = None
#         if unknown_query_probability is not None:
#             number_of_unknown_queries = round(size * unknown_query_probability)
#             number_of_known_queries = size - number_of_unknown_queries
#             assert number_of_known_queries + number_of_unknown_queries == size

#             known_query_classes = self.rnd.sample(self.known_query_classes, number_of_known_queries)
#             unknown_query_classes = self.rnd.sample(self.unknown_query_classes, number_of_unknown_queries)
#             query_classes = known_query_classes
#             query_classes.extend(unknown_query_classes)
#             workload_query_classes = tuple(query_classes)
#             assert len(workload_query_classes) == size
#         else:
#             workload_query_classes = tuple(self.rnd.sample(self.available_query_classes, size))

#         # Create frequencies
#         if self.varying_frequencies:
#             query_class_frequencies = tuple(list(self.np_rnd.integers(1, 10000, size)))
#         else:
#             query_class_frequencies = tuple([1 for frequency in range(size)])

#         workload_tuple = (workload_query_classes, query_class_frequencies)

#         return workload_tuple

#     def _only_utilized_indexes(self, indexable_columns):
#         frequencies = [1 for frequency in range(len(self.available_query_classes))]
#         workload_tuple = (self.available_query_classes, frequencies)
#         workload = self._workloads_from_tuples([workload_tuple])[0]

#         candidates = candidates_per_query(
#             workload,
#             max_index_width=1,
#             candidate_generator=syntactically_relevant_indexes,
#         )

#         connector = PostgresDatabaseConnector(self.database_name, autocommit=True)
#         connector.drop_indexes()
#         cost_evaluation = CostEvaluation(connector)

#         utilized_indexes, query_details = get_utilized_indexes(workload, candidates, cost_evaluation, True)

#         columns_of_utilized_indexes = set()
#         for utilized_index in utilized_indexes:
#             column = utilized_index.columns[0]
#             columns_of_utilized_indexes.add(column)

#         output_columns = columns_of_utilized_indexes & set(indexable_columns)
#         excluded_columns = set(indexable_columns) - output_columns
#         logging.critical(f"Excluding columns based on utilization:\n   {excluded_columns}")

#         return output_columns

#     def _select_indexable_columns(self, only_utilized_indexes=False):
#         available_query_classes = tuple(self.available_query_classes)
#         query_class_frequencies = tuple([1 for frequency in range(len(available_query_classes))])

#         logging.info(f"Selecting indexable columns on {len(available_query_classes)} query classes.")

#         workload = self._workloads_from_tuples([(available_query_classes, query_class_frequencies)])[0]

#         indexable_columns = workload.indexable_columns()
#         if only_utilized_indexes:
#             indexable_columns = self._only_utilized_indexes(indexable_columns)
#         selected_columns = []

#         global_column_id = 0
#         for column in self.workload_columns:
#             if column in indexable_columns:
#                 column.global_column_id = global_column_id
#                 global_column_id += 1

#                 selected_columns.append(column)

#         return selected_columns
