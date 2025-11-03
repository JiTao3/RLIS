import logging
from .index import Index
import sqlglot
from sqlglot import expressions as exp
from sqlglot import parse_one, optimizer, exp
from sqlglot.optimizer import normalize
import numpy as np
import math

from .dbms.postgres_dbms import PostgresDatabaseConnector
from .utils import log_real


class Workload:
    def __init__(self, database_name, queries, db_size, description="", workload_idx=None):
        self.database_name = database_name
        self.queries = queries
        self.budget = None
        self.description = description
        self.db_size = db_size
        self.workload_idx = workload_idx

    def indexable_columns(self, return_sorted=True):
        indexable_columns = set()
        for query in self.queries:
            indexable_columns |= set(query.columns)
        if not return_sorted:
            return indexable_columns
        return sorted(list(indexable_columns))

    def potential_indexes(self):
        return sorted([Index([c]) for c in self.indexable_columns()])

    def __repr__(self):
        ids = []
        fr = []
        for query in self.queries:
            ids.append(query.nr)
            fr.append(query.frequency)

        return f"Query IDs: {ids} with {fr}. {self.description} Budget: {self.budget}"

    def __len__(self):
        return len(self.queries)


class Column:
    def __init__(self, name):
        self.name = name.lower()
        self.table = None
        self.global_column_id = None
        self.length = None
        self.distinct_values = None
        self.is_padding_column = False
        self.width = None
        self.domain = None

        self.histogram = None
        self.current_index = None
        self.is_primary_key = False
        self.is_foreign_key = False
        self.distinct_percnetage = None
        self.data_type = None
        self.valve_freq_percentage = None

    def __lt__(self, other):
        return self.name < other.name

    def __repr__(self):
        return f"C {self.table}.{self.name}"

    # We cannot check self.table == other.table here since Table.__eq__()
    # internally checks Column.__eq__. This would lead to endless recursions.
    def __eq__(self, other):
        if isinstance(other, str):
            return self.table.name + "." + self.name == other

        if not isinstance(other, Column):
            return False

        assert (
            self.table is not None and other.table is not None
        ), "Table objects should not be None for Column.__eq__()"

        return self.table.name == other.table.name and self.name == other.name

    def __hash__(self):
        return hash((self.name, self.table.name))

    def prepare(self, database_connector: PostgresDatabaseConnector):
        self.histogram = database_connector.get_hist_bounds(self.table.name, self.name)

        self.domain = database_connector.get_column_domain(self.table.name, self.name)

        self.current_index = database_connector.index_exists(self.table.name, self.name)
        self.is_primary_key = database_connector.get_column_is_primary_key(self.table.name, self.name)
        self.is_foreign_key = database_connector.get_column_is_foreign_key(self.table.name, self.name)
        self.distinct_values = log_real(database_connector.get_distinct_count(self.table.name, self.name))
        self.distinct_percnetage = database_connector.distinct_percentage(self.table.name, self.name)
        self.data_type = database_connector.get_column_data_type(self.table.name, self.name)
        self.most_frequent_value = database_connector.get_most_common_value(self.table.name, self.name)

    def to_vector(self):
        # 确保histogram长度固定且类型为float32
        histogram = log_real(np.array(self.histogram, dtype=np.float32))

        # 确保domain长度固定为2
        if self.data_type < 3:
            domain = log_real(np.array(self.domain, dtype=np.float32))
        else:
            domain = np.array((0, 0), dtype=np.float32)

        current_index = np.array([self.current_index], dtype=np.float32)
        is_primary_key = np.array([self.is_primary_key], dtype=np.float32)
        is_foreign_key = np.array([self.is_foreign_key], dtype=np.float32)
        distinct_values = np.array([self.distinct_values], dtype=np.float32)
        distinct_percnetage = np.array([self.distinct_percnetage], dtype=np.float32)
        data_type = np.array([self.data_type], dtype=np.float32)

        freq_values = [0.0 if x is None else float(x) for x in self.most_frequent_value]
        most_frequent_value = np.array(freq_values, dtype=np.float32)

        vec = np.concatenate(
            (
                histogram,
                domain,
                current_index,
                is_primary_key,
                is_foreign_key,
                distinct_values,
                distinct_percnetage,
                data_type,
                most_frequent_value,
            )
        )
        # assert vec.shape[0] == 158, f"vec.shape: {vec.shape}"
        return vec


class Table:
    def __init__(self, name):
        self.name = name.lower()
        self.columns = []

        self.rows = None
        self.columns_num = None
        self.out_degree = None
        self.in_degree = None
        self.num_primary_keys = None

    def add_column(self, column):
        column.table = self
        self.columns.append(column)

    def add_columns(self, columns):
        for column in columns:
            self.add_column(column)

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        if not isinstance(other, Table):
            return False

        return self.name == other.name and tuple(self.columns) == tuple(other.columns)

    def __hash__(self):
        return hash((self.name, tuple(self.columns)))

    def prepare(self, database_connector: PostgresDatabaseConnector):
        self.rows = database_connector.get_table_rows(self.name)
        self.columns_num = database_connector.get_table_columns_num(self.name)
        self.out_degree, self.in_degree = database_connector.get_table_in_out_degree(self.name)
        self.num_primary_keys = database_connector.get_table_primary_keys_num(self.name)

    def to_vector(self):
        rows = log_real(np.array([self.rows]))
        columns_num = np.array([self.columns_num])
        out_degree = np.array([self.out_degree])
        in_degree = np.array([self.in_degree])
        num_primary_keys = np.array([self.num_primary_keys])
        return np.concatenate((rows, columns_num, out_degree, in_degree, num_primary_keys))


class Query:
    def __init__(self, query_id, query_text, columns=None, frequency=1, parse=False, schema=None):
        self.nr = query_id
        self.text = query_text
        self.frequency = frequency
        # Indexable columns
        self.predicted_cardinality = 0
        if columns is None:
            self.columns = set()
        else:
            self.columns = columns

        self.schema = schema

        if parse:
            self.parse_query()

    def parse_query(self):
        self.tables = set()
        self.alias_table = {}
        self.predicates = set()
        self.joins = set()
        self.aggregates = set()
        self.group_by = set()
        self.having = set()
        self.order_by = set()

        parsed = parse_one(self.text, read="postgres")
        parsed = normalize.normalize(parsed)
        parsed = optimizer.qualify.qualify(parsed, schema=self.schema)

        def is_join_condition(eq):
            tables = set()
            for col in eq.find_all(exp.Column):
                tables.add(col.table)
            return len(tables) >= 2

        for node in parsed.walk():
            if isinstance(node, sqlglot.exp.Column):
                if not node.table:
                    continue
                table_name = node.table
                if table_name not in self.schema.keys():
                    if table_name not in self.alias_table.keys():
                        continue
                    self.columns.add(self.alias_table[table_name] + "." + node.name)
                else:
                    self.columns.add(table_name + "." + node.name)
            elif isinstance(node, sqlglot.exp.Table):
                alias = node.args.get("alias")
                alias_name = alias.name if alias else None
                table_name = node.name
                self.tables.add((table_name, alias_name))
                self.alias_table[alias_name] = table_name
            elif isinstance(node, (sqlglot.exp.Where)):
                for pred in node.walk():
                    if isinstance(pred, exp.EQ) and is_join_condition(pred):  # 只提取等值条件
                        self.joins.add(pred.sql().replace('"', ""))
                    elif isinstance(pred, exp.Binary) and pred.key in ("gt", "gte", "lt", "lte", "eq", "neq"):
                        self.predicates.add(pred.sql().replace('"', ""))
            elif isinstance(node, (sqlglot.exp.AggFunc)):
                self.aggregates.add(node.sql().replace('"', ""))
            elif isinstance(node, (sqlglot.exp.Group)):
                self.group_by.add(node.sql().replace('"', ""))
            elif isinstance(node, (sqlglot.exp.Having)):
                self.having.add(node.sql().replace('"', ""))
            elif isinstance(node, (sqlglot.exp.Order)):
                self.order_by.add(node.sql().replace('"', ""))

    def __repr__(self):
        return f"Q{self.nr}"

    def __eq__(self, other):
        if not isinstance(other, Query):
            return False

        return self.nr == other.nr

    def __hash__(self):
        return hash(self.nr)

    def predicates_vector(self, vector_size, statistics):
        predicate_vector = {} 
        column_predicates = {}

        for predicate_str in self.predicates:
            try:
                expr = sqlglot.parse_one(predicate_str, read="postgres")
            except:
                logging.warning(f"Failed to parse predicate: {predicate_str}. Skipping...")
                continue

            if isinstance(expr, exp.Binary) and expr.key in ("gt", "gte", "lt", "lte", "eq", "neq"):
                left, right = expr.left, expr.right

                if not isinstance(left, exp.Column):
                    continue

                table_name = left.table if left.table else None
                if table_name and table_name in self.alias_table:
                    full_col_key = f"{self.alias_table[table_name]}.{left.name}"
                else:
                    full_col_key = f"{table_name}.{left.name}" if table_name else left.name

                if not isinstance(right, exp.Literal):
                    continue

                try:
                    value = float(right.this)
                except (ValueError, TypeError):
                    continue

                if full_col_key not in column_predicates:
                    column_predicates[full_col_key] = []

                column_predicates[full_col_key].append({"operator": expr.key, "value": value, "sql": predicate_str})

        for full_col_key, predicates in column_predicates.items():
            if full_col_key not in statistics.columns_domain:
                logging.warning(f"Column {full_col_key} not found in statistics. Skipping...")
                continue

            col_min, col_max = statistics.columns_domain[full_col_key]
            if isinstance(col_min, str) or isinstance(col_max, str):
                continue

            valid_intervals = []

            current_intervals = [(col_min, col_max)]

            for pred in predicates:
                operator = pred["operator"]
                value = pred["value"]
                new_intervals = []

                for interval_start, interval_end in current_intervals:
                    if operator in ("gt", "gte"):
                        if operator == "gt":
                            if value < interval_end:
                                new_intervals.append((max(interval_start, value), interval_end))
                        else:  # gte
                            if value <= interval_end:
                                new_intervals.append((max(interval_start, value), interval_end))

                    elif operator in ("lt", "lte"):
                        if operator == "lt":
                            if value > interval_start:
                                new_intervals.append((interval_start, min(interval_end, value)))
                        else:  # lte
                            if value >= interval_start:
                                new_intervals.append((interval_start, min(interval_end, value)))

                    elif operator == "eq":
                        if interval_start <= value <= interval_end:
                            new_intervals.append((value, value))

                    elif operator == "neq":
                        if value > interval_start:
                            new_intervals.append((interval_start, value))
                        if value < interval_end:
                            new_intervals.append((value, interval_end))

                current_intervals = new_intervals

                if not current_intervals:
                    break

            valid_intervals = current_intervals

            if not valid_intervals:
                logging.debug(f"No valid intervals for column {full_col_key} after applying predicates. Skipping...")
                continue

            bin_width = (col_max - col_min) / vector_size
            bins = [col_min + i * bin_width for i in range(vector_size + 1)]

            covered_indices = []

            for interval_start, interval_end in valid_intervals:
                for i in range(vector_size):
                    bin_start = bins[i]
                    bin_end = bins[i + 1] if i < vector_size - 1 else bins[i + 1]

                    if not (bin_end < interval_start or bin_start > interval_end):
                        covered_indices.append(i)

            covered_indices = sorted(list(set(covered_indices)))

            covered_indices = [i for i in covered_indices if 0 <= i < vector_size]
            m = len(covered_indices)

            if m == 0:
                logging.warning(f"No valid bins for column {full_col_key}. Skipping...")
                continue

            if full_col_key not in predicate_vector:
                predicate_vector[full_col_key] = np.zeros(vector_size, dtype=np.float32)

            for idx in covered_indices:
                predicate_vector[full_col_key][idx] += 1.0 / m

        return predicate_vector

    def _get_cardinality_from_pg_(self, database_connector: PostgresDatabaseConnector):
        result = database_connector._get_plan(self.text)
        if result:
            self.predicted_cardinality = result["Plan Rows"]
            return self.predicted_cardinality
        else:
            logging.warning(f"Failed to execute query: {self.text}")
            return None

    def to_vector(self, statistics, max_columns: int, predicate_vec_size: int, max_edges: int):
        predicates_vector = self.predicates_vector(predicate_vec_size, statistics)
        query_vec = []
        node_masking = [0 for _ in range(max_columns)]
        edge_index = [[], []]  # [start_node, end_node]
        edge_weight = []
        column_index = {}
        same_table_col = {}
        for col in self.columns:
            table_name, col_name = col.split(".")
            if same_table_col.get(table_name, None) is None:
                same_table_col[table_name] = []
            same_table_col[table_name].append(col_name)

            table_column = col
            if table_column not in column_index:
                column_index[table_column] = len(column_index)
            col_distribution = statistics.columns_vec_dict[table_column]
            table_level_vector = statistics.tables_vec_dict[table_name]
            if table_column in predicates_vector.keys():
                predicate_range_vec = predicates_vector[table_column]
            else:
                predicate_range_vec = np.zeros(predicate_vec_size, dtype=np.float32)

            if self.predicted_cardinality == 0:
                statistics._ensure_database_connection()
                self.predicted_cardinality = self._get_cardinality_from_pg_(statistics.connector)
                statistics.connector.close()
                statistics.connector = None
            cardinality = math.log(self.predicted_cardinality + 1)
            vec = np.concatenate(
                (
                    col_distribution.astype(np.float32),
                    table_level_vector.astype(np.float32),
                    predicate_range_vec.astype(np.float32),
                    np.array([cardinality], dtype=np.float32),
                )
            )
            query_vec.append(vec)
        for join in self.joins:
            # u.id = o.user_id
            left, right = join.split("=")
            left = left.strip()
            right = right.strip()
            left_full = f"{self.alias_table[left.split('.')[0]]}.{left.split('.')[1]}"
            right_full = f"{self.alias_table[right.split('.')[0]]}.{right.split('.')[1]}"
            edge_index[0].append(column_index[left_full])
            edge_index[1].append(column_index[right_full])
            edge_weight.append(0)
        for table_name, cols in same_table_col.items():
            for c1 in cols:
                for c2 in cols:
                    if c1 == c2:
                        continue
                    left_full = f"{table_name}.{c1}"
                    right_full = f"{table_name}.{c2}"
                    edge_index[0].append(column_index[left_full])
                    edge_index[1].append(column_index[right_full])
                    edge_weight.append(1)

        # padding & masking
        node_masking = [1 for _ in range(len(query_vec))]
        if len(query_vec) > max_columns:
            query_vec = query_vec[:max_columns]
            node_masking = node_masking[:max_columns]

        if len(query_vec) < max_columns:
            node_masking = [0 if i > len(query_vec) else 1 for i in range(max_columns)]
            for _ in range(max_columns - len(query_vec)):
                query_vec.append(np.zeros(len(query_vec[0])))
        edge_masking = [1 for _ in range(len(edge_weight))]
        if len(edge_weight) > max_edges:
            edge_index[0] = edge_index[0][:max_edges]
            edge_index[1] = edge_index[1][:max_edges]
            edge_weight = edge_weight[:max_edges]
            edge_masking = edge_masking[:max_edges]

        if len(edge_weight) < max_edges:
            edge_masking = [0 if i >= len(edge_weight) else 1 for i in range(max_edges)]
            for _ in range(max_edges - len(edge_weight)):
                edge_index[0].append(0)
                edge_index[1].append(0)
                edge_weight.append(0)
        return (
            np.array(query_vec, dtype=np.float32),
            np.array(node_masking, dtype=np.int32),
            np.array(edge_index, dtype=np.int32),
            np.array(edge_weight, dtype=np.int32),
            np.array(edge_masking, dtype=np.int32),
        )
