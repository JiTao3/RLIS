import logging
import re
import threading
import os
from contextlib import contextmanager

import psycopg2
import numpy as np
from ..database_connector import DatabaseConnector

USER = "weixun"
PASSWORD = ""
HOST = "127.0.0.1"
PORT = "12335"

# 进程本地连接池管理 - 避免跨进程pickle问题
_process_connection_pools = {}
_pool_lock = threading.Lock()
_pool_config = {
    'max_connections': 5,  # 每个进程每个数据库最多3个连接
}

def _get_process_pools():
    """获取当前进程的连接池字典"""
    global _process_connection_pools
    current_pid = os.getpid()
    if current_pid not in _process_connection_pools:
        _process_connection_pools[current_pid] = {}
    return _process_connection_pools[current_pid]

class PostgresDatabaseConnector(DatabaseConnector):
    def __init__(self, db_name, autocommit=False):
        DatabaseConnector.__init__(self, db_name, autocommit=autocommit)
        self.db_system = "postgres"
        self._connection = None
        self._available_connections = None
        self._process_id = os.getpid()

        if not self.db_name:
            self.db_name = "postgres"
        
        self._ensure_connection_pool()
        self.create_connection()

        self.set_random_seed()

        self.exec_only("SET max_parallel_workers_per_gather = 0;")
        self.exec_only("SET enable_bitmapscan TO off;")

        logging.debug("Postgres connector created: {} (PID: {})".format(db_name, self._process_id))

    def _ensure_connection_pool(self):
        """为当前进程的特定数据库确保连接池存在"""
        with _pool_lock:
            process_pools = _get_process_pools()
            if self.db_name not in process_pools:
                try:
                    # 为每个数据库创建简单的连接列表而不是复杂的连接池
                    process_pools[self.db_name] = {
                        'connections': [],
                        'max_connections': _pool_config['max_connections'],
                        'created_count': 0
                    }
                    logging.info(f"Created connection pool for database: {self.db_name} in process {self._process_id}")
                except Exception as e:
                    logging.error(f"Failed to create connection pool for {self.db_name}: {e}")
                    raise
            self._available_connections = process_pools[self.db_name]

    def _create_new_connection(self):
        """创建新的数据库连接"""
        try:
            conn = psycopg2.connect(
                f"dbname={self.db_name} user={USER} password={PASSWORD} host={HOST} port={PORT}"
            )
            conn.autocommit = self.autocommit
            logging.debug(f"Created new connection for database: {self.db_name}")
            return conn
        except Exception as e:
            logging.error(f"Failed to create connection for {self.db_name}: {e}")
            raise

    def create_connection(self):
        """获取数据库连接"""
        if self._connection:
            self.close()
        
        with _pool_lock:
            # 检查是否有可用的连接
            if self._available_connections['connections']:
                self._connection = self._available_connections['connections'].pop()
                logging.debug(f"Reused connection for database: {self.db_name}")
            else:
                # 检查是否可以创建新连接
                if self._available_connections['created_count'] < self._available_connections['max_connections']:
                    self._connection = self._create_new_connection()
                    self._available_connections['created_count'] += 1
                else:
                    # 如果达到最大连接数，创建临时连接
                    logging.warning(f"Max connections reached for {self.db_name}, creating temporary connection")
                    self._connection = self._create_new_connection()
            
            self._connection.autocommit = self.autocommit
            self._cursor = self._connection.cursor()

    def close(self):
        """释放数据库连接回连接池"""
        if self._connection:
            try:
                if hasattr(self, '_cursor') and self._cursor:
                    self._cursor.close()
                
                # 检查连接是否仍然有效
                if not self._connection.closed:
                    with _pool_lock:
                        # 将连接返回到池中
                        if (self._available_connections and 
                            len(self._available_connections['connections']) < self._available_connections['max_connections']):
                            self._available_connections['connections'].append(self._connection)
                            logging.debug(f"Returned connection to pool for database: {self.db_name}")
                        else:
                            # 如果池已满，直接关闭连接
                            self._connection.close()
                            if self._available_connections:
                                self._available_connections['created_count'] -= 1
                            logging.debug(f"Closed excess connection for database: {self.db_name}")
                else:
                    # 连接已关闭，只需要更新计数
                    with _pool_lock:
                        if self._available_connections:
                            self._available_connections['created_count'] -= 1
                    
            except Exception as e:
                logging.error(f"Error handling connection for {self.db_name}: {e}")
                try:
                    self._connection.close()
                except:
                    pass
            finally:
                self._connection = None
                self._cursor = None

    @classmethod
    def get_pool_status(cls):
        """获取当前进程所有连接池的状态信息"""
        with _pool_lock:
            current_pid = os.getpid()
            if current_pid not in _process_connection_pools:
                return {}
            
            status = {}
            process_pools = _process_connection_pools[current_pid]
            for db_name, pool_info in process_pools.items():
                status[db_name] = {
                    'process_id': current_pid,
                    'max_connections': pool_info['max_connections'],
                    'created_connections': pool_info['created_count'],
                    'available_connections': len(pool_info['connections']),
                    'active_connections': pool_info['created_count'] - len(pool_info['connections'])
                }
            return status

    @classmethod
    def close_all_pools(cls):
        """关闭当前进程的所有连接池"""
        with _pool_lock:
            current_pid = os.getpid()
            if current_pid in _process_connection_pools:
                process_pools = _process_connection_pools[current_pid]
                for db_name, pool_info in process_pools.items():
                    try:
                        # 关闭所有连接
                        for conn in pool_info['connections']:
                            try:
                                conn.close()
                            except:
                                pass
                        pool_info['connections'].clear()
                        pool_info['created_count'] = 0
                        logging.info(f"Closed all connections for database: {db_name} in process {current_pid}")
                    except Exception as e:
                        logging.error(f"Error closing connections for {db_name}: {e}")
                
                # 清理当前进程的连接池
                del _process_connection_pools[current_pid]

    def __del__(self):
        """析构函数，确保连接被正确释放"""
        self.close()

    def enable_simulation(self):
        self.exec_only("CREATE EXTENSION IF NOT EXISTS hypopg;")
        self.commit()

    def database_names(self):
        result = self.exec_fetch("select datname from pg_database", False)
        return [x[0] for x in result]

    # Updates query syntax to work in PostgreSQL
    def update_query_text(self, text):
        text = text.replace(";\nlimit ", " limit ").replace("limit -1", "")
        text = re.sub(r" ([0-9]+) days\)", r" interval '\1 days')", text)
        text = self._add_alias_subquery(text)
        return text

    # PostgreSQL requires an alias for subqueries
    def _add_alias_subquery(self, query_text):
        text = query_text.lower()
        positions = []
        for match in re.finditer(r"((from)|,)[  \n]*\(", text):
            counter = 1
            pos = match.span()[1]
            while counter > 0:
                char = text[pos]
                if char == "(":
                    counter += 1
                elif char == ")":
                    counter -= 1
                pos += 1
            next_word = query_text[pos:].lstrip().split(" ")[0].split("\n")[0]
            if next_word[0] in [")", ","] or next_word in [
                "limit",
                "group",
                "order",
                "where",
            ]:
                positions.append(pos)
        for pos in sorted(positions, reverse=True):
            query_text = query_text[:pos] + " as alias123 " + query_text[pos:]
        return query_text

    def create_database(self, database_name):
        self.exec_only("create database {}".format(database_name))
        logging.info("Database {} created".format(database_name))

    def import_data(self, table, path, delimiter="|", encoding=None):
        if encoding:
            with open(path, "r", encoding=encoding) as file:
                self._cursor.copy_expert(
                    (
                        f"COPY {table} FROM STDIN WITH DELIMITER AS '{delimiter}' NULL "
                        f"AS 'NULL' CSV QUOTE AS '\"' ENCODING '{encoding}'"
                    ),
                    file,
                )
        else:
            with open(path, "r") as file:
                self._cursor.copy_from(file, table, sep=delimiter, null="")

    def indexes_size(self):
        # Returns size in bytes
        statement = (
            "select sum(pg_indexes_size(table_name::text)) from "
            "(select table_name from information_schema.tables "
            "where table_schema='public') as all_tables"
        )
        result = self.exec_fetch(statement, one=True)
        return result[0]

    def drop_database(self, database_name):
        statement = f"DROP DATABASE {database_name};"
        self.exec_only(statement)

        logging.info(f"Database {database_name} dropped")

    def create_statistics(self):
        logging.info("Postgres: Run `analyze`")
        self.commit()
        self._connection.autocommit = True
        self.exec_only("analyze;")
        self._connection.autocommit = self.autocommit

    def set_random_seed(self, value=0.17):
        logging.debug(f"Postgres: Set random seed `SELECT setseed({value})`")
        self.exec_only(f"SELECT setseed({value})")

    def supports_index_simulation(self):
        if self.db_system == "postgres":
            return True
        return False

    def _simulate_index(self, index):
        table_name = index.table()
        statement = (
            "select * from hypopg_create_index( " f"'create index on {table_name} " f"({index.joined_column_names()})')"
        )
        result = self.exec_fetch(statement, one=True)
        return result

    def _drop_simulated_index(self, oid):
        statement = f"select * from hypopg_drop_index({oid})"
        result = self.exec_fetch(statement, one=True)

        assert result[0] is True, f"Could not drop simulated index with oid = {oid}."

    def create_index(self, index):
        table_name = index.table()
        statement = f"create index {index.index_idx()} " f"on {table_name} ({index.joined_column_names()})"
        self.exec_only(statement)
        size = self.exec_fetch(f"select relpages from pg_class c " f"where c.relname = '{index.index_idx()}'", one=True)
        size = size[0]
        index.estimated_size = size * 8 * 1024

    def drop_indexes(self, drop_consistent=False):
        if drop_consistent:
            logging.info("Dropping indexes (consistent)")
            stmt = """
            SELECT
                t.relname AS table_name,
                i.relname AS index_name,
                CASE
                    WHEN idx.indisprimary THEN 'PRIMARY KEY'
                    WHEN idx.indisunique THEN 'UNIQUE CONSTRAINT'
                END AS index_type
            FROM
                pg_index AS idx
            JOIN
                pg_class AS i ON i.oid = idx.indexrelid
            JOIN
                pg_class AS t ON t.oid = idx.indrelid
            JOIN
                pg_namespace AS ns ON ns.oid = t.relnamespace
            WHERE
                (idx.indisprimary OR idx.indisunique)
                AND t.relkind = 'r' -- 普通表
                AND ns.nspname NOT IN ('pg_catalog', 'information_schema') -- 排除系统 schema
                AND ns.nspname NOT LIKE 'pg_toast%' -- 排除 TOAST 表
            ORDER BY
                t.relname,
                index_type DESC;
            """
            indexes = self.exec_fetch(stmt, one=False)
            for table, index in indexes:
                drop_stmt = f"ALTER TABLE {table} DROP CONSTRAINT {index};"
                logging.info(f"Dropping consistent index {table} {index}")
                self.exec_only(drop_stmt)
            logging.info("Dropping consistent indexes finished")

        stmt = """
            SELECT
                t.relname AS table_name,
                i.relname AS index_name
            FROM
                pg_index AS idx
            JOIN
                pg_class AS i ON i.oid = idx.indexrelid
            JOIN
                pg_class AS t ON t.oid = idx.indrelid
            JOIN
                pg_namespace AS ns ON ns.oid = t.relnamespace
            WHERE
                NOT idx.indisprimary
                AND NOT idx.indisunique
                AND t.relkind = 'r' -- 只包括普通表
                AND ns.nspname NOT IN ('pg_catalog', 'information_schema') -- 排除系统表
                AND ns.nspname NOT LIKE 'pg_toast%' -- 排除 TOAST 表
            ORDER BY
                t.relname,
                i.relname;
        """
        indexes = self.exec_fetch(stmt, one=False)
        for table, index in indexes:
            index_name = index
            drop_stmt = "drop index {}".format(index_name)
            logging.info("Dropping index {}".format(index_name))
            self.exec_only(drop_stmt)

    # PostgreSQL expects the timeout in milliseconds
    def exec_query(self, query, timeout=None, cost_evaluation=False):
        # Committing to not lose indexes after timeout
        if not cost_evaluation:
            self._connection.commit()
        query_text = self._prepare_query(query)
        if timeout:
            set_timeout = f"set statement_timeout={timeout}"
            self.exec_only(set_timeout)
        statement = f"explain (analyze, buffers, format json) {query_text}"
        try:
            plan = self.exec_fetch(statement)[0][0]["Plan"]
            result = plan["Actual Total Time"], plan
        except Exception as e:
            logging.error(f"{query.nr}, {e}")
            self._connection.rollback()
            result = None, self._get_plan(query)
        # Disable timeout
        self._cursor.execute("set statement_timeout = 0")
        self._cleanup_query(query)
        return result

    def _cleanup_query(self, query):
        if isinstance(query, str):
            query = query
        else:
            query = query.text
        for query_statement in query.split(";"):
            if "drop view" in query_statement:
                self.exec_only(query_statement)
                self.commit()

    def _get_cost(self, query):
        query_plan = self._get_plan(query)
        total_cost = query_plan["Total Cost"]
        return total_cost

    def get_raw_plan(self, query):
        query_text = self._prepare_query(query)
        statement = f"explain (format json) {query_text}"
        query_plan = self.exec_fetch(statement)[0]
        self._cleanup_query(query)
        return query_plan

    def _get_plan(self, query):
        query_text = self._prepare_query(query)
        statement = f"explain (format json) {query_text}"
        query_plan = self.exec_fetch(statement, one=True)[0][0]["Plan"]
        self._cleanup_query(query)
        return query_plan

    def number_of_indexes(self):
        statement = """select count(*) from pg_indexes
                       where schemaname = 'public'"""
        result = self.exec_fetch(statement)
        return result[0]

    def table_exists(self, table_name):
        statement = f"""SELECT EXISTS (
            SELECT 1
            FROM pg_tables
            WHERE tablename = '{table_name}');"""
        result = self.exec_fetch(statement)
        return result[0]

    def database_exists(self, database_name):
        statement = f"""SELECT EXISTS (
            SELECT 1
            FROM pg_database
            WHERE datname = '{database_name}');"""
        result = self.exec_fetch(statement)
        return result[0]

    def index_exists(self, table_name, column_name):
        statement = f"""SELECT EXISTS (
            SELECT 1 
            FROM pg_indexes
            WHERE tablename = '{table_name}'
            AND indexdef ILIKE '%({column_name})%'
        );"""
        result = self.exec_fetch(statement, one=True)
        if result[0]:
            return 1
        else:
            return 0

    def distinct_percentage(self, table_name, column_name):
        # 查询列的不同值的比例
        statement = f"""
        SELECT COUNT(DISTINCT {column_name})*1.0 / COUNT(*)
        FROM {table_name};
        """
        result = self.exec_fetch(statement, one=True)
        return result[0]

    def get_distinct_count(self, table_name, column_name):
        # 查询列的不同值的数量
        statement = f"""
        SELECT COUNT(DISTINCT {column_name})
        FROM {table_name};
        """
        result = self.exec_fetch(statement, one=True)
        if result[0] == 0:
            return 1
        else:
            return result[0]

    def get_column_data_type(self, table_name, column_name):
        # 查询列数据类型，整数 1，浮点数 2，字符串 3，日期 4，布尔值 5，其他 6
        statement = f"""
        SELECT data_type
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        AND column_name = '{column_name}'; 
        """
        result = self.exec_fetch(statement, one=True)
        data_type = result[0]

        # PostgreSQL数据类型映射
        if data_type in ["integer", "bigint", "smallint", "serial", "bigserial", "smallserial"]:
            return 1
        elif data_type in ["real", "double precision", "numeric", "decimal", "float"]:
            return 2
        elif data_type in ["character varying", "varchar", "character", "char", "text"]:
            return 3
        elif data_type in ["date", "timestamp", "timestamp without time zone", "timestamp with time zone", "time"]:
            return 4
        elif data_type == "boolean":
            return 5
        else:
            return 6

    def get_most_common_value(self, table_name, column_name):
        # 查询列的最常见值对应的百分比，返回为列表
        statement = f"""
            SELECT 
                COALESCE(
                    (SELECT most_common_freqs 
                     FROM pg_stats 
                     WHERE tablename = '{table_name}' 
                     AND attname = '{column_name}'), 
                    ARRAY[]::real[]
                ) || 
                ARRAY_FILL(0::real, ARRAY[GREATEST(50 - COALESCE(array_length(
                    (SELECT most_common_freqs 
                     FROM pg_stats 
                     WHERE tablename = '{table_name}' 
                     AND attname = '{column_name}'), 1), 0), 0)]) AS most_common_freqs_top50;
        """
        result = self.exec_fetch(statement, one=True)
        return result[0][:50]

    def get_column_domain(self, table_name, column_name):
        # 查询列的取值范围，返回为列表
        statement = f"""
        SELECT 
            MIN({column_name}) AS min_value,
            MAX({column_name}) AS max_value
        FROM {table_name};
        """
        try:
            result = self.exec_fetch(statement, one=True)
            return float(result[0]), float(result[1])
        except Exception as e:
            logging.error(f"Error getting column domain for {table_name}.{column_name}: {e}")
            return (0, 0)

    def get_table_rows(self, table_name):
        statement = f"""
        SELECT
            reltuples
        FROM
            pg_class
        WHERE
            relname = '{table_name}';
        """
        result = self.exec_fetch(statement, one=True)
        return result[0]

    def get_table_columns_num(self, table_name):
        statement = f"""
        SELECT
            relnatts
        FROM
            pg_class
        WHERE
            relname = '{table_name}';
        """
        result = self.exec_fetch(statement, one=True)
        return result[0]

    def get_table_in_out_degree(self, table_name):
        statement_in = f"""
        SELECT 
            COUNT(*) AS in_degree
        FROM 
            pg_constraint
        WHERE 
            confrelid = '{table_name}'::regclass
            AND contype = 'f';
        """
        statement_out = f"""
        SELECT
            COUNT(*) AS out_degree
        FROM
            pg_constraint
        WHERE
            conrelid = '{table_name}'::regclass
            AND contype = 'f';
        """
        result_in = self.exec_fetch(statement_in, one=True)
        result_out = self.exec_fetch(statement_out, one=True)
        return result_in[0], result_out[0]

    def get_table_primary_keys_num(self, table_name):
        statement = f"""
        SELECT 
            COUNT(*) AS primary_key_column_count
        FROM 
            pg_constraint AS c
        WHERE 
            c.contype = 'p' 
            AND c.conrelid = '{table_name}'::regclass;
        """
        result = self.exec_fetch(statement, one=True)
        return result[0]

    def get_hist_bounds(self, table_name, column_name, hist_size=100):
        # 获取列的数据类型
        data_type = self.get_column_data_type(table_name, column_name)

        # 根据数据类型选择不同的SQL处理方式
        if data_type == 1:  # 整数类型
            sql = f"""
            WITH min_max AS (
                SELECT MIN({column_name}) as min_val, MAX({column_name}) as max_val
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
            ),
            buckets AS (
                SELECT 
                    width_bucket({column_name}, min_val, max_val + 1, {hist_size}) as bucket_num,
                    COUNT(*) as bucket_count
                FROM {table_name}, min_max
                WHERE {column_name} IS NOT NULL
                GROUP BY width_bucket({column_name}, min_val, max_val + 1, {hist_size})
            )
            SELECT 
                COALESCE(bucket_count, 0) as count
            FROM generate_series(1, {hist_size}) as series(bucket_num)
            LEFT JOIN buckets USING (bucket_num)
            ORDER BY bucket_num;
            """
        elif data_type == 2:  # 浮点数类型
            sql = f"""
            WITH min_max AS (
                SELECT MIN({column_name}) as min_val, MAX({column_name}) as max_val
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
            ),
            buckets AS (
                SELECT 
                    width_bucket({column_name}, min_val, max_val + 0.000001, {hist_size}) as bucket_num,
                    COUNT(*) as bucket_count
                FROM {table_name}, min_max
                WHERE {column_name} IS NOT NULL
                GROUP BY width_bucket({column_name}, min_val, max_val + 0.000001, {hist_size})
            )
            SELECT 
                COALESCE(bucket_count, 0) as count
            FROM generate_series(1, {hist_size}) as series(bucket_num)
            LEFT JOIN buckets USING (bucket_num)
            ORDER BY bucket_num;
            """
        elif data_type == 4:  # 日期类型
            sql = f"""
            WITH date_to_numeric AS (
                SELECT EXTRACT(EPOCH FROM {column_name}) as numeric_val
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
            ),
            min_max AS (
                SELECT MIN(numeric_val) as min_val, MAX(numeric_val) as max_val
                FROM date_to_numeric
            ),
            buckets AS (
                SELECT 
                    width_bucket(numeric_val, min_val, max_val + 1, {hist_size}) as bucket_num,
                    COUNT(*) as bucket_count
                FROM date_to_numeric, min_max
                GROUP BY width_bucket(numeric_val, min_val, max_val + 1, {hist_size})
            )
            SELECT 
                COALESCE(bucket_count, 0) as count
            FROM generate_series(1, {hist_size}) as series(bucket_num)
            LEFT JOIN buckets USING (bucket_num)
            ORDER BY bucket_num;
            """
        else:  # 字符串类型和其他类型
            sql = f"""
            WITH string_to_numeric AS (
                SELECT 
                    ROW_NUMBER() OVER (ORDER BY {column_name}) as numeric_val
                FROM (
                    SELECT DISTINCT {column_name}
                    FROM {table_name}
                    WHERE {column_name} IS NOT NULL
                ) as distinct_vals
            ),
            min_max AS (
                SELECT MIN(numeric_val) as min_val, MAX(numeric_val) as max_val
                FROM string_to_numeric
            ),
            value_mapping AS (
                SELECT 
                    t.{column_name},
                    stn.numeric_val
                FROM {table_name} t
                JOIN (
                    SELECT 
                        {column_name},
                        ROW_NUMBER() OVER (ORDER BY {column_name}) as numeric_val
                    FROM (
                        SELECT DISTINCT {column_name}
                        FROM {table_name}
                        WHERE {column_name} IS NOT NULL
                    ) as distinct_vals
                ) stn ON t.{column_name} = stn.{column_name}
                WHERE t.{column_name} IS NOT NULL
            ),
            buckets AS (
                SELECT 
                    width_bucket(vm.numeric_val, mm.min_val, mm.max_val + 1, {hist_size}) as bucket_num,
                    COUNT(*) as bucket_count
                FROM value_mapping vm, min_max mm
                GROUP BY width_bucket(vm.numeric_val, mm.min_val, mm.max_val + 1, {hist_size})
            )
            SELECT 
                COALESCE(bucket_count, 0) as count
            FROM generate_series(1, {hist_size}) as series(bucket_num)
            LEFT JOIN buckets USING (bucket_num)
            ORDER BY bucket_num;
            """

        try:
            result = self.exec_fetch(sql, one=False)
            hists = [row[0] for row in result]
            # 确保返回的列表长度为hist_size
            while len(hists) < hist_size:
                hists.append(0)
            return hists[:hist_size]
        except Exception as e:
            logging.error(f"Error getting histogram for {table_name}.{column_name}: {e}")
            # 降级到原始方法
            return self._get_hist_bounds_fallback(table_name, column_name, hist_size)

    def _get_hist_bounds_fallback(self, table_name, column_name, hist_size=100):
        """原始实现作为降级方案"""

        def to_vals(data_list):
            for dat in data_list:
                val = dat[0]
                if val is not None:
                    break
            try:
                float(val)
                return np.array(data_list, dtype=float).squeeze()
            except:
                res = []
                val_set = set()
                for dat in data_list:
                    try:
                        mi = dat[0].timestamp()
                    except:
                        mi = len(val_set)
                    val_set.add(dat[0])
                    res.append(mi)
                return np.array(res, dtype=float).squeeze()

        cmd = f"SELECT {column_name} FROM {table_name};"
        result = self.exec_fetch(cmd)
        col_array = to_vals(result)
        hists = np.nanpercentile(col_array, np.array(list(range(0, hist_size, 1))) / 2, axis=0).tolist()
        return hists

    def get_column_is_primary_key(self, table_name, column_name):
        """
        check if the column is a primary key 1 for yes 0 for no
        """
        statement = f"""
        SELECT EXISTS (
            SELECT 1
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = '{table_name}'::regclass
            AND i.indisprimary
            AND a.attname = '{column_name}'
        ) AS is_primary_key;"""
        result = self.exec_fetch(statement, one=True)[0]
        if result:
            return 1
        else:
            return 0

    def get_column_is_foreign_key(self, table_name, column_name):
        statement = f"""
        SELECT EXISTS (
            SELECT 1
            FROM pg_constraint con
            JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = ANY(con.conkey)
            WHERE con.conrelid = '{table_name}'::regclass
            AND con.contype = 'f'
            AND a.attname = '{column_name}'
        ) AS is_foreign_key;"""
        result = self.exec_fetch(statement, one=True)[0]
        if result:
            return 1
        else:
            return 0

    def get_db_size_MB(self):
        statement = f"""
        SELECT ROUND(pg_database_size(current_database()) / 1048576.0, 2) AS size;
        """
        result = self.exec_fetch(statement, one=True)[0]
        return float(result)
