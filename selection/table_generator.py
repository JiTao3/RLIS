import logging
import os
import platform
import re
import subprocess

from .utils import b_to_mb
from .workload import Column, Table


class TableGenerator:
    def __init__(
        self,
        benchmark_name,
        scale_factor,
        database_connector,
        explicit_database_name=None,
    ):
        self.scale_factor = scale_factor
        self.benchmark_name = benchmark_name
        self.db_connector = database_connector
        self.explicit_database_name = explicit_database_name

        self.database_names = self.db_connector.database_names()
        self.tables = []
        self.columns = []
        self._prepare()
        if self.database_name() not in self.database_names:
            self._generate()
            self.create_database()
        else:
            logging.debug("Database with given scale factor already " "existing")
        self._read_column_names()

    def database_name(self):
        if self.explicit_database_name:
            return self.explicit_database_name

        name = "indexselection_" + self.benchmark_name + "___"
        name += str(self.scale_factor).replace(".", "_")
        return name

    def _postgres_to_sqlglot_type(self, pg_type):
        """将PostgreSQL数据类型映射到SQLGlot支持的类型"""
        # 标准化类型字符串（移除精度、长度等修饰符）
        base_type = re.sub(r'\([^)]*\)', '', pg_type.lower().strip())
        
        type_mapping = {
            # 数值类型
            'integer': 'INT',
            'int': 'INT', 
            'int4': 'INT',
            'bigint': 'BIGINT',
            'int8': 'BIGINT',
            'smallint': 'SMALLINT',
            'int2': 'SMALLINT',
            'decimal': 'DECIMAL',
            'numeric': 'DECIMAL',
            'real': 'REAL',
            'float4': 'REAL',
            'double precision': 'DOUBLE',
            'float8': 'DOUBLE',
            'serial': 'INT',
            'bigserial': 'BIGINT',
            
            # 字符串类型
            'varchar': 'STRING',
            'character varying': 'STRING',
            'char': 'STRING',
            'character': 'STRING',
            'text': 'STRING',
            
            # 日期时间类型
            'date': 'DATE',
            'time': 'TIME', 
            'timestamp': 'TIMESTAMP',
            'timestamptz': 'TIMESTAMP',
            'timestamp with time zone': 'TIMESTAMP',
            'timestamp without time zone': 'TIMESTAMP',
            
            # 布尔类型
            'boolean': 'BOOLEAN',
            'bool': 'BOOLEAN',
            
            # JSON类型
            'json': 'JSON',
            'jsonb': 'JSON',
        }
        
        return type_mapping.get(base_type, 'STRING')  # 默认为STRING类型
    
    def _read_column_names(self):
        """读取并解析表和列名以及数据类型信息"""
        filename = self.directory + "/" + self.create_table_statements_file
        with open(filename, "r") as file:
            data = file.read()
        
        # 使用正则表达式更精确地解析CREATE TABLE语句
        # 匹配完整的CREATE TABLE语句
        table_pattern = r'create\s+table\s+(\w+)\s*\(([^;]+)\);'
        
        for match in re.finditer(table_pattern, data, re.IGNORECASE | re.DOTALL):
            table_name = match.group(1).lower().strip()
            table_content = match.group(2)
            
            table = Table(table_name)
            self.tables.append(table)
            
            # 解析列定义
            # 按行分割并清理
            lines = [line.strip() for line in table_content.split('\n') if line.strip()]
            
            for line in lines:
                # 跳过约束定义
                if (line.lower().startswith('primary key') or 
                    line.lower().startswith('foreign key') or
                    line.lower().startswith('constraint ') or
                    line.lower().startswith('unique ') or
                    line.lower().startswith('check ')):
                    continue
                    
                # 改进的列定义解析逻辑
                line = line.rstrip(',').strip()
                if not line:
                    continue
                
                # 分割为单词
                parts = line.split()
                if len(parts) < 2:
                    logging.warning(f"无效的列定义: {line}")
                    continue
                
                column_name = parts[0].lower()
                
                # 找到约束关键字的位置
                constraint_keywords = ['not', 'null', 'default', 'primary', 'unique', 'check', 'references']
                data_type_parts = []
                
                for i, part in enumerate(parts[1:], 1):
                    # 检查是否遇到约束关键字
                    if part.lower() in constraint_keywords:
                        # 特殊处理：检查是否是 'not null' 组合
                        if part.lower() == 'not' and i + 1 < len(parts) and parts[i + 1].lower() == 'null':
                            break
                        elif part.lower() in ['null', 'default', 'primary', 'unique', 'check', 'references']:
                            break
                    data_type_parts.append(part)
                
                if data_type_parts:
                    data_type = ' '.join(data_type_parts)
                    
                    column_object = Column(column_name)
                    column_object.data_type = data_type  # 保存原始数据类型
                    table.add_column(column_object)
                    self.columns.append(column_object)
                else:
                    logging.warning(f"无法提取数据类型: {line}")
    
    def get_sqlglot_schema(self):
        """
        生成SQLGlot所需的schema格式
        返回: {表名: {列名: SQLGlot类型}}
        """
        schema = {}
        for table in self.tables:
            # if table.name == "on_time_on_time_performance_2016_1":
            #     print(table.columns)
            schema[table.name] = {}
            for column in table.columns:
                # 如果列有data_type属性，则转换为SQLGlot类型
                if hasattr(column, 'data_type') and column.data_type:
                    sqlglot_type = self._postgres_to_sqlglot_type(column.data_type)
                else:
                    # 默认类型
                    sqlglot_type = 'STRING'
                schema[table.name][column.name] = sqlglot_type
        return schema
    
    def print_schema_info(self):
        """打印表结构信息，用于调试"""
        print("=== 数据库表结构信息 ===")
        for table in self.tables:
            print(f"\n表名: {table.name}")
            for column in table.columns:
                data_type = getattr(column, 'data_type', 'N/A')
                sqlglot_type = self._postgres_to_sqlglot_type(data_type) if data_type != 'N/A' else 'N/A'
                print(f"  列名: {column.name:20} 类型: {data_type:20} SQLGlot类型: {sqlglot_type}")

    def _generate(self):
        logging.info("Generating {} data".format(self.benchmark_name))
        logging.info("scale factor: {}".format(self.scale_factor))
        self._run_make()
        self._run_command(self.cmd)
        if self.benchmark_name == "tpcds":
            self._run_command(["bash", "../../scripts/replace_in_dat.sh"])
        logging.info("[Generate command] " + " ".join(self.cmd))
        self._table_files()
        logging.info("Files generated: {}".format(self.table_files))

    def create_database(self):
        self.db_connector.create_database(self.database_name())
        filename = self.directory + "/" + self.create_table_statements_file
        with open(filename, "r") as file:
            create_statements = file.read()
        # Do not create primary keys
        create_statements = re.sub(r",\s*primary key (.*)", "", create_statements)
        self.db_connector.db_name = self.database_name()
        self.db_connector.create_connection()
        self.create_tables(create_statements)
        self._load_table_data(self.db_connector)
        self.db_connector.enable_simulation()

    def create_tables(self, create_statements):
        logging.info("Creating tables")
        for create_statement in create_statements.split(";")[:-1]:
            self.db_connector.exec_only(create_statement)
        self.db_connector.commit()

    def _load_table_data(self, database_connector):
        logging.info("Loading data into the tables")
        for filename in self.table_files:
            logging.debug("    Loading file {}".format(filename))

            table = filename.replace(".tbl", "").replace(".dat", "")
            path = self.directory + "/" + filename
            size = os.path.getsize(path)
            size_string = f"{b_to_mb(size):,.4f} MB"
            logging.debug(f"    Import data of size {size_string}")
            database_connector.import_data(table, path)
            os.remove(os.path.join(self.directory, filename))
        database_connector.commit()

    def _run_make(self):
        if "dbgen" not in self._files() and "dsdgen" not in self._files():
            logging.info("Running make in {}".format(self.directory))
            self._run_command(self.make_command)
        else:
            logging.info("No need to run make")

    def _table_files(self):
        self.table_files = [x for x in self._files() if ".tbl" in x or ".dat" in x]

    def _run_command(self, command):
        cmd_out = "[SUBPROCESS OUTPUT] "
        p = subprocess.Popen(
            command,
            cwd=self.directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with p.stdout:
            for line in p.stdout:
                logging.info(cmd_out + line.decode("utf-8").replace("\n", ""))
        p.wait()

    def _files(self):
        return os.listdir(self.directory)

    def _prepare(self):
        database_name_list = [
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
        ]
        if self.benchmark_name == "tpch":
            self.make_command = ["make", "DATABASE=POSTGRESQL"]
            if platform.system() == "Darwin":
                self.make_command.append("MACHINE=MACOS")

            self.directory = "./index_selection_evaluation/tpch-kit/dbgen"
            self.create_table_statements_file = "dss.ddl"
            self.cmd = ["./dbgen", "-s", str(self.scale_factor), "-f"]
        elif self.benchmark_name == "tpcds":
            self.make_command = ["make"]
            if platform.system() == "Darwin":
                self.make_command.append("OS=MACOS")

            self.directory = "./index_selection_evaluation/tpcds-kit/tools"
            self.create_table_statements_file = "tpcds.sql"
            self.cmd = ["./dsdgen", "-SCALE", str(self.scale_factor), "-FORCE"]

            # 0.001 is allowed for testing
            if (
                int(self.scale_factor) - self.scale_factor != 0
                and self.scale_factor != 0.001
            ):
                raise Exception("Wrong TPCDS scale factor")
        elif self.benchmark_name in database_name_list:
            self.directory = "/home/weixun/data/db_workload/datasets/" + self.benchmark_name
            self.create_table_statements_file = "postgres_create_" + self.benchmark_name + ".sql"
            self.explicit_database_name = self.benchmark_name
        else:
            raise NotImplementedError("only tpch/ds implemented.")
