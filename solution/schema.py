import importlib
import logging

from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.table_generator import TableGenerator


class Schema(object):
    def __init__(self, benchmark_name, scale_factor):
        generating_connector = PostgresDatabaseConnector(None, autocommit=True)
        table_generator = TableGenerator(
            benchmark_name=benchmark_name.lower(), scale_factor=scale_factor, database_connector=generating_connector
        )

        self.database_name = table_generator.database_name()
        self.tables = table_generator.tables

        self.columns = []
        for table in self.tables:
            for column in table.columns:
                self.columns.append(column)

        self.sqlglot_schema = table_generator.get_sqlglot_schema()

