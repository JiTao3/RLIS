from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.workload import Column, Table, Query
import os
import pickle
import logging


class Statistics:
    def __init__(self, database_name, tables, columns):
        self.database_name = database_name
        self.connector = None
        self.tables = tables
        self.columns = columns
        self.tables_vec_dict = {}
        self.columns_vec_dict = {}
        self.columns_vec_len = None
        self.columns_domain = {}
        self.prepare()

    def _ensure_database_connection(self):

        if self.connector is None or not self._is_connection_active():
            if not hasattr(self, 'database_name') or self.database_name is None:
                raise ValueError("Statistics对象缺少database_name属性，无法创建数据库连接")
            
            try:
                if self.connector is not None:
                    try:
                        self.connector.close()
                    except:
                        pass
                    self.connector = None
                
                self.connector = PostgresDatabaseConnector(self.database_name, autocommit=True)
                logging.debug(f"成功创建到数据库 {self.database_name} 的连接")
            except Exception as e:
                logging.error(f"创建数据库连接失败: {e}")
                raise RuntimeError(f"无法连接到数据库 {self.database_name}: {e}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state['connector'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.connector = None

    def prepare(self):
        self._ensure_database_connection()
        
        for table in self.tables:
            table.prepare(self.connector)
            if table.name not in self.tables_vec_dict:
                self.tables_vec_dict[table.name] = table.to_vector()
            else:
                raise ValueError("Table name already exists")
        for column in self.columns:
            column_full_name = f"{column.table.name}.{column.name}"
            column.prepare(self.connector)
            if column_full_name not in self.columns_vec_dict:
                self.columns_vec_dict[column_full_name] = column.to_vector()
                if self.columns_vec_len is None:
                    self.columns_vec_len = self.columns_vec_dict[column_full_name].shape[0]
            else:
                raise ValueError("Column name already exists")
            self.columns_domain[column_full_name] = column.domain
        
        if self.connector:
            self.connector.close()
            self.connector = None

    @classmethod
    def load_from_cache_or_create(cls, cache_dir, database_name, tables, columns):
        statistics_pkl_path = os.path.join(cache_dir, f"statistics_{database_name}.pkl")
        
        if os.path.exists(statistics_pkl_path):
            logging.info(f"Loading cached statistics for database: {database_name}")
            try:
                with open(statistics_pkl_path, 'rb') as f:
                    statistic = pickle.load(f)
                statistic.database_name = database_name
                logging.info(f"Successfully loaded cached statistics for database: {database_name}")
                return statistic
            except Exception as e:
                logging.warning(f"Failed to load cached statistics for {database_name}: {e}")
                logging.info(f"Creating new statistics for database: {database_name}")
        else:
            logging.info(f"Creating new statistics for database: {database_name}")
        statistic = cls(database_name, tables, columns)
        statistic.save_to_cache(cache_dir)
        return statistic
    
    def save_to_cache(self, cache_dir):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        statistics_pkl_path = os.path.join(cache_dir, f"statistics_{self.database_name}.pkl")
        
        try:
            logging.info(f"Saving statistics cache for database: {self.database_name}")
            with open(statistics_pkl_path, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Successfully saved statistics cache for database: {self.database_name}")
        except Exception as e:
            logging.error(f"Failed to save statistics cache for {self.database_name}: {e}")

    def call_db(self, func, *args, **kwargs):
        try:
            self._ensure_database_connection()
            result = func(*args, **kwargs)
            return result
            
        except Exception as e:
            logging.error(f"Error during database operation: {e}")
            raise
            
        finally:
            if self.connector:
                self.connector.close()
                self.connector = None
                logging.debug(f"Released database connection for {self.database_name}")

    def _is_connection_active(self):
        if self.connector is None:
            return False
        
        try:
            self.connector.exec_fetch("SELECT 1", one=True)
            return True
        except:
            return False