from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.workload import Column, Table, Query
import os
import pickle
import logging


class Statistics:
    def __init__(self, database_name, tables, columns):
        self.database_name = database_name  # 保存数据库名称以备后用
        self.connector = None  # 初始化时不创建连接
        self.tables = tables
        self.columns = columns
        self.tables_vec_dict = {}
        self.columns_vec_dict = {}
        self.columns_vec_len = None
        self.columns_domain = {}
        
        # 在初始化时立即准备统计信息
        self.prepare()

    def _ensure_database_connection(self):
        """在需要时重新创建数据库连接"""
        # 检查连接是否不存在或已关闭
        if self.connector is None or not self._is_connection_active():
            if not hasattr(self, 'database_name') or self.database_name is None:
                raise ValueError("Statistics对象缺少database_name属性，无法创建数据库连接")
            
            try:
                # 如果连接存在但已关闭，先清理
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
        """支持 pickle 序列化：排除不可序列化的数据库连接对象"""
        state = self.__dict__.copy()
        # 移除不可序列化的数据库连接
        state['connector'] = None
        return state

    def __setstate__(self, state):
        """支持 pickle 反序列化：状态恢复但不立即创建数据库连接"""
        self.__dict__.update(state)
        # 数据库连接将在需要时创建
        self.connector = None

    def prepare(self):
        # prepare column statistics 
        self._ensure_database_connection()  # 确保连接存在
        
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
        
        # 准备完成后关闭连接
        if self.connector:
            self.connector.close()
            self.connector = None

    @classmethod
    def load_from_cache_or_create(cls, cache_dir, database_name, tables, columns):
        """从缓存加载Statistics或创建新的实例"""
        statistics_pkl_path = os.path.join(cache_dir, f"statistics_{database_name}.pkl")
        
        if os.path.exists(statistics_pkl_path):
            # 读取已保存的statistics
            logging.info(f"Loading cached statistics for database: {database_name}")
            try:
                with open(statistics_pkl_path, 'rb') as f:
                    statistic = pickle.load(f)
                # 更新数据库名称以确保连接正确
                statistic.database_name = database_name
                logging.info(f"Successfully loaded cached statistics for database: {database_name}")
                return statistic
            except Exception as e:
                logging.warning(f"Failed to load cached statistics for {database_name}: {e}")
                logging.info(f"Creating new statistics for database: {database_name}")
        else:
            logging.info(f"Creating new statistics for database: {database_name}")
        
        # 创建新的statistics并保存
        statistic = cls(database_name, tables, columns)
        statistic.save_to_cache(cache_dir)
        return statistic
    
    def save_to_cache(self, cache_dir):
        """保存Statistics到缓存目录"""
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
        """
        安全的数据库调用包装器：在函数调用前确保数据库连接，调用后释放连接
        
        Args:
            func: 需要调用的函数
            *args: 函数的位置参数
            **kwargs: 函数的关键字参数
            
        Returns:
            函数的返回值
        """
        try:
            # 确保数据库连接
            self._ensure_database_connection()
            
            # 调用函数
            result = func(*args, **kwargs)
            return result
            
        except Exception as e:
            logging.error(f"Error during database operation: {e}")
            raise
            
        finally:
            # 释放数据库连接
            if self.connector:
                self.connector.close()
                self.connector = None
                logging.debug(f"Released database connection for {self.database_name}")

    def _is_connection_active(self):
        """检查数据库连接是否活跃"""
        if self.connector is None:
            return False
        
        try:
            # 尝试执行一个简单的查询来检查连接状态
            self.connector.exec_fetch("SELECT 1", one=True)
            return True
        except:
            return False