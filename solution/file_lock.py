import os
import time
import logging
import tempfile
import threading
from pathlib import Path
from typing import Optional, Dict

class SerializableFileLock:
    """
    基于文件的可序列化锁，支持跨进程使用
    """
    
    def __init__(self, name: str, lock_dir: Optional[str] = None, timeout: float = 30.0):
        """
        初始化文件锁
        
        Args:
            name: 锁的名称（用于生成锁文件名）
            lock_dir: 锁文件存储目录，如果为None则使用系统临时目录
            timeout: 获取锁的超时时间（秒）
        """
        self.name = name
        self.timeout = timeout
        
        # 设置锁文件路径
        if lock_dir is None:
            lock_dir = tempfile.gettempdir()
        
        self.lock_dir = Path(lock_dir)
        self.lock_dir.mkdir(exist_ok=True)
        
        # 锁文件路径
        safe_name = self._sanitize_name(name)
        self.lock_file_path = self.lock_dir / f"db_lock_{safe_name}.lock"
        self.owner_file_path = self.lock_dir / f"db_lock_{safe_name}.owner"
        
        # 当前进程是否持有锁
        self._is_owned = False
        self._process_id = os.getpid()
        self._thread_id = threading.get_ident()
        self._local_lock = threading.Lock()  # 同进程内线程安全
        
    def _sanitize_name(self, name: str) -> str:
        """清理锁名称，确保可以用作文件名"""
        # 替换不安全的文件名字符
        import re
        return re.sub(r'[^\w\-_.]', '_', name)
        
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        获取锁
        
        Args:
            blocking: 是否阻塞等待
            timeout: 超时时间，如果为None则使用初始化时的timeout
            
        Returns:
            True if lock acquired, False otherwise
        """
        with self._local_lock:
            if self._is_owned:
                return True
                
            timeout = timeout or self.timeout
            start_time = time.time()
            
            while True:
                try:
                    # 尝试创建锁文件（原子操作）
                    with open(self.lock_file_path, 'x') as f:
                        f.write(str(self._process_id))
                    
                    # 记录锁的所有者信息
                    with open(self.owner_file_path, 'w') as f:
                        f.write(f"{self._process_id}\n{self._thread_id}\n{time.time()}")
                    
                    self._is_owned = True
                    logging.debug(f"Acquired lock '{self.name}' by process {self._process_id}")
                    return True
                    
                except FileExistsError:
                    # 锁文件已存在，检查是否是僵尸锁
                    if self._is_stale_lock():
                        self._force_release()
                        continue
                    
                    if not blocking:
                        return False
                    
                    # 检查超时
                    if time.time() - start_time >= timeout:
                        logging.warning(f"Failed to acquire lock '{self.name}' within {timeout}s")
                        return False
                    
                    # 等待一小段时间后重试
                    time.sleep(0.1)
                    
                except Exception as e:
                    logging.error(f"Error acquiring lock '{self.name}': {e}")
                    return False
    
    def release(self) -> bool:
        """
        释放锁
        
        Returns:
            True if lock released, False otherwise
        """
        with self._local_lock:
            if not self._is_owned:
                return True
                
            try:
                # 验证当前进程/线程确实持有这个锁
                if not self._verify_ownership():
                    logging.warning(f"Attempting to release lock '{self.name}' not owned by current process/thread")
                    return False
                
                # 删除锁文件
                if self.lock_file_path.exists():
                    self.lock_file_path.unlink()
                
                if self.owner_file_path.exists():
                    self.owner_file_path.unlink()
                
                self._is_owned = False
                logging.debug(f"Released lock '{self.name}' by process {self._process_id}")
                return True
                
            except Exception as e:
                logging.error(f"Error releasing lock '{self.name}': {e}")
                return False
    
    def locked(self) -> bool:
        """
        检查锁是否被任何进程持有
        
        Returns:
            True if lock is held, False otherwise
        """
        try:
            if self.lock_file_path.exists():
                # 检查是否是僵尸锁
                if self._is_stale_lock():
                    # 如果是僵尸锁，清理它并返回False
                    self._force_release()
                    return False
                return True
            return False
        except Exception as e:
            logging.error(f"Error checking lock status for '{self.name}': {e}")
            return False
    
    def _verify_ownership(self) -> bool:
        """验证当前进程/线程是否真的持有这个锁"""
        try:
            if not self.owner_file_path.exists():
                return False
            
            with open(self.owner_file_path, 'r') as f:
                lines = f.read().strip().split('\n')
                if len(lines) >= 2:
                    owner_pid = int(lines[0])
                    owner_tid = int(lines[1])
                    return (owner_pid == self._process_id and 
                            owner_tid == self._thread_id)
            return False
        except Exception:
            return False
    
    def _is_stale_lock(self) -> bool:
        """
        检查是否是僵尸锁（持有锁的进程已经不存在）
        
        Returns:
            True if lock is stale, False otherwise
        """
        try:
            if not self.owner_file_path.exists():
                return True
            
            with open(self.owner_file_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    return True
                
                lines = content.split('\n')
                if len(lines) < 3:
                    return True
                
                try:
                    owner_pid = int(lines[0])
                    lock_time = float(lines[2])
                except (ValueError, IndexError):
                    return True
                
                # 检查进程是否还存在
                try:
                    os.kill(owner_pid, 0)  # 信号0不会杀死进程，只是检查进程是否存在
                    # 进程存在，检查锁是否太老
                    age = time.time() - lock_time
                    if age > self.timeout * 3:  # 如果锁的年龄超过超时时间的3倍
                        logging.warning(f"Lock '{self.name}' held by PID {owner_pid} is too old ({age:.1f}s)")
                        return True
                    return False
                except OSError:
                    # 进程不存在
                    logging.info(f"Removing stale lock '{self.name}' from dead process {owner_pid}")
                    return True
                    
        except Exception as e:
            logging.error(f"Error checking stale lock '{self.name}': {e}")
            return True
    
    def _force_release(self):
        """强制释放锁（清理僵尸锁）"""
        try:
            if self.lock_file_path.exists():
                self.lock_file_path.unlink()
            if self.owner_file_path.exists():
                self.owner_file_path.unlink()
            logging.info(f"Force released stale lock '{self.name}'")
        except Exception as e:
            logging.error(f"Error force releasing lock '{self.name}': {e}")
    
    def __enter__(self):
        """支持 with 语句"""
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句"""
        self.release()
    
    def __del__(self):
        """析构函数，确保锁被释放"""
        if hasattr(self, '_is_owned') and self._is_owned:
            self.release()
    
    def __getstate__(self):
        """支持pickle序列化"""
        state = self.__dict__.copy()
        # 序列化时重置所有权状态和线程锁
        state['_is_owned'] = False
        state['_local_lock'] = None
        return state
    
    def __setstate__(self, state):
        """支持pickle反序列化"""
        self.__dict__.update(state)
        # 反序列化后更新进程ID和线程ID
        self._process_id = os.getpid()
        self._thread_id = threading.get_ident()
        self._is_owned = False
        self._local_lock = threading.Lock()


class SerializableLockDict:
    """
    模拟 Manager().dict() 行为的可序列化锁字典
    """
    
    def __init__(self, lock_dir: Optional[str] = None, default_timeout: float = 30.0):
        self.lock_dir = lock_dir
        self.default_timeout = default_timeout
        self._locks: Dict[str, SerializableFileLock] = {}
    
    def __getitem__(self, key: str) -> SerializableFileLock:
        """获取指定名称的锁"""
        if key not in self._locks:
            self._locks[key] = SerializableFileLock(
                name=key,
                lock_dir=self.lock_dir,
                timeout=self.default_timeout
            )
        return self._locks[key]
    
    def __setitem__(self, key: str, value):
        """设置锁（通常不需要，但为了兼容性）"""
        if isinstance(value, SerializableFileLock):
            self._locks[key] = value
        else:
            raise ValueError("Value must be SerializableFileLock instance")
    
    def __contains__(self, key: str) -> bool:
        """检查是否包含指定的锁"""
        return key in self._locks
    
    def keys(self):
        """返回所有锁的名称"""
        return self._locks.keys()
    
    def values(self):
        """返回所有锁对象"""
        return self._locks.values()
    
    def items(self):
        """返回所有锁的键值对"""
        return self._locks.items()
    
    def get(self, key: str, default=None):
        """获取锁，如果不存在返回默认值"""
        if key in self._locks:
            return self._locks[key]
        return default
    
    def cleanup_stale_locks(self):
        """清理所有僵尸锁"""
        for lock in self._locks.values():
            if lock._is_stale_lock():
                lock._force_release()
    
    def release_all(self):
        """释放所有当前进程持有的锁"""
        for lock in self._locks.values():
            if lock._is_owned:
                lock.release()
    
    def __getstate__(self):
        """支持pickle序列化"""
        return {
            'lock_dir': self.lock_dir,
            'default_timeout': self.default_timeout,
            '_locks': {}  # 不序列化锁实例，让子进程重新创建
        }
    
    def __setstate__(self, state):
        """支持pickle反序列化"""
        self.__dict__.update(state)
        self._locks = {}


def create_db_locks(database_names, lock_dir: Optional[str] = None, timeout: float = 30.0) -> SerializableLockDict:
    """
    创建数据库锁字典，用于替代 Manager().dict()
    
    Args:
        database_names: 数据库名称列表
        lock_dir: 锁文件存储目录
        timeout: 默认超时时间
        
    Returns:
        SerializableLockDict instance
    """
    locks = SerializableLockDict(lock_dir=lock_dir, default_timeout=timeout)
    
    # 清理锁文件夹中的僵尸锁
    locks.cleanup_stale_locks()
    
    # 预创建所有数据库的锁
    for db_name in database_names:
        _ = locks[db_name]  # 这会触发锁的创建
    
    return locks
