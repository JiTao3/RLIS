import os
import time
import logging
import tempfile
import threading
from pathlib import Path
from typing import Optional, Dict

class SerializableFileLock:

    def __init__(self, name: str, lock_dir: Optional[str] = None, timeout: float = 30.0):
        self.name = name
        self.timeout = timeout
        
        if lock_dir is None:
            lock_dir = tempfile.gettempdir()
        
        self.lock_dir = Path(lock_dir)
        self.lock_dir.mkdir(exist_ok=True)

        safe_name = self._sanitize_name(name)
        self.lock_file_path = self.lock_dir / f"db_lock_{safe_name}.lock"
        self.owner_file_path = self.lock_dir / f"db_lock_{safe_name}.owner"
        
        self._is_owned = False
        self._process_id = os.getpid()
        self._thread_id = threading.get_ident()
        self._local_lock = threading.Lock()
        
    def _sanitize_name(self, name: str) -> str:
        import re
        return re.sub(r'[^\w\-_.]', '_', name)
        
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        with self._local_lock:
            if self._is_owned:
                return True
                
            timeout = timeout or self.timeout
            start_time = time.time()
            
            while True:
                try:
                    with open(self.lock_file_path, 'x') as f:
                        f.write(str(self._process_id))
                    
                    with open(self.owner_file_path, 'w') as f:
                        f.write(f"{self._process_id}\n{self._thread_id}\n{time.time()}")
                    
                    self._is_owned = True
                    logging.debug(f"Acquired lock '{self.name}' by process {self._process_id}")
                    return True
                    
                except FileExistsError:
                    if self._is_stale_lock():
                        self._force_release()
                        continue
                    
                    if not blocking:
                        return False
                    
                    if time.time() - start_time >= timeout:
                        logging.warning(f"Failed to acquire lock '{self.name}' within {timeout}s")
                        return False
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    logging.error(f"Error acquiring lock '{self.name}': {e}")
                    return False
    
    def release(self) -> bool:
        with self._local_lock:
            if not self._is_owned:
                return True
                
            try:
                if not self._verify_ownership():
                    logging.warning(f"Attempting to release lock '{self.name}' not owned by current process/thread")
                    return False
                
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
        try:
            if self.lock_file_path.exists():
                if self._is_stale_lock():
                    self._force_release()
                    return False
                return True
            return False
        except Exception as e:
            logging.error(f"Error checking lock status for '{self.name}': {e}")
            return False
    
    def _verify_ownership(self) -> bool:
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
                
                try:
                    os.kill(owner_pid, 0) 
                    age = time.time() - lock_time
                    if age > self.timeout * 3: 
                        logging.warning(f"Lock '{self.name}' held by PID {owner_pid} is too old ({age:.1f}s)")
                        return True
                    return False
                except OSError:
                    logging.info(f"Removing stale lock '{self.name}' from dead process {owner_pid}")
                    return True
                    
        except Exception as e:
            logging.error(f"Error checking stale lock '{self.name}': {e}")
            return True
    
    def _force_release(self):
        try:
            if self.lock_file_path.exists():
                self.lock_file_path.unlink()
            if self.owner_file_path.exists():
                self.owner_file_path.unlink()
            logging.info(f"Force released stale lock '{self.name}'")
        except Exception as e:
            logging.error(f"Error force releasing lock '{self.name}': {e}")
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def __del__(self):
        if hasattr(self, '_is_owned') and self._is_owned:
            self.release()
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_is_owned'] = False
        state['_local_lock'] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._process_id = os.getpid()
        self._thread_id = threading.get_ident()
        self._is_owned = False
        self._local_lock = threading.Lock()


class SerializableLockDict:
    def __init__(self, lock_dir: Optional[str] = None, default_timeout: float = 30.0):
        self.lock_dir = lock_dir
        self.default_timeout = default_timeout
        self._locks: Dict[str, SerializableFileLock] = {}
    
    def __getitem__(self, key: str) -> SerializableFileLock:
        if key not in self._locks:
            self._locks[key] = SerializableFileLock(
                name=key,
                lock_dir=self.lock_dir,
                timeout=self.default_timeout
            )
        return self._locks[key]
    
    def __setitem__(self, key: str, value):
        if isinstance(value, SerializableFileLock):
            self._locks[key] = value
        else:
            raise ValueError("Value must be SerializableFileLock instance")
    
    def __contains__(self, key: str) -> bool:
        return key in self._locks
    
    def keys(self):
        return self._locks.keys()
    
    def values(self):
        return self._locks.values()
    
    def items(self):
        return self._locks.items()
    
    def get(self, key: str, default=None):
        if key in self._locks:
            return self._locks[key]
        return default
    
    def cleanup_stale_locks(self):
        for lock in self._locks.values():
            if lock._is_stale_lock():
                lock._force_release()
    
    def release_all(self):
        for lock in self._locks.values():
            if lock._is_owned:
                lock.release()
    
    def __getstate__(self):
        return {
            'lock_dir': self.lock_dir,
            'default_timeout': self.default_timeout,
            '_locks': {}
        }
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._locks = {}


def create_db_locks(database_names, lock_dir: Optional[str] = None, timeout: float = 30.0) -> SerializableLockDict:
    locks = SerializableLockDict(lock_dir=lock_dir, default_timeout=timeout)
    
    locks.cleanup_stale_locks()
    
    for db_name in database_names:
        _ = locks[db_name]
    
    return locks
