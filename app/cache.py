import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional


class TTLStore:
    def __init__(self, max_size: int = 1000, ttl: float = 1800):
        self._store: OrderedDict[str, tuple] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        if key not in self._store:
            return None
        value, ts = self._store[key]
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: List[Dict[str, Any]]):
        self._store[key] = (value, time.time())
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def cleanup(self):
        now = time.time()
        expired = [k for k, (_, ts) in self._store.items() if now - ts > self._ttl]
        for k in expired:
            del self._store[k]
