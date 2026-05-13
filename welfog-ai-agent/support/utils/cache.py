from datetime import datetime

_cache = {}
_cache_ttl_sec = 60


def _cache_get(key):
    try:
        item = _cache.get(key)
        if not item:
            return None
        ts, val = item
        if (datetime.now().timestamp() - ts) > _cache_ttl_sec:
            _cache.pop(key, None)
            return None
        return val
    except Exception:
        return None


def _cache_set(key, val):
    try:
        if len(_cache) > 300:
            _cache.clear()
        _cache[key] = (datetime.now().timestamp(), val)
    except Exception:
        pass
