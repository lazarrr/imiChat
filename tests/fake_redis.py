"""In-memory stand-in for the few redis-py calls the API uses (tests only)."""

import fnmatch


class FakeRedis:
    def __init__(self):
        self.data: dict[str, object] = {}
        self.ttl: dict[str, int] = {}  # key → seconds passed to the last expire()

    def incr(self, key):
        self.data[key] = int(self.data.get(key, 0)) + 1
        return self.data[key]

    def expire(self, key, seconds):
        self.ttl[key] = seconds
        return key in self.data

    def rpush(self, key, *values):
        self.data.setdefault(key, []).extend(values)
        return len(self.data[key])

    def lrange(self, key, start, end):
        items = self.data.get(key, [])
        return list(items[start:] if end == -1 else items[start:end + 1])

    def scan_iter(self, match="*", count=None):
        return iter([key for key in self.data if fnmatch.fnmatch(key, match)])

    def pipeline(self):
        return _FakePipeline(self)


class _FakePipeline:
    """Queues calls and runs them on execute(), like redis-py's pipeline."""

    def __init__(self, redis):
        self._redis = redis
        self._calls = []

    def __getattr__(self, name):
        method = getattr(self._redis, name)

        def queue(*args, **kwargs):
            self._calls.append((method, args, kwargs))
            return self

        return queue

    def execute(self):
        results = [method(*args, **kwargs) for method, args, kwargs in self._calls]
        self._calls = []
        return results
