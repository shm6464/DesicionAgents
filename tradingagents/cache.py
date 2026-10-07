"""Two-layer response cache for LLM calls and market-data requests.

The framework called the LLM and its market-data vendors afresh on every run,
so re-running the same ticker+date (or a backtest over a grid) re-billed every
request. This module adds a response cache with two backends, selected by
``cache_backend`` in the config:

- ``"redis"`` — a real Redis server, configured via ``redis_url``. Needs the
  optional ``redis`` package (``pip install tradingagents[redis]``).
- ``"disk"``  — a local, dependency-free fallback that stores entries as JSON
  files under ``data_cache_dir``. This is the default and always works, so the
  project still runs with no Redis installed.

Both backends expose the same tiny interface (``get`` / ``set`` / ``clear`` /
``clear_scope``), so callers never branch on the backend. Caching is opt-out:
set ``cache_enabled: false`` (or ``TRADINGAGENTS_CACHE_ENABLED=false``) to turn
it off entirely, and ``cache.clear()`` drops everything.

Point-in-time safety
--------------------
Market data is cached *after* the run date has been clamped into the request
(``as_of`` / ``as_of_window`` run in ``tools.py`` before ``route_to_vendor``),
and every market-data key embeds the resolved request arguments including the
dates, so a request for a later date can never read a value cached for an
earlier one. Keys are namespaced ``llm:`` / ``data:`` so the two layers can be
cleared independently.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cache version, bumped when the key layout or serialization changes so stale
# entries written by an older build are ignored rather than misread.
_CACHE_VERSION = "v1"


def _stable_dump(value: Any) -> str:
    """A deterministic JSON string for key building (sorted keys, UTF-8 safe)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def build_key(namespace: str, parts: dict[str, Any]) -> str:
    """A namespaced, versioned, collision-resistant cache key.

    ``parts`` are the discriminator fields (provider, model, args, prompt
    digest, dates, ...). Order is normalized via ``sort_keys`` so the caller
    can pass a plain dict without worrying about insertion order.
    """
    digest = hashlib.sha256(_stable_dump(parts).encode("utf-8")).hexdigest()
    return f"{namespace}:{_CACHE_VERSION}:{digest}"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _BaseStore:
    """Common key/TTL handling shared by the backends."""

    def __init__(self, ttl: int | None, enabled: bool = True):
        self.ttl = ttl
        self.enabled = enabled

    def _expired(self, stored_at: float) -> bool:
        return self.ttl is not None and (time.time() - stored_at) > self.ttl


class _DiskStore(_BaseStore):
    """JSON-file-backed cache under ``data_cache_dir``; no third-party deps."""

    def __init__(self, cache_dir: str | os.PathLike, ttl: int | None, enabled: bool = True):
        super().__init__(ttl, enabled)
        # Keys contain ':' (Redis style); filesystem is happier with '_'.
        self.root = Path(cache_dir) / "response_cache"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Keys are hex digests already; flattening ':' avoids nested dirs.
        return self.root / f"{key.replace(':', '_')}.json"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if self._expired(float(payload.get("stored_at", 0))):
            path.unlink(missing_ok=True)
            return None
        return payload.get("value")

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        payload = {"stored_at": time.time(), "value": value}
        try:
            self._path(key).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:  # never let a cache write abort the run
            logger.debug("cache write skipped for %s: %s", key, exc)

    def clear(self) -> None:
        for path in self.root.glob("*.json"):
            path.unlink(missing_ok=True)

    def clear_scope(self, namespace: str) -> None:
        prefix = f"{namespace}_{_CACHE_VERSION}_"
        for path in self.root.glob(f"{prefix}*.json"):
            path.unlink(missing_ok=True)


class _RedisStore(_BaseStore):
    """Redis-backed cache; ``redis`` is imported lazily so it stays optional."""

    def __init__(self, redis_url: str, ttl: int | None, enabled: bool = True):
        super().__init__(ttl, enabled)
        import redis  # noqa: WPS433 — optional dependency, loaded on demand

        self.client = redis.Redis.from_url(redis_url, decode_responses=True)

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        try:
            raw = self.client.get(key)
        except Exception as exc:  # Redis down must not crash the run
            logger.warning("cache read failed (falling through): %s", exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        try:
            self.client.set(key, json.dumps(value, ensure_ascii=False, default=str),
                            ex=self.ttl)
        except Exception as exc:
            logger.debug("cache write skipped for %s: %s", key, exc)

    def clear(self) -> None:
        try:
            self.client.flushdb()
        except Exception as exc:
            logger.warning("cache clear failed: %s", exc)

    def clear_scope(self, namespace: str) -> None:
        # Redis has no prefix scan without iterating; drop matching keys by pattern.
        try:
            pattern = f"{namespace}:{_CACHE_VERSION}:*"
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor=cursor, match=pattern, count=200)
                if keys:
                    self.client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as exc:
            logger.warning("cache clear_scope failed: %s", exc)


def _coerce_bool(value: Any) -> bool:
    """Parse a config bool that may arrive as a string (env overrides).

    Mirrors ``default_config._BOOL_TRUE/_BOOL_FALSE``; a plain bool passes
    through, ``"false"/"0"/"off"/"no"`` become False, anything else truthy.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(value)


def _make_store(config: dict) -> _BaseStore:
    backend = (config.get("cache_backend") or "disk").lower()
    ttl = config.get("cache_ttl")
    # Env-var overrides arrive as strings (the config has no type info for a
    # None default), so coerce an explicit numeric string the same way the LLM
    # kwargs layer does for temperature/max_tokens.
    if ttl is not None and ttl != "":
        ttl = int(ttl)
    enabled = _coerce_bool(config.get("cache_enabled", True))
    if backend == "redis":
        url = config.get("redis_url")
        if not url:
            logger.warning("cache_backend=redis but redis_url is unset; falling back to disk cache")
            return _DiskStore(config["data_cache_dir"], ttl, enabled)
        return _RedisStore(url, ttl, enabled)
    return _DiskStore(config["data_cache_dir"], ttl, enabled)


class ResponseCache:
    """A thin, backend-agnostic cache facade.

    Instantiate once per run (the graph does, in ``TradingAgentsGraph.__init__``)
    and share it with the LLM and data layers. ``get_or_set`` is the one method
    callers use; it logs a clear HIT/MISS so cache behavior is visible in the
    logs.
    """

    def __init__(self, config: dict):
        self.config = config
        self.enabled = _coerce_bool(config.get("cache_enabled", True))
        self._store = _make_store(config)
        backend = config.get("cache_backend") or "disk"
        if not self.enabled:
            logger.info("response cache disabled (cache_enabled=false)")
        else:
            logger.info("response cache ready (backend=%s, ttl=%s)", backend, config.get("cache_ttl"))

    def get(self, namespace: str, parts: dict[str, Any]) -> Any | None:
        """Return the cached value for a namespaced key, or ``None`` on miss."""
        if not self.enabled:
            return None
        key = build_key(namespace, parts)
        value = self._store.get(key)
        if value is not None:
            logger.debug("cache HIT  %s", key)
        else:
            logger.debug("cache MISS %s", key)
        return value

    def set(self, namespace: str, parts: dict[str, Any], value: Any) -> None:
        if not self.enabled:
            return
        self._store.set(build_key(namespace, parts), value)

    def clear(self) -> None:
        """Drop the entire cache (both layers)."""
        self._store.clear()
        logger.info("response cache cleared")

    def clear_scope(self, namespace: str) -> None:
        """Drop one namespace (``llm`` or ``data``) only."""
        self._store.clear_scope(namespace)
        logger.info("response cache cleared for namespace %r", namespace)


# Convenience: a module-level cache built from the process config, so the data
# layer (which runs inside graph tool calls and does not hold a graph reference)
# can reach the same cache instance the graph built.
_module_cache: ResponseCache | None = None


def get_cache(config: dict | None = None) -> ResponseCache:
    """The process-wide cache, built lazily from ``config`` (or the run config)."""
    global _module_cache
    if _module_cache is None:
        if config is None:
            from tradingagents.dataflows.config import get_config

            config = get_config()
        _module_cache = ResponseCache(config)
    return _module_cache


def set_cache(cache: ResponseCache) -> None:
    """Register a cache instance as the process-wide one.

    ``TradingAgentsGraph.__init__`` calls this so the data layer (which runs
    inside graph tool calls and cannot see the graph object) reaches the exact
    same cache the graph built, sharing its backend and hit state.
    """
    global _module_cache
    _module_cache = cache


def reset_cache() -> None:
    """Drop the module-level cache (tests and config reloads use this)."""
    global _module_cache
    _module_cache = None
