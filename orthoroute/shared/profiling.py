"""
Zero-cost-when-disabled phase instrumentation for the pathfinder hot loop.

Enable with ORTHO_PROFILE=1. When enabled, ``profile_span(label)`` wraps a
region in an NVTX range (visible in Nsight timelines when CuPy is present)
and accumulates wall time per label; ``log_profile_summary()`` emits one
``[PROFILE] label=1.23s ...`` line and resets the accumulators — call it once
per PathFinder iteration.

When ORTHO_PROFILE is unset the module resolves to a singleton no-op context
manager: entering a span is one attribute lookup and one boolean check that
was made at import time — no dict writes, no perf_counter calls, no NVTX.
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

# Checked once per process at import; the disabled path never re-reads it.
PROFILING_ENABLED = os.getenv("ORTHO_PROFILE") == "1"

# label -> accumulated seconds since the last log_profile_summary()
_accumulated = {}

# NVTX hooks resolve lazily on first enabled span so importing this module
# never imports CuPy.
_nvtx_push = None
_nvtx_pop = None
_nvtx_resolved = False


def _resolve_nvtx():
    global _nvtx_push, _nvtx_pop, _nvtx_resolved
    _nvtx_resolved = True
    try:
        from cupy.cuda import nvtx
        _nvtx_push = nvtx.RangePush
        _nvtx_pop = nvtx.RangePop
    except Exception:
        _nvtx_push = None
        _nvtx_pop = None


class _NullSpan:
    """Shared no-op span used whenever profiling is disabled."""
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


_NULL_SPAN = _NullSpan()


class _Span:
    __slots__ = ("label", "_start")

    def __init__(self, label):
        self.label = label
        self._start = 0.0

    def __enter__(self):
        if not _nvtx_resolved:
            _resolve_nvtx()
        if _nvtx_push is not None:
            _nvtx_push(self.label)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self._start
        if _nvtx_pop is not None:
            _nvtx_pop()
        _accumulated[self.label] = _accumulated.get(self.label, 0.0) + elapsed
        return False


def profile_span(label):
    """Context manager timing a labeled region; no-op unless ORTHO_PROFILE=1."""
    if not PROFILING_ENABLED:
        return _NULL_SPAN
    return _Span(label)


def log_profile_summary(prefix="[PROFILE]"):
    """Log one accumulated-times line and reset. No-op unless enabled."""
    if not PROFILING_ENABLED or not _accumulated:
        return
    parts = " ".join(
        f"{label}={seconds:.2f}s"
        for label, seconds in sorted(
            _accumulated.items(), key=lambda kv: -kv[1]
        )
    )
    logger.info("%s %s", prefix, parts)
    _accumulated.clear()
