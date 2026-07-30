"""Unit tests for orthoroute.shared.utils.performance_utils."""
import logging
import os
import time

import pytest


class TestProfileTime:
    """@profile_time decorator behaviour."""

    def test_returns_correct_value(self, monkeypatch):
        """Decorated function must return its original return value."""
        monkeypatch.setenv("ORTHO_DEBUG", "1")
        from orthoroute.shared.utils.performance_utils import profile_time

        @profile_time
        def add(a, b):
            return a + b

        assert add(3, 4) == 7

    def test_noop_without_debug(self, monkeypatch):
        """Without ORTHO_DEBUG the decorator must not emit WARNING log records."""
        monkeypatch.delenv("ORTHO_DEBUG", raising=False)
        from orthoroute.shared.utils.performance_utils import profile_time

        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Capture(level=logging.WARNING)
        logging.getLogger().addHandler(handler)
        try:
            @profile_time
            def noop():
                return 42

            result = noop()
            assert result == 42
            profile_records = [r for r in records if "[PROFILE]" in r.getMessage()]
            assert not profile_records, "profile_time emitted [PROFILE] without ORTHO_DEBUG"
        finally:
            logging.getLogger().removeHandler(handler)

    def test_emits_profile_log_with_debug(self, monkeypatch, caplog):
        """With ORTHO_DEBUG=1, [PROFILE] must appear at WARNING level."""
        monkeypatch.setenv("ORTHO_DEBUG", "1")
        from importlib import reload
        import orthoroute.shared.utils.performance_utils as pu
        reload(pu)

        @pu.profile_time
        def slow():
            return "done"

        with caplog.at_level(logging.WARNING):
            slow()

        profile_msgs = [r for r in caplog.records if "[PROFILE]" in r.message]
        assert profile_msgs, "[PROFILE] log not emitted with ORTHO_DEBUG=1"
