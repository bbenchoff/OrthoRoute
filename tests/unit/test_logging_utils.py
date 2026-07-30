"""Unit tests for orthoroute.shared.utils.logging_utils."""
import logging
import os

import pytest


class TestInitLogging:
    """Verify init_logging() produces the correct handler configuration."""

    def test_normal_mode_console_level(self, tmp_path, monkeypatch):
        """Console handler must be ERROR in normal mode."""
        monkeypatch.delenv("ORTHO_DEBUG", raising=False)
        monkeypatch.chdir(tmp_path)
        # Clear existing handlers to avoid cross-test pollution
        root = logging.getLogger()
        root.handlers.clear()
        from orthoroute.shared.utils.logging_utils import init_logging
        init_logging()
        root = logging.getLogger()
        console_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert console_handlers, "No StreamHandler found after init_logging()"
        assert console_handlers[0].level == logging.ERROR, (
            f"Console handler level {console_handlers[0].level} != ERROR in normal mode"
        )

    def test_debug_mode_file_level(self, tmp_path, monkeypatch):
        """File handler must be DEBUG when ORTHO_DEBUG=1."""
        monkeypatch.setenv("ORTHO_DEBUG", "1")
        monkeypatch.chdir(tmp_path)
        root = logging.getLogger()
        root.handlers.clear()
        from importlib import reload
        import orthoroute.shared.utils.logging_utils as lu
        reload(lu)
        lu.init_logging()
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert file_handlers, "No FileHandler found after init_logging() with ORTHO_DEBUG=1"
        assert file_handlers[0].level == logging.DEBUG, (
            f"File handler level {file_handlers[0].level} != DEBUG in debug mode"
        )

    def test_no_debug_console_filters_warnings(self, tmp_path, monkeypatch):
        """In normal mode, console StreamHandler level must be > WARNING."""
        monkeypatch.delenv("ORTHO_DEBUG", raising=False)
        monkeypatch.chdir(tmp_path)
        root = logging.getLogger()
        root.handlers.clear()
        from orthoroute.shared.utils.logging_utils import init_logging
        init_logging()
        root = logging.getLogger()
        console = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        if console:
            assert console[0].level > logging.WARNING, (
                f"Console level {console[0].level} allows WARNING messages in normal mode"
            )
