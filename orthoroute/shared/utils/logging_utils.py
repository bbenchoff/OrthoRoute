"""Logging utilities for OrthoRoute."""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict

from ..configuration.settings import LoggingSettings


def setup_logging(settings: LoggingSettings) -> None:
    """Setup logging configuration based on settings.

    Args:
        settings: Logging settings configuration
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt=settings.format_string,
        datefmt=settings.date_format
    )

    # Console handler with Windows Unicode fix
    if settings.console_output:
        # Fix Windows console Unicode issues
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, settings.level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if settings.file_output:
        try:
            # Ensure log directory exists
            log_path = Path(settings.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Create rotating file handler with Windows-compatible settings
            file_handler = logging.handlers.RotatingFileHandler(
                filename=settings.log_file,
                maxBytes=settings.max_file_size_mb * 1024 * 1024,
                backupCount=settings.backup_count,
                encoding='utf-8',
                delay=True  # Don't open file until first log - prevents Windows locking issues
            )
            file_handler.setLevel(getattr(logging, settings.level.upper()))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        except Exception as e:
            # If file logging fails, log to console
            root_logger.error(f"Failed to setup file logging: {e}")

    # Set component-specific levels
    for component, level in settings.component_levels.items():
        component_logger = logging.getLogger(component)
        component_logger.setLevel(getattr(logging, level.upper()))

    # INFO LOGGING FOR CLEAN DEMO OUTPUT
    root_logger.setLevel(logging.INFO)
    for h in root_logger.handlers:
        h.setLevel(logging.INFO)

    # Add dedicated debug file handler (DEBUG to file only)
    try:
        debug_fh = logging.handlers.RotatingFileHandler(
            "orthoroute_debug.log",
            maxBytes=10_000_000,  # 10 MB
            backupCount=3,
            encoding="utf-8",
            delay=True  # Don't open file until first log - prevents Windows locking issues
        )
        debug_fh.setLevel(logging.DEBUG)
        debug_fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(debug_fh)
        root_logger.info("[LOG] Debug file handler added: orthoroute_debug.log")
    except Exception as e:
        root_logger.error(f"Failed to add debug file handler: {e}")

    # Ensure console shows only INFO and above
    root_logger.info("[LOG] Console logging set to INFO level for clean demo output")

    # Log startup message
    root_logger.debug("[LOG] Debug level enabled on all handlers")
    root_logger.info("OrthoRoute logging initialized")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ContextLogger:
    """Logger with additional context information."""

    def __init__(self, logger: logging.Logger, context: Dict[str, str]):
        """Initialize context logger.

        Args:
            logger: Base logger instance
            context: Context information to include
        """
        self.logger = logger
        self.context = context

    def _format_message(self, message: str) -> str:
        """Format message with context."""
        if self.context:
            context_str = " ".join(f"{k}={v}" for k, v in self.context.items())
            return f"[{context_str}] {message}"
        return message

    def debug(self, message: str, *args, **kwargs):
        """Log debug message with context."""
        self.logger.debug(self._format_message(message), *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log info message with context."""
        self.logger.info(self._format_message(message), *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log warning message with context."""
        self.logger.warning(self._format_message(message), *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log error message with context."""
        self.logger.error(self._format_message(message), *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log critical message with context."""
        self.logger.critical(self._format_message(message), *args, **kwargs)


def init_logging():
    """Initialize logging with INFO+ to orthoroute_debug.log file."""
    # Ensure file handler writes INFO+ and flushes
    fh = logging.handlers.RotatingFileHandler(
        "orthoroute_debug.log",
        maxBytes=10_000_000,  # 10 MB
        backupCount=3,
        encoding="utf-8",
        delay=True  # Don't open file until first log - prevents Windows locking issues
    )
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)

    # Keep console handler as-is (INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # Ensure the UPF module logger is at INFO level for portal logs
    upf_logger = logging.getLogger("orthoroute.algorithms.manhattan.unified_pathfinder")
    upf_logger.setLevel(logging.INFO)


def get_context_logger(name: str, **context) -> ContextLogger:
    """Get a context logger with additional information.

    Args:
        name: Logger name
        **context: Context key-value pairs

    Returns:
        ContextLogger instance
    """
    base_logger = get_logger(name)
    return ContextLogger(base_logger, context)
