#!/usr/bin/env python3
"""Native KiCad IPC action entrypoint for OrthoRoute."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parent


def main() -> int:
    """Launch OrthoRoute against the KiCad instance that started this action."""
    os.chdir(PLUGIN_ROOT)
    if str(PLUGIN_ROOT) not in sys.path:
        sys.path.insert(0, str(PLUGIN_ROOT))

    from main import run_plugin

    run_plugin(show_gui=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
