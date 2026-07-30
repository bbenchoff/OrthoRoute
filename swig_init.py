"""KiCad 9/10 ActionPlugin bridge for OrthoRoute.

Adapted from PR #17 (commit deee9a6).  KiCad loads this tiny wrapper through
the legacy ActionPlugin system so the toolbar button remains available even
when native IPC action discovery is unreliable.  The button launches the
separately packaged IPC application in KiCad's managed Python environment.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pcbnew


PLUGIN_IDENTIFIER = "com.github.bbenchoff.orthoroute"


class OrthoRoutePlugin(pcbnew.ActionPlugin):
    """Toolbar launcher that hands execution to the native IPC plugin."""

    def defaults(self):
        self.name = "OrthoRoute"
        self.category = "Routing"
        self.description = "GPU-accelerated PCB autorouter"
        self.show_toolbar_button = True
        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        self.icon_file_name = os.path.join(self.plugin_dir, "icon-24.png")

    def _kicad_version(self) -> str:
        try:
            return pcbnew.GetMajorMinorVersion()
        except AttributeError:
            build = str(pcbnew.GetBuildVersion())
            return ".".join(build.split(".")[:2])

    def _native_plugin_dir(self) -> Path:
        # .../<version>/3rdparty/plugins/<swig package>
        version_root = Path(self.plugin_dir).resolve().parents[2]
        return version_root / "plugins" / PLUGIN_IDENTIFIER

    def _python_executable(self) -> Path | None:
        version = self._kicad_version()
        candidates = []

        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            local_root = Path(local_app_data)
            candidates.extend(
                [
                    local_root
                    / "KiCad"
                    / version
                    / "python-environments"
                    / PLUGIN_IDENTIFIER
                    / "Scripts"
                    / "pythonw.exe",
                    local_root
                    / "Programs"
                    / "KiCad"
                    / version
                    / "bin"
                    / "pythonw.exe",
                ]
            )

        program_files = os.environ.get("PROGRAMFILES")
        if program_files:
            candidates.append(
                Path(program_files) / "KiCad" / version / "bin" / "pythonw.exe"
            )

        candidates.extend([Path(sys.executable), Path("python3")])
        return next((path for path in candidates if path.is_file()), None)

    def Run(self):
        import wx

        native_dir = self._native_plugin_dir()
        main_py = native_dir / "main.py"
        python_exe = self._python_executable()

        if not main_py.is_file():
            wx.MessageBox(
                f"Cannot find the OrthoRoute IPC package:\n{native_dir}",
                "OrthoRoute Error",
                wx.OK | wx.ICON_ERROR,
            )
            return

        if python_exe is None:
            wx.MessageBox(
                "Cannot find OrthoRoute's KiCad Python environment.",
                "OrthoRoute Error",
                wx.OK | wx.ICON_ERROR,
            )
            return

        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env["PYTHONUTF8"] = "1"
        env["PATH"] = str(python_exe.parent) + os.pathsep + env.get("PATH", "")

        site_packages = (
            python_exe.parent.parent / "Lib" / "site-packages"
        )
        qt_plugins = site_packages / "PyQt6" / "Qt6" / "plugins"
        if qt_plugins.is_dir():
            env["QT_PLUGIN_PATH"] = str(qt_plugins)

        try:
            subprocess.Popen(
                [str(python_exe), str(main_py), "plugin"],
                cwd=str(native_dir),
                env=env,
            )
        except Exception as error:
            wx.MessageBox(
                f"Failed to launch OrthoRoute:\n{error}",
                "OrthoRoute Error",
                wx.OK | wx.ICON_ERROR,
            )


OrthoRoutePlugin().register()
