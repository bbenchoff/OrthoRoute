"""OrthoRoute KiCad SWIG Plugin — Toolbar button registration.

Follows the same pattern as InteractiveHtmlBom: registers a pcbnew.ActionPlugin
so that a toolbar button appears in the PCB Editor. When clicked, launches the
OrthoRoute IPC plugin (main.py) as a subprocess.
"""
import os
import sys
import subprocess

import pcbnew


class OrthoRoutePlugin(pcbnew.ActionPlugin):
    """KiCad Action Plugin wrapper for OrthoRoute."""

    def defaults(self):
        self.name = "OrthoRoute"
        self.category = "Routing"
        self.description = "GPU-accelerated PCB autorouter"
        self.show_toolbar_button = True
        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        self.icon_file_name = os.path.join(self.plugin_dir, "icon-24.png")

    # -- Platform helpers ----------------------------------------------------

    @staticmethod
    def _kicad_bin_dir():
        """Return the KiCad bin/ directory for the current platform."""
        if sys.platform == "win32":
            return os.path.join(
                os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                "KiCad", "9.0", "bin",
            )
        elif sys.platform == "darwin":
            return "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin"
        else:  # Linux
            return "/usr/bin"

    @staticmethod
    def _thirdparty_root(plugin_dir):
        """Return the 3rdparty root (two levels above the plugin dir).

        Layout:  .../3rdparty/plugins/<plugin_name>/
        So parent of parent of plugin_dir is the 3rdparty root.
        """
        return os.path.normpath(os.path.join(plugin_dir, os.pardir, os.pardir))

    @staticmethod
    def _thirdparty_site_packages(thirdparty_root):
        """Find the 3rdparty site-packages directory (platform-dependent).

        Windows:  3rdparty/Python311/site-packages/
        Linux:    3rdparty/lib/python3/dist-packages/  (Debian/Ubuntu)
                  — or —  system site-packages (no 3rdparty Python on Linux)
        macOS:    3rdparty/lib/python/site-packages/
        """
        candidates = [
            os.path.join(thirdparty_root, "Python311", "site-packages"),    # Windows
            os.path.join(thirdparty_root, "lib", "python3", "dist-packages"),  # Linux
            os.path.join(thirdparty_root, "lib", "python", "site-packages"),   # macOS
        ]
        for path in candidates:
            if os.path.isdir(path):
                return path
        return None

    # -- Run ----------------------------------------------------------------

    def Run(self):
        """Launch OrthoRoute as a subprocess in plugin mode.

        Always uses plugin mode with RichKiCadInterface which auto-discovers
        the IPC socket via kipy.  KICAD_API_SOCKET env var is NOT required.
        """
        import wx

        main_py = os.path.join(self.plugin_dir, "main.py")
        if not os.path.exists(main_py):
            wx.MessageBox(
                f"Cannot find main.py at:\n{main_py}",
                "OrthoRoute Error",
                wx.OK | wx.ICON_ERROR,
            )
            return

        env = os.environ.copy()
        kicad_bin = self._kicad_bin_dir()
        thirdparty_root = self._thirdparty_root(self.plugin_dir)

        if sys.platform == "win32":
            # Windows: PYTHONHOME is required for KiCad's bundled Python
            # to find its stdlib.  Replicates kicad-cmd.bat environment.
            env["PYTHONHOME"] = kicad_bin
            env["PYTHONUTF8"] = "1"

            # PATH: KiCad bin + 3rdparty Scripts first
            thirdparty_scripts = os.path.join(thirdparty_root, "Python311", "Scripts")
            kicad_scripts = os.path.join(kicad_bin, "Scripts")
            path_prefix = os.pathsep.join(
                p for p in [kicad_bin, thirdparty_scripts, kicad_scripts]
                if os.path.isdir(p)
            )
            env["PATH"] = path_prefix + os.pathsep + env.get("PATH", "")
        else:
            # Linux/macOS: KiCad Python is usually on PATH already.
            # Just ensure kicad_bin is present.
            path = env.get("PATH", "")
            if kicad_bin not in path:
                env["PATH"] = kicad_bin + os.pathsep + path

        # PYTHONPATH: 3rdparty site-packages (kipy, PyQt6, etc.)
        thirdparty_sp = self._thirdparty_site_packages(thirdparty_root)
        if thirdparty_sp:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                thirdparty_sp + os.pathsep + existing if existing else thirdparty_sp
            )

            # PyQt6 needs QT_PLUGIN_PATH to find platform plugins
            qt_plugins = os.path.join(thirdparty_sp, "PyQt6", "Qt6", "plugins")
            if os.path.isdir(qt_plugins):
                env["QT_PLUGIN_PATH"] = qt_plugins

        python_exe = self._python_exe()
        if not python_exe:
            wx.MessageBox(
                "Cannot find KiCad's Python interpreter.\n\n"
                "Searched in:\n"
                f"  {kicad_bin}",
                "OrthoRoute Error",
                wx.OK | wx.ICON_ERROR,
            )
            return

        # Always use plugin mode — RichKiCadInterface auto-discovers the
        # IPC socket via kipy, no KICAD_API_SOCKET env var needed.
        cmd = [python_exe, main_py, "plugin"]

        try:
            # Use CREATE_NEW_CONSOLE on Windows so the user can see output
            kwargs = {"cwd": self.plugin_dir, "env": env}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
            subprocess.Popen(cmd, **kwargs)
        except Exception as e:
            wx.MessageBox(
                f"Failed to launch OrthoRoute:\n{e}",
                "OrthoRoute Error",
                wx.OK | wx.ICON_ERROR,
            )

    def _python_exe(self):
        """Return KiCad's bundled Python interpreter.

        Inside KiCad's SWIG process sys.executable points to kicad.exe
        (or the kicad binary on Linux/macOS), so we locate the known
        KiCad Python path per-platform instead.
        """
        kicad_bin = self._kicad_bin_dir()

        if sys.platform == "win32":
            candidates = [
                os.path.join(kicad_bin, "python.exe"),
                os.path.join(
                    os.environ.get("LOCALAPPDATA", ""),
                    "KiCad", "9.0", "python-environments",
                    "com.github.bbenchoff.orthoroute",
                    "Scripts", "python.exe",
                ),
            ]
        elif sys.platform == "darwin":
            candidates = [
                os.path.join(kicad_bin, "python3"),
                "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3",
            ]
        else:  # Linux
            candidates = [
                os.path.join(kicad_bin, "python3"),
                "/usr/bin/python3",
            ]

        for path in candidates:
            if os.path.exists(path):
                return path

        return None  # will be caught in Run()


# register plugin with kicad backend
OrthoRoutePlugin().register()
