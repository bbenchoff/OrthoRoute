"""
OrthoRoute Build System

Builds a KiCad SWIG ActionPlugin package that registers a toolbar button in
the PCB Editor.  When clicked the button launches main.py as a subprocess
using the IPC API (KICAD_API_SOCKET).

Usage:
    python build.py              # build package directory + ZIP
    python build.py --deploy     # build + install to local KiCad
    python build.py --zip        # build ZIP only (no deploy)
    python build.py --clean      # remove build directory

Outputs (in build/):
    com_github_bbenchoff_orthoroute/     Package directory (ready to copy)
    OrthoRoute-1.0.0.zip                 ZIP for manual install or distribution

ZIP installation:
    1. Open KiCad → Plugin and Content Manager
    2. "Install from File…" → select OrthoRoute-<version>.zip
    — OR —
    Extract ZIP into:
      Windows:  Documents/KiCad/9.0/3rdparty/plugins/
      macOS:    ~/Documents/KiCad/9.0/3rdparty/plugins/
      Linux:    ~/.local/share/KiCad/9.0/3rdparty/plugins/
"""

import os
import re
import sys
import json
import shutil
import stat
import logging
import platform
import zipfile
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Directory name must be a valid Python identifier (underscores, not dots)
# so KiCad's SWIG loader can import it.
PLUGIN_DIR_NAME = "com_github_bbenchoff_orthoroute"
PLUGIN_IDENTIFIER = "com.github.bbenchoff.orthoroute"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_version(project_root: Path) -> str:
    """Extract version from setup.py."""
    setup_file = project_root / "setup.py"
    if setup_file.exists():
        content = setup_file.read_text(encoding="utf-8")
        match = re.search(r'version\s*=\s*["\']([^"\'\n]+)["\']', content)
        if match:
            return match.group(1)
    return "1.0.0"


def _kicad_3rdparty_plugins_dir() -> Optional[Path]:
    """Return the 3rdparty/plugins directory KiCad actually scans.

    On Windows with OneDrive folder-redirection the shell Documents folder
    may differ from %USERPROFILE%\\Documents.  KiCad follows the shell
    folder, so we parse pcbnew.json to find the real path.
    """
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        pcbnew_json = Path(appdata) / "kicad" / "9.0" / "pcbnew.json"
        if pcbnew_json.exists():
            try:
                data = json.loads(pcbnew_json.read_text(encoding="utf-8"))
                for entry in data.get("action_plugins", []):
                    if isinstance(entry, dict):
                        for key in entry:
                            if "3rdparty" in key:
                                idx = key.find("3rdparty")
                                raw = key[:idx].replace("\\\\", "\\")
                                return Path(raw) / "3rdparty" / "plugins"
            except Exception:
                pass

    if platform.system() == "Windows":
        docs = Path(os.environ.get("USERPROFILE", "")) / "Documents"
        return docs / "KiCad" / "9.0" / "3rdparty" / "plugins"
    elif platform.system() == "Darwin":
        return (
            Path.home() / "Documents" / "KiCad" / "9.0" / "3rdparty" / "plugins"
        )
    else:
        return (
            Path.home()
            / ".local"
            / "share"
            / "KiCad"
            / "9.0"
            / "3rdparty"
            / "plugins"
        )


def _force_rmtree(path: Path):
    """Remove a directory tree, handling OneDrive locks and read-only files."""
    def _on_error(func, fpath, _exc_info):
        os.chmod(fpath, stat.S_IWRITE)
        func(fpath)

    try:
        shutil.rmtree(path, onerror=_on_error)
    except PermissionError:
        import time
        alt = path.with_name(f"{path.name}_old_{int(time.time())}")
        path.rename(alt)
        logger.warning(f"  Renamed locked dir → {alt.name}")


# ---------------------------------------------------------------------------
# Build system
# ---------------------------------------------------------------------------
class OrthoRouteBuildSystem:
    """Build system for OrthoRoute KiCad plugin."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent
        self.build_dir = self.project_root / "build"
        self.version = _read_version(self.project_root)
        self.package_dir = self.build_dir / PLUGIN_DIR_NAME
        self.zip_path = self.build_dir / f"OrthoRoute-{self.version}.zip"

    # -- clean --------------------------------------------------------------
    def clean(self):
        """Remove and recreate the package directory inside build/.

        Only removes the package subdirectory, not the entire build/ dir,
        to avoid OneDrive locking issues with the ZIP file.
        """
        logger.info("Cleaning build directory...")
        self.build_dir.mkdir(exist_ok=True)
        if self.package_dir.exists():
            _force_rmtree(self.package_dir)
        # Remove old ZIP if possible (OneDrive may lock it)
        if self.zip_path.exists():
            try:
                self.zip_path.unlink()
            except PermissionError:
                logger.warning(f"  Cannot remove locked {self.zip_path.name} — will overwrite")
        logger.info("[OK] Build directory ready")

    # -- copy sources -------------------------------------------------------
    def _copy_sources(self):
        """Assemble the plugin tree under build/<PLUGIN_DIR_NAME>/."""
        pkg = self.package_dir
        pkg.mkdir(parents=True, exist_ok=True)

        # orthoroute/ package
        src = self.project_root / "orthoroute"
        if src.exists():
            shutil.copytree(
                src,
                pkg / "orthoroute",
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", "*.pyo", "*.backup",
                ),
            )
            n = len(list((pkg / "orthoroute").rglob("*.py")))
            logger.info(f"  [OK] orthoroute/ ({n} .py files)")

        # main.py
        shutil.copy2(self.project_root / "main.py", pkg / "main.py")
        logger.info("  [OK] main.py")

        # __init__.py — SWIG ActionPlugin registration.
        # KiCad's LoadPlugins() imports subdirectories via __init__.py,
        # so the OrthoRoutePlugin().register() call MUST live here.
        swig_init = self.project_root / "swig_init.py"
        if swig_init.exists():
            shutil.copy2(swig_init, pkg / "__init__.py")
            logger.info("  [OK] __init__.py  (SWIG ActionPlugin)")
        else:
            logger.error("  [FAIL] swig_init.py not found!")

        # plugin.json — IPC API descriptor (for future KiCad IPC support)
        plugin_json = {
            "$schema": "https://go.kicad.org/api/schemas/v1",
            "identifier": PLUGIN_IDENTIFIER,
            "name": "OrthoRoute",
            "description": "GPU-accelerated PCB autorouter",
            "runtime": {"type": "python", "min_version": "3.10"},
            "actions": [
                {
                    "identifier": "orthoroute.route",
                    "name": "OrthoRoute",
                    "description": "Launch OrthoRoute GPU-accelerated autorouter",
                    "scopes": ["pcb"],
                    "entrypoint": "main.py",
                    "show-button": True,
                    "icons-light": ["icon-24.png", "icon-64.png"],
                }
            ],
        }
        (pkg / "plugin.json").write_text(
            json.dumps(plugin_json, indent=2), encoding="utf-8"
        )
        logger.info("  [OK] plugin.json")

        # metadata.json — required by KiCad PCM "Install from File…"
        metadata = {
            "$schema": "https://go.kicad.org/pcm/schemas/v1",
            "name": "OrthoRoute",
            "description": "GPU-accelerated PCB autorouter using PathFinder negotiated congestion on a Manhattan lattice.",
            "description_full": "OrthoRoute is a GPU-accelerated PCB autorouter for KiCad that uses the PathFinder negotiated congestion algorithm on a Manhattan lattice to route high-density boards.",
            "identifier": PLUGIN_IDENTIFIER,
            "type": "plugin",
            "author": {
                "name": "Brian Benchoff",
                "contact": {"github": "https://github.com/bbenchoff"},
            },
            "license": "MIT",
            "resources": {
                "homepage": "https://github.com/bbenchoff/OrthoRoute",
            },
            "versions": [
                {
                    "version": self.version,
                    "status": "stable",
                    "kicad_version": "9.0",
                }
            ],
        }
        (pkg / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        logger.info("  [OK] metadata.json")

        # Icons
        gfx = self.project_root / "graphics"
        for src_name, dst_name in [
            ("icon24.png", "icon-24.png"),
            ("icon64.png", "icon-64.png"),
        ]:
            icon_src = gfx / src_name
            if icon_src.exists():
                shutil.copy2(icon_src, pkg / dst_name)
                logger.info(f"  [OK] {dst_name}")

        # requirements.txt, LICENSE, orthoroute.json (runtime config)
        for fname in ("requirements.txt", "LICENSE", "orthoroute.json"):
            f = self.project_root / fname
            if f.exists():
                shutil.copy2(f, pkg / fname)
                logger.info(f"  [OK] {fname}")

    # -- validate -----------------------------------------------------------
    def _validate(self) -> bool:
        """Check that all required files exist in the assembled package."""
        required = [
            "__init__.py",
            "main.py",
            "plugin.json",
            "metadata.json",
            "icon-24.png",
            "orthoroute.json",
            "orthoroute/__init__.py",
        ]
        missing = [f for f in required if not (self.package_dir / f).exists()]
        if missing:
            logger.error(f"  [FAIL] Missing: {missing}")
            return False
        logger.info("[OK] Validation passed")
        return True

    # -- create ZIP ---------------------------------------------------------
    def _create_zip(self) -> Path:
        """Create a ZIP file suitable for KiCad 'Install from File…'.

        Structure inside the ZIP:
            metadata.json                          ← PCM requires this at root
            plugins/
              com_github_bbenchoff_orthoroute/
                __init__.py
                main.py
                orthoroute/
                ...
        """
        logger.info(f"\nCreating ZIP: {self.zip_path.name}")

        # On OneDrive the old ZIP may be locked; write to a temp name then
        # rename, or fall back to a timestamped name.
        import time
        tmp_zip = self.zip_path.with_suffix(f".tmp_{int(time.time())}.zip")
        try:
            zip_target = self.zip_path
            with zipfile.ZipFile(zip_target, "w", zipfile.ZIP_DEFLATED) as zf:
                pass  # test if writable
        except PermissionError:
            zip_target = tmp_zip
            logger.warning(f"  ZIP locked by OneDrive — writing to {zip_target.name}")

        with zipfile.ZipFile(zip_target, "w", zipfile.ZIP_DEFLATED) as zf:
            # metadata.json at ZIP root (required by PCM)
            meta_file = self.package_dir / "metadata.json"
            zf.write(meta_file, "metadata.json")

            # All package files under plugins/<PLUGIN_DIR_NAME>/
            prefix = f"plugins/{PLUGIN_DIR_NAME}"
            for file_path in sorted(self.package_dir.rglob("*")):
                if file_path.is_file():
                    # Skip __pycache__ and .pyc
                    if "__pycache__" in file_path.parts:
                        continue
                    rel = file_path.relative_to(self.package_dir)
                    arcname = f"{prefix}/{rel.as_posix()}"
                    zf.write(file_path, arcname)

            file_count = len(zf.namelist())

        # If we wrote to a temp file, try to rename; keep temp if rename fails
        if zip_target != self.zip_path:
            try:
                self.zip_path.unlink(missing_ok=True)
                zip_target.rename(self.zip_path)
            except PermissionError:
                self.zip_path = zip_target  # use temp name

        size_kb = self.zip_path.stat().st_size / 1024
        logger.info(f"[OK] {self.zip_path.name}  ({file_count} files, {size_kb:.0f} KB)")
        return self.zip_path

    # -- build (main entry) -------------------------------------------------
    def build(self) -> Path:
        """Build the package directory and ZIP."""
        logger.info(f"Building OrthoRoute {self.version}")
        logger.info(f"  Plugin dir name: {PLUGIN_DIR_NAME}\n")

        self.clean()
        self._copy_sources()
        ok = self._validate()
        if ok:
            self._create_zip()

        logger.info(f"\n{'=' * 60}")
        if ok:
            logger.info("BUILD COMPLETE")
            logger.info(f"  Directory: {self.package_dir}")
            logger.info(f"  ZIP:       {self.zip_path}")
        else:
            logger.error("BUILD FAILED")
        logger.info(f"{'=' * 60}")
        return self.package_dir

    # -- deploy -------------------------------------------------------------
    def deploy(self) -> bool:
        """Copy the built package into KiCad's 3rdparty/plugins directory."""
        plugins_dir = _kicad_3rdparty_plugins_dir()
        if plugins_dir is None:
            logger.error("[FAIL] Cannot determine KiCad 3rdparty plugins path")
            return False

        dest = plugins_dir / PLUGIN_DIR_NAME
        logger.info(f"\nDeploying to: {dest}")

        if dest.exists():
            _force_rmtree(dest)

        shutil.copytree(self.package_dir, dest)
        n = len(list(dest.rglob("*.py")))
        logger.info(f"[OK] Deployed ({n} .py files)")

        # Clean up old dot-name folder if present
        old_dot = plugins_dir / "com.github.bbenchoff.orthoroute"
        if old_dot.exists():
            try:
                _force_rmtree(old_dot)
                logger.info(f"  Removed old dir: {old_dot.name}")
            except Exception:
                pass

        logger.info("\n  >>> Restart KiCad to see the OrthoRoute toolbar button <<<")
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="OrthoRoute Build System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python build.py              Build package + ZIP
  python build.py --deploy     Build + install to local KiCad
  python build.py --zip        Build ZIP only (skip deploy)
  python build.py --clean      Remove build directory
""",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean build directory only"
    )
    parser.add_argument(
        "--deploy", action="store_true", help="Build and deploy to local KiCad"
    )
    parser.add_argument(
        "--zip", action="store_true", help="Build ZIP only (no deploy)"
    )
    args = parser.parse_args()

    builder = OrthoRouteBuildSystem()

    if args.clean:
        builder.clean()
        return 0

    builder.build()

    if args.deploy:
        if not builder.deploy():
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
