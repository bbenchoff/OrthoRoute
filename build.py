#!/usr/bin/env python3
"""Build and deploy the native KiCad IPC plugin package.

KiCad 10 discovers IPC plugins under ``<KiCad documents>/<version>/plugins``.
Each plugin is a directory containing ``plugin.json`` and an action entrypoint.
KiCad creates a dedicated virtual environment and installs ``requirements.txt``
before exposing the action in PCB Editor.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable


LOGGER = logging.getLogger("orthoroute.build")
PLUGIN_IDENTIFIER = "com.github.bbenchoff.orthoroute"
PLUGIN_NAME = "OrthoRoute"
PLUGIN_MANIFEST = "plugin.json"
PLUGIN_ENTRYPOINT = "kicad_plugin.py"
PLUGIN_REQUIREMENTS = "requirements-kicad.txt"
SWIG_PLUGIN_DIR_NAME = "com_github_bbenchoff_orthoroute"
PCM_SCHEMA = "https://go.kicad.org/pcm/schemas/v2"
PCM_MIN_KICAD_VERSION = "10.0"


def _read_version(project_root: Path) -> str:
    setup_text = (project_root / "setup.py").read_text(encoding="utf-8")
    match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', setup_text)
    if not match:
        raise RuntimeError("Could not determine version from setup.py")
    return match.group(1)


def _documents_directory() -> Path:
    """Return the OS documents directory, including Windows redirection."""
    if platform.system() == "Windows":
        try:
            import winreg

            key_path = (
                r"Software\Microsoft\Windows\CurrentVersion"
                r"\Explorer\User Shell Folders"
            )
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "Personal")
            return Path(os.path.expandvars(value)).expanduser()
        except (OSError, ImportError):
            return Path.home() / "Documents"
    if platform.system() == "Darwin":
        return Path.home() / "Documents"
    return Path.home() / ".local" / "share"


def _version_key(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"\d+(?:\.\d+){1,2}", value):
        return ()
    return tuple(int(part) for part in value.split("."))


def find_kicad_version(kicad_root: Path, requested: str | None = None) -> str:
    """Find the newest installed KiCad user-data version."""
    if requested:
        if not _version_key(requested):
            raise ValueError(f"Invalid KiCad version: {requested!r}")
        return requested

    candidates = [
        path.name
        for path in kicad_root.iterdir()
        if path.is_dir() and _version_key(path.name)
    ] if kicad_root.exists() else []
    if not candidates:
        raise RuntimeError(f"No KiCad user-data versions found under {kicad_root}")
    return max(candidates, key=_version_key)


def _assert_child(path: Path, parent: Path) -> None:
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    if resolved_path == resolved_parent or resolved_parent not in resolved_path.parents:
        raise RuntimeError(f"Refusing operation outside {resolved_parent}: {resolved_path}")


def _remove_tree(path: Path, parent: Path) -> None:
    if not path.exists():
        return
    _assert_child(path, parent)
    shutil.rmtree(path)


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".pytest_cache",
            "*.backup",
        ),
    )


class KiCadPluginBuilder:
    """Assemble, validate, archive, and deploy OrthoRoute."""

    def __init__(
        self,
        project_root: Path | None = None,
        output_root: Path | None = None,
    ) -> None:
        self.project_root = (project_root or Path(__file__).parent).resolve()
        self.output_root = (
            output_root or self.project_root / "build" / "kicad-plugin"
        ).resolve()
        self.version = _read_version(self.project_root)
        self.package_dir = self.output_root / PLUGIN_IDENTIFIER
        self.swig_package_dir = self.output_root / SWIG_PLUGIN_DIR_NAME
        self.pcm_package_dir = self.output_root / "pcm"
        self.zip_path = (
            self.project_root
            / "build"
            / f"OrthoRoute-{self.version}-KiCad-IPC.zip"
        )
        self.pcm_zip_path = (
            self.project_root
            / "build"
            / f"OrthoRoute-{self.version}-KiCad-PCM.zip"
        )

    @property
    def required_files(self) -> tuple[str, ...]:
        return (
            PLUGIN_MANIFEST,
            PLUGIN_ENTRYPOINT,
            "main.py",
            "requirements.txt",
            "orthoroute.json",
            "LICENSE",
            "graphics/icon24.png",
            "graphics/icon64.png",
            "orthoroute/__init__.py",
        )

    def clean(self) -> None:
        self.output_root.parent.mkdir(parents=True, exist_ok=True)
        _remove_tree(self.output_root, self.output_root.parent)
        if self.zip_path.exists():
            self.zip_path.unlink()
        if self.pcm_zip_path.exists():
            self.pcm_zip_path.unlink()

    def _copy_files(self) -> None:
        self.package_dir.mkdir(parents=True)

        for name in (
            PLUGIN_MANIFEST,
            PLUGIN_ENTRYPOINT,
            "main.py",
            "orthoroute.json",
            "LICENSE",
            "README.md",
        ):
            shutil.copy2(self.project_root / name, self.package_dir / name)

        shutil.copy2(
            self.project_root / PLUGIN_REQUIREMENTS,
            self.package_dir / "requirements.txt",
        )
        _copy_tree(
            self.project_root / "orthoroute",
            self.package_dir / "orthoroute",
        )

        graphics_dir = self.package_dir / "graphics"
        graphics_dir.mkdir()
        for name in ("icon24.png", "icon64.png"):
            shutil.copy2(
                self.project_root / "graphics" / name,
                graphics_dir / name,
            )

        install_text = f"""# OrthoRoute {self.version} for KiCad

This is a native KiCad IPC plugin.

1. Copy `{PLUGIN_IDENTIFIER}` into the KiCad `<version>/plugins` directory.
2. In KiCad, enable the API server under Preferences > Plugins.
3. Restart PCB Editor and wait for the plugin environment to finish installing.
4. Launch OrthoRoute from the PCB Editor toolbar.

KiCad installs the dependencies in `requirements.txt` into an isolated
environment. The first load can take several minutes because the GUI and GPU
runtime wheels are large.
"""
        (self.package_dir / "INSTALL.md").write_text(
            install_text,
            encoding="utf-8",
        )

        # PR #17's KiCad 9/10 compatibility bridge.  This intentionally has
        # no plugin.json: the native descriptor above remains the sole IPC
        # registration, while the SWIG loader supplies a reliable toolbar
        # button that launches it.
        self.swig_package_dir.mkdir(parents=True)
        shutil.copy2(
            self.project_root / "swig_init.py",
            self.swig_package_dir / "__init__.py",
        )
        shutil.copy2(
            self.project_root / "graphics" / "icon24.png",
            self.swig_package_dir / "icon-24.png",
        )

    def validate(self) -> dict:
        missing = [
            relative
            for relative in self.required_files
            if not (self.package_dir / relative).is_file()
        ]
        if missing:
            raise RuntimeError(f"Plugin package is missing required files: {missing}")

        manifest = json.loads(
            (self.package_dir / PLUGIN_MANIFEST).read_text(encoding="utf-8")
        )
        required_keys = {"identifier", "name", "description", "runtime", "actions"}
        missing_keys = required_keys - manifest.keys()
        if missing_keys:
            raise RuntimeError(f"plugin.json is missing fields: {sorted(missing_keys)}")
        if manifest["identifier"] != PLUGIN_IDENTIFIER:
            raise RuntimeError("plugin.json identifier does not match package directory")
        if manifest["runtime"].get("type") != "python":
            raise RuntimeError("OrthoRoute must use KiCad's native Python IPC runtime")
        if not manifest["actions"]:
            raise RuntimeError("plugin.json has no actions")

        for action in manifest["actions"]:
            entrypoint = action.get("entrypoint")
            if not entrypoint or not (self.package_dir / entrypoint).is_file():
                raise RuntimeError(f"Invalid action entrypoint: {entrypoint!r}")

        requirements = (
            self.package_dir / "requirements.txt"
        ).read_text(encoding="utf-8")
        forbidden = ("pytest", "flake8", "mypy", "sphinx", "\nblack")
        if any(name in requirements.lower() for name in forbidden):
            raise RuntimeError("Development dependency found in plugin requirements")

        for relative in ("__init__.py", "icon-24.png"):
            if not (self.swig_package_dir / relative).is_file():
                raise RuntimeError(f"SWIG bridge is missing required file: {relative}")
        if (self.swig_package_dir / PLUGIN_MANIFEST).exists():
            raise RuntimeError("SWIG bridge must not duplicate the native plugin.json")
        return manifest

    def create_zip(self) -> Path:
        self.zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            self.zip_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for source in sorted(self.package_dir.rglob("*")):
                if source.is_file():
                    relative = source.relative_to(self.package_dir)
                    archive.write(
                        source,
                        (Path(PLUGIN_IDENTIFIER) / relative).as_posix(),
                    )
        return self.zip_path

    def _pcm_metadata(self) -> dict:
        """Return metadata for KiCad 10's Plugin and Content Manager."""
        return {
            "$schema": PCM_SCHEMA,
            "name": PLUGIN_NAME,
            "description": (
                "GPU-accelerated PCB autorouter using PathFinder negotiated "
                "congestion on a Manhattan lattice."
            ),
            "description_full": (
                "OrthoRoute is an open-source PCB autorouter for KiCad. "
                "It connects through KiCad's IPC API, provides an interactive "
                "board viewer, enforces keepout rule areas, and can use CUDA "
                "or Apple MLX acceleration when available."
            ),
            "identifier": PLUGIN_IDENTIFIER,
            "type": "plugin",
            "author": {
                "name": "OrthoRoute contributors",
                "contact": {
                    "github": "https://github.com/bbenchoff/OrthoRoute",
                },
            },
            "maintainer": {
                "name": "Brian Benchoff",
                "contact": {
                    "github": "https://github.com/bbenchoff",
                },
            },
            "license": "MIT",
            "resources": {
                "homepage": "https://github.com/bbenchoff/OrthoRoute",
                "repository": "https://github.com/bbenchoff/OrthoRoute",
            },
            "tags": ["autorouter", "pcb", "routing"],
            "versions": [
                {
                    "version": self.version,
                    "status": "stable",
                    "kicad_version": PCM_MIN_KICAD_VERSION,
                    "platforms": ["windows", "macos", "linux"],
                    "runtime": "ipc",
                },
            ],
        }

    def create_pcm_zip(self) -> Path:
        """Create a KiCad 10 PCM archive suitable for Install from File."""
        plugins_dir = self.pcm_package_dir / "plugins"
        resources_dir = self.pcm_package_dir / "resources"
        self.pcm_package_dir.mkdir(parents=True)
        _copy_tree(self.package_dir, plugins_dir)
        resources_dir.mkdir()
        shutil.copy2(
            self.project_root / "graphics" / "icon64.png",
            resources_dir / "icon.png",
        )
        metadata = self._pcm_metadata()
        (self.pcm_package_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        self.validate_pcm_package()

        self.pcm_zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            self.pcm_zip_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for source in sorted(self.pcm_package_dir.rglob("*")):
                if source.is_file():
                    archive.write(
                        source,
                        source.relative_to(self.pcm_package_dir).as_posix(),
                    )
        return self.pcm_zip_path

    def validate_pcm_package(self) -> dict:
        metadata_path = self.pcm_package_dir / "metadata.json"
        plugin_root = self.pcm_package_dir / "plugins"
        required = (
            metadata_path,
            self.pcm_package_dir / "resources" / "icon.png",
            plugin_root / PLUGIN_MANIFEST,
            plugin_root / PLUGIN_ENTRYPOINT,
            plugin_root / "requirements.txt",
        )
        missing = [
            path.relative_to(self.pcm_package_dir).as_posix()
            for path in required
            if not path.is_file()
        ]
        if missing:
            raise RuntimeError(f"PCM package is missing required files: {missing}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        required_keys = {
            "name",
            "description",
            "description_full",
            "identifier",
            "type",
            "author",
            "license",
            "resources",
            "versions",
        }
        missing_keys = required_keys - metadata.keys()
        if missing_keys:
            raise RuntimeError(
                f"PCM metadata is missing fields: {sorted(missing_keys)}"
            )
        if metadata.get("$schema") != PCM_SCHEMA:
            raise RuntimeError("PCM metadata must target KiCad schema v2")
        if metadata["identifier"] != PLUGIN_IDENTIFIER:
            raise RuntimeError("PCM identifier does not match plugin identifier")
        if len(metadata["versions"]) != 1:
            raise RuntimeError("Archive metadata must contain exactly one version")
        version = metadata["versions"][0]
        if version.get("runtime") != "ipc":
            raise RuntimeError("PCM package must declare the IPC runtime")
        forbidden_download_fields = {
            "download_url",
            "download_sha256",
            "download_size",
            "install_size",
        }
        if forbidden_download_fields & version.keys():
            raise RuntimeError(
                "Archive metadata must not contain download_* or size fields"
            )
        return metadata

    def build(self, make_zip: bool = True, make_pcm_zip: bool = True) -> Path:
        LOGGER.info("Building %s %s native KiCad IPC plugin", PLUGIN_NAME, self.version)
        self.clean()
        self._copy_files()
        self.validate()
        if make_zip:
            self.create_zip()
        if make_pcm_zip:
            self.create_pcm_zip()
        LOGGER.info("Package: %s", self.package_dir)
        if make_zip:
            LOGGER.info("Manual-install ZIP: %s", self.zip_path)
        if make_pcm_zip:
            LOGGER.info("KiCad PCM ZIP: %s", self.pcm_zip_path)
        return self.package_dir

    def deploy(
        self,
        kicad_version: str | None = None,
        kicad_root: Path | None = None,
    ) -> Path:
        root = (kicad_root or _documents_directory() / "KiCad").resolve()
        version = find_kicad_version(root, kicad_version)
        plugins_dir = (root / version / "plugins").resolve()
        destination = plugins_dir / PLUGIN_IDENTIFIER
        swig_plugins_dir = (root / version / "3rdparty" / "plugins").resolve()
        swig_destination = swig_plugins_dir / SWIG_PLUGIN_DIR_NAME

        plugins_dir.mkdir(parents=True, exist_ok=True)
        swig_plugins_dir.mkdir(parents=True, exist_ok=True)
        _assert_child(destination, plugins_dir)
        _assert_child(swig_destination, swig_plugins_dir)
        _remove_tree(destination, plugins_dir)
        _remove_tree(swig_destination, swig_plugins_dir)
        shutil.copytree(self.package_dir, destination)
        shutil.copytree(self.swig_package_dir, swig_destination)
        LOGGER.info("Deployed to KiCad %s: %s", version, destination)
        LOGGER.info("Deployed PR #17 ActionPlugin bridge: %s", swig_destination)
        return destination


def _remove_legacy_build_artifacts(project_root: Path) -> Iterable[Path]:
    build_root = project_root / "build"
    legacy = (
        build_root / PLUGIN_IDENTIFIER,
        build_root / "pcm_package",
        build_root / f"{PLUGIN_IDENTIFIER}-pcm-1.0.0.zip",
    )
    for path in legacy:
        if path.is_dir():
            _remove_tree(path, build_root)
            yield path
        elif path.is_file():
            _assert_child(path, build_root)
            path.unlink()
            yield path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and deploy the OrthoRoute native KiCad IPC plugin",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Install the built plugin into the local KiCad plugins directory",
    )
    parser.add_argument(
        "--kicad-version",
        help="KiCad user-data version to deploy to (default: newest installed)",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Skip creation of the manual-install ZIP",
    )
    parser.add_argument(
        "--no-pcm-zip",
        action="store_true",
        help="Skip creation of the KiCad 10 PCM Install-from-File ZIP",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove generated plugin build artifacts and exit",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    builder = KiCadPluginBuilder()

    if args.clean:
        builder.clean()
        for path in _remove_legacy_build_artifacts(builder.project_root):
            LOGGER.info("Removed legacy build artifact: %s", path)
        return 0

    builder.build(
        make_zip=not args.no_zip,
        make_pcm_zip=not args.no_pcm_zip,
    )
    if args.deploy:
        builder.deploy(kicad_version=args.kicad_version)
        LOGGER.info(
            "Enable the KiCad API server and restart PCB Editor to load OrthoRoute."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
