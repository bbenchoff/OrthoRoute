import json
import zipfile
from pathlib import Path

import pytest

from build import (
    PLUGIN_ENTRYPOINT,
    PLUGIN_IDENTIFIER,
    SWIG_PLUGIN_DIR_NAME,
    KiCadPluginBuilder,
    find_kicad_version,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def built_plugin(tmp_path):
    builder = KiCadPluginBuilder(
        project_root=PROJECT_ROOT,
        output_root=tmp_path / "package",
    )
    builder.zip_path = tmp_path / "OrthoRoute.zip"
    builder.build()
    return builder


def test_manifest_declares_native_ipc_action(built_plugin):
    manifest = json.loads(
        (built_plugin.package_dir / "plugin.json").read_text(encoding="utf-8")
    )

    assert manifest["identifier"] == PLUGIN_IDENTIFIER
    assert manifest["runtime"]["type"] == "python"
    assert manifest["actions"][0]["entrypoint"] == PLUGIN_ENTRYPOINT
    assert manifest["actions"][0]["scopes"] == ["pcb"]


def test_package_contains_runtime_files_but_not_caches(built_plugin):
    for relative in built_plugin.required_files:
        assert (built_plugin.package_dir / relative).is_file()

    requirements = (
        built_plugin.package_dir / "requirements.txt"
    ).read_text(encoding="utf-8")
    assert "kicad-python" in requirements
    assert "PyQt6" in requirements
    assert "cupy-cuda12x" in requirements
    assert "pytest" not in requirements
    assert not list(built_plugin.package_dir.rglob("__pycache__"))
    assert not list(built_plugin.package_dir.rglob("*.pyc"))

    plugin_source = (
        built_plugin.package_dir
        / "orthoroute/presentation/plugin/kicad_plugin.py"
    ).read_text(encoding="utf-8")
    assert "pcbnew.ActionPlugin" not in plugin_source

    bridge_source = (
        built_plugin.swig_package_dir / "__init__.py"
    ).read_text(encoding="utf-8")
    assert "pcbnew.ActionPlugin" in bridge_source
    assert "OrthoRoutePlugin().register()" in bridge_source
    assert not (built_plugin.swig_package_dir / "plugin.json").exists()


def test_zip_extracts_as_one_plugin_directory(built_plugin):
    with zipfile.ZipFile(built_plugin.zip_path) as archive:
        names = archive.namelist()

    prefix = f"{PLUGIN_IDENTIFIER}/"
    assert names
    assert all(name.startswith(prefix) for name in names)
    assert f"{prefix}plugin.json" in names
    assert f"{prefix}{PLUGIN_ENTRYPOINT}" in names


def test_deploy_targets_requested_kicad_version(built_plugin, tmp_path):
    kicad_root = tmp_path / "KiCad"
    (kicad_root / "9.0").mkdir(parents=True)
    (kicad_root / "10.0").mkdir()

    destination = built_plugin.deploy(kicad_root=kicad_root)

    assert destination == kicad_root / "10.0" / "plugins" / PLUGIN_IDENTIFIER
    assert (destination / "plugin.json").is_file()
    bridge = (
        kicad_root
        / "10.0"
        / "3rdparty"
        / "plugins"
        / SWIG_PLUGIN_DIR_NAME
    )
    assert (bridge / "__init__.py").is_file()
    assert (bridge / "icon-24.png").is_file()


def test_find_kicad_version_rejects_invalid_version(tmp_path):
    with pytest.raises(ValueError):
        find_kicad_version(tmp_path, "../10.0")
