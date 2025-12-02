"""
OrthoRoute Build System - Creates manual installation package for KiCad IPC plugins
Follows the working example from layout_stamp (https://github.com/hraftery/layout_stamp)
"""

import os
import sys
import json
import shutil
import zipfile
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class OrthoRouteBuildSystem:
    """Build system for OrthoRoute manual installation package"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent
        self.build_dir = self.project_root / "build"
        self.version = "1.0.0"
        self.plugin_identifier = "com.github.bbenchoff.orthoroute"
        self.plugin_name = "OrthoRoute"

    def clean_build_directory(self):
        """Clean the build directory"""
        logger.info("Cleaning build directory...")
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        self.build_dir.mkdir(exist_ok=True)
        logger.info(f"[OK] Build directory cleaned: {self.build_dir}")

    def create_plugin_json(self) -> Dict:
        """Create plugin.json using modern schema v1 (strict compliance)"""
        return {
            "$schema": "https://go.kicad.org/api/schemas/v1",
            "identifier": self.plugin_identifier,
            "name": self.plugin_name,
            "description": "GPU-accelerated PCB autorouter with Manhattan routing and real-time visualization",
            "runtime": {
                "type": "python",
                "min_version": "3.10"
            },
            "actions": [
                {
                    "identifier": "orthoroute.route",
                    "name": "OrthoRoute",
                    "description": "Launch OrthoRoute GPU-accelerated autorouter",
                    "scopes": ["pcb"],
                    "entrypoint": "main.py",
                    "show-button": True,
                    "icons-light": ["icon-24.png", "icon-48.png"]
                }
            ]
        }

    def copy_core_files(self, package_dir: Path):
        """Copy core plugin files for manual installation"""
        logger.info(f"Copying core files to {package_dir.name}...")

        # Copy the orthoroute package directory
        orthoroute_src = self.project_root / "orthoroute"
        if orthoroute_src.exists():
            orthoroute_dst = package_dir / "orthoroute"
            shutil.copytree(orthoroute_src, orthoroute_dst)
            py_files = len(list(orthoroute_src.rglob('*.py')))
            logger.info(f"  [OK] Copied orthoroute package: {py_files} Python files")

        # Copy main.py entry point
        main_file = self.project_root / "main.py"
        if main_file.exists():
            shutil.copy2(main_file, package_dir / "main.py")
            logger.info(f"  [OK] Copied main.py entry point")

        # Create plugin.json
        plugin_json = self.create_plugin_json()
        with open(package_dir / "plugin.json", 'w', encoding='utf-8') as f:
            json.dump(plugin_json, f, indent=2)
        logger.info(f"  [OK] Created plugin.json (modern schema v1)")

        # Copy icons (rename to match plugin.json)
        graphics_src = self.project_root / "graphics"
        if graphics_src.exists():
            # Copy and rename icon24.png -> icon-24.png
            icon24_src = graphics_src / "icon24.png"
            if icon24_src.exists():
                shutil.copy2(icon24_src, package_dir / "icon-24.png")
                logger.info(f"  [OK] Copied icon-24.png")

            # Copy and rename icon64.png -> icon-48.png (or create if needed)
            icon48_src = graphics_src / "icon64.png"
            if icon48_src.exists():
                shutil.copy2(icon48_src, package_dir / "icon-48.png")
                logger.info(f"  [OK] Copied icon-48.png")

        # Copy requirements.txt
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            shutil.copy2(requirements_file, package_dir / "requirements.txt")
            logger.info("  [OK] Copied requirements.txt")

        # Copy LICENSE
        license_file = self.project_root / "LICENSE"
        if license_file.exists():
            shutil.copy2(license_file, package_dir / "LICENSE")
            logger.info("  [OK] Copied LICENSE")

    def create_installation_instructions(self, package_dir: Path):
        """Create detailed installation instructions"""
        instructions = f"""# OrthoRoute {self.version} - Manual Installation Instructions

## ⚠️ Important: IPC API Must Be Enabled

Before installing this plugin, you MUST enable the IPC API in KiCad:

1. Open KiCad
2. Go to Preferences → Plugins
3. Check the box "Enable Python API"
4. Click OK
5. Restart KiCad

## Installation Instructions

### Windows
1. Extract the `{self.plugin_identifier}` folder from this ZIP
2. Copy it to: `C:\\Users\\<your-username>\\Documents\\KiCad\\9.0\\plugins\\`
3. Restart KiCad
4. The OrthoRoute button should appear in the PCB Editor toolbar

### macOS
1. Extract the `{self.plugin_identifier}` folder from this ZIP
2. Copy it to: `/Users/<your-username>/Documents/KiCad/9.0/plugins/`
3. Restart KiCad
4. The OrthoRoute button should appear in the PCB Editor toolbar

### Linux
1. Extract the `{self.plugin_identifier}` folder from this ZIP
2. Copy it to: `~/.local/share/KiCad/9.0/plugins/`
3. Restart KiCad
4. The OrthoRoute button should appear in the PCB Editor toolbar

## Dependency Management

KiCad will automatically create a virtual environment and install dependencies
from requirements.txt when you first run the plugin.

The virtual environment is located at:
- Windows: `C:\\Users\\<username>\\AppData\\Local\\KiCad\\9.0\\python-environments\\{self.plugin_identifier}\\`
- macOS: `/Users/<username>/Library/Caches/KiCad/9.0/python-environments/{self.plugin_identifier}/`
- Linux: `~/.cache/KiCad/9.0/python-environments/{self.plugin_identifier}/`

## Requirements

- KiCad 9.0 or later
- Python 3.10 or later (usually included with KiCad)
- Dependencies listed in requirements.txt (auto-installed by KiCad)

## Troubleshooting

### Plugin button doesn't appear
- Verify IPC API is enabled (Preferences → Plugins)
- Check that the folder name matches: `{self.plugin_identifier}`
- Restart KiCad after installation
- Check KiCad logs for errors: `Documents/KiCad/9.0/logs/`

### Dependencies not installing
- Check your internet connection
- Look at: `Documents/KiCad/9.0/logs/api.log` for installation errors
- Manually activate the venv and install: `pip install -r requirements.txt`

### Can't find plugins directory
- Create the directory if it doesn't exist
- The path varies by KiCad version (use 9.0, 9.1, etc.)

## More Information

- Project: https://github.com/bbenchoff/OrthoRoute
- KiCad IPC API Docs: https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/
- Plugin Reference: https://github.com/hraftery/layout_stamp

## Why Manual Installation?

PCM (Plugin and Content Manager) support for IPC plugins is currently broken
on Windows (GitLab issue #19465). Manual installation is the only reliable
method for now. We'll add PCM support once KiCad fixes the issue.

---
Generated: {datetime.now().strftime('%Y-%m-%d')}
Version: {self.version}
"""
        with open(package_dir / "INSTALL.txt", 'w', encoding='utf-8') as f:
            f.write(instructions)
        logger.info("  [OK] Created INSTALL.txt")

    def create_package_zip(self, package_dir: Path) -> Path:
        """Create ZIP package for manual installation"""
        zip_name = f"{self.plugin_identifier}-{self.version}.zip"
        zip_path = self.build_dir / zip_name

        logger.info(f"\nCreating installation package: {zip_name}")

        # Create ZIP with the plugin folder inside
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add INSTALL.txt at root of ZIP
            install_file = package_dir / "INSTALL.txt"
            if install_file.exists():
                zipf.write(install_file, "INSTALL.txt")

            # Add all plugin files under the plugin identifier folder
            for file_path in package_dir.rglob('*'):
                if file_path.is_file() and file_path.name != "INSTALL.txt":
                    # Create archive path as: com.github.bbenchoff.orthoroute/...
                    arcname = self.plugin_identifier + "/" + str(file_path.relative_to(package_dir))
                    zipf.write(file_path, arcname)

        # Calculate size
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        logger.info(f"[OK] Package created: {zip_name} ({size_mb:.2f} MB)")

        return zip_path

    def build_package(self) -> Optional[Path]:
        """Build the manual installation package"""
        logger.info(f"Building OrthoRoute {self.version} manual installation package")
        logger.info(f"Project root: {self.project_root}\n")

        # Create package directory
        package_dir = self.build_dir / self.plugin_identifier
        package_dir.mkdir(parents=True, exist_ok=True)

        # Copy files
        self.copy_core_files(package_dir)

        # Create installation instructions
        self.create_installation_instructions(package_dir)

        # Create ZIP package
        zip_path = self.create_package_zip(package_dir)

        return zip_path

    def show_completion_message(self, zip_path: Path):
        """Show completion message with installation instructions"""
        size_mb = zip_path.stat().st_size / (1024 * 1024)

        print("\n" + "="*70)
        print("BUILD COMPLETE!")
        print("="*70)
        print(f"\nPackage: {zip_path.name} ({size_mb:.2f} MB)")
        print(f"Location: {zip_path.parent}")
        print("\n" + "="*70)
        print("INSTALLATION INSTRUCTIONS")
        print("="*70)
        print("\n1. First, enable IPC API in KiCad:")
        print("   - Open KiCad -> Preferences -> Plugins")
        print("   - Check 'Enable Python API'")
        print("   - Restart KiCad")
        print("\n2. Install the plugin:")
        print(f"   - Extract the ZIP file")
        print(f"   - Copy the '{self.plugin_identifier}' folder to your plugins directory:")
        print(f"     * Windows: C:\\Users\\<username>\\Documents\\KiCad\\9.0\\plugins\\")
        print(f"     * macOS: /Users/<username>/Documents/KiCad/9.0/plugins/")
        print(f"     * Linux: ~/.local/share/KiCad/9.0/plugins/")
        print("\n3. Restart KiCad")
        print("\n4. Look for the OrthoRoute button in PCB Editor toolbar")
        print("\n" + "="*70)
        print("Full instructions included in INSTALL.txt inside the ZIP")
        print("="*70 + "\n")

    def create_pcm_metadata(self) -> Dict:
        """Create metadata.json for PCM installation (SWIG plugin)"""
        return {
            "$schema": "https://go.kicad.org/pcm/schemas/v1",
            "name": self.plugin_name,
            "description": "GPU-accelerated PCB autorouter with Manhattan routing",
            "description_full": "Advanced PCB autorouter that leverages GPU acceleration for high-performance routing. Features intelligent pathfinding, real-time visualization, and seamless KiCad integration.",
            "identifier": self.plugin_identifier,
            "type": "plugin",
            "author": {
                "name": "OrthoRoute Team",
                "contact": {
                    "web": "https://github.com/bbenchoff/OrthoRoute"
                }
            },
            "maintainer": {
                "name": "OrthoRoute Team",
                "contact": {
                    "web": "https://github.com/bbenchoff/OrthoRoute"
                }
            },
            "license": "MIT",
            "resources": {
                "homepage": "https://github.com/bbenchoff/OrthoRoute",
                "repository": "https://github.com/bbenchoff/OrthoRoute",
                "issues": "https://github.com/bbenchoff/OrthoRoute/issues",
                "icon": "resources/icon.png"
            },
            "tags": [
                "autorouter",
                "routing",
                "pcb",
                "gpu",
                "automation"
            ],
            "versions": [
                {
                    "version": self.version,
                    "status": "stable",
                    "kicad_version": "9.0",
                    "platforms": ["windows", "macos", "linux"],
                    "runtime": "swig"
                }
            ]
        }

    def create_swig_init_py(self) -> str:
        """Create __init__.py for SWIG ActionPlugin registration"""
        return '''"""OrthoRoute KiCad SWIG Plugin"""
import os
import sys

# Add plugin directory to path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

try:
    import pcbnew

    class OrthoRoutePlugin(pcbnew.ActionPlugin):
        """KiCad Action Plugin wrapper for OrthoRoute."""

        def defaults(self):
            """Set plugin defaults."""
            self.name = "OrthoRoute"
            self.category = "Routing"
            self.description = "GPU-accelerated PCB autorouter"
            self.show_toolbar_button = True
            self.icon_file_name = os.path.join(os.path.dirname(__file__), "icon.png")

        def Run(self):
            """Run the plugin."""
            try:
                # Import the main plugin class
                from orthoroute.presentation.plugin.kicad_plugin import KiCadPlugin

                # Create and run plugin
                plugin = KiCadPlugin()
                result = plugin.run_with_gui()

                if result:
                    pcbnew.Refresh()

            except Exception as e:
                import traceback
                import wx
                error_msg = f"OrthoRoute error: {e}\n\n{traceback.format_exc()}"
                wx.LogError(error_msg)  # KiCad 9 uses wx.LogError, not pcbnew.LogError
                print(error_msg)  # Also print to console

    # Register the plugin
    OrthoRoutePlugin().register()

except ImportError:
    # pcbnew not available - running outside KiCad
    pass
'''

    def build_pcm_package(self) -> Optional[Path]:
        """Build PCM-installable SWIG plugin package"""
        logger.info(f"Building OrthoRoute {self.version} PCM package (SWIG)")
        logger.info(f"Project root: {self.project_root}\n")

        # Create package directory structure
        package_dir = self.build_dir / "pcm_package"
        package_dir.mkdir(parents=True, exist_ok=True)

        # NOTE: Files must be at root level, NOT in plugins/ subdirectory
        resources_dir = package_dir / "resources"
        resources_dir.mkdir(exist_ok=True)

        logger.info("Copying files for PCM package...")

        # Copy orthoroute package to root level
        orthoroute_src = self.project_root / "orthoroute"
        if orthoroute_src.exists():
            orthoroute_dst = package_dir / "orthoroute"
            shutil.copytree(orthoroute_src, orthoroute_dst)
            py_files = len(list(orthoroute_src.rglob('*.py')))
            logger.info(f"  [OK] Copied orthoroute package: {py_files} Python files")

        # Copy main.py to root level
        main_file = self.project_root / "main.py"
        if main_file.exists():
            shutil.copy2(main_file, package_dir / "main.py")
            logger.info(f"  [OK] Copied main.py")

        # Create __init__.py for SWIG registration at root level
        init_content = self.create_swig_init_py()
        with open(package_dir / "__init__.py", 'w', encoding='utf-8') as f:
            f.write(init_content)
        logger.info(f"  [OK] Created __init__.py (SWIG ActionPlugin)")

        # Copy 24x24 icon to root level
        graphics_src = self.project_root / "graphics"
        if graphics_src.exists():
            icon24_src = graphics_src / "icon24.png"
            if icon24_src.exists():
                shutil.copy2(icon24_src, package_dir / "icon.png")
                logger.info(f"  [OK] Copied toolbar icon (24x24)")

            # Copy 64x64 icon to resources/
            icon64_src = graphics_src / "icon64.png"
            if icon64_src.exists():
                shutil.copy2(icon64_src, resources_dir / "icon.png")
                logger.info(f"  [OK] Copied catalog icon (64x64)")

        # Create metadata.json at package root
        metadata = self.create_pcm_metadata()
        with open(package_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  [OK] Created metadata.json (PCM)")

        # Create ZIP package
        zip_name = f"{self.plugin_identifier}-pcm-{self.version}.zip"
        zip_path = self.build_dir / zip_name

        logger.info(f"\nCreating PCM package: {zip_name}")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in package_dir.rglob('*'):
                if file_path.is_file():
                    arcname = str(file_path.relative_to(package_dir))
                    zipf.write(file_path, arcname)

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        logger.info(f"[OK] PCM package created: {zip_name} ({size_mb:.2f} MB)")

        return zip_path

    def show_pcm_completion_message(self, zip_path: Path):
        """Show completion message for PCM package"""
        size_mb = zip_path.stat().st_size / (1024 * 1024)

        print("\n" + "="*70)
        print("BUILD COMPLETE - PCM PACKAGE")
        print("="*70)
        print(f"\nPackage: {zip_path.name} ({size_mb:.2f} MB)")
        print(f"Location: {zip_path.parent}")
        print("\n" + "="*70)
        print("INSTALLATION INSTRUCTIONS")
        print("="*70)
        print("\n1. Open KiCad")
        print("2. Go to Tools -> Plugin and Content Manager")
        print("3. Click 'Install from File'")
        print(f"4. Select: {zip_path}")
        print("5. Restart KiCad")
        print("6. Look for OrthoRoute button in PCB Editor toolbar")
        print("\n" + "="*70)
        print("NOTE: This is a SWIG plugin using KiCad's embedded Python")
        print("Some features may differ from the IPC version")
        print("="*70 + "\n")

def main():
    """Main build script entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='OrthoRoute Build System')
    parser.add_argument('--version', default='1.0.0', help='Version number')
    parser.add_argument('--clean', action='store_true', help='Clean build directory only')
    parser.add_argument('--pcm', action='store_true', help='Build PCM package (SWIG) instead of manual IPC package')

    args = parser.parse_args()

    builder = OrthoRouteBuildSystem()
    builder.version = args.version

    if args.clean:
        builder.clean_build_directory()
        return 0

    # Clean and build
    builder.clean_build_directory()

    if args.pcm:
        # Build PCM-installable SWIG package
        zip_path = builder.build_pcm_package()
        if zip_path and zip_path.exists():
            builder.show_pcm_completion_message(zip_path)
            return 0
    else:
        # Build manual installation IPC package
        zip_path = builder.build_package()
        if zip_path and zip_path.exists():
            builder.show_completion_message(zip_path)
            return 0

    logger.error("[ERROR] Build failed!")
    return 1

if __name__ == "__main__":
    sys.exit(main())
