#!/usr/bin/env python3
"""
OrthoRoute Build System - Unified builder for all package formats
Creates KiCad plugin packages in multiple formats for distribution
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrthoRouteBuildSystem:
    """Unified build system for OrthoRoute plugin packages"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent
        self.build_dir = self.project_root / "build"
        self.version = "1.0.0"
        
        # Single package configuration
        self.packages = {
            'default': {
                'name': 'orthoroute',
                'description': 'OrthoRoute - GPU-accelerated PCB autorouting plugin',
                'include_gpu': True,
                'include_docs': False,
                'include_tests': False
            }
        }
    
    def clean_build_directory(self):
        """Clean the build directory"""
        logger.info("🧹 Cleaning build directory...")
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        self.build_dir.mkdir(exist_ok=True)
        logger.info(f"✓ Build directory cleaned: {self.build_dir}")
    
    def create_plugin_metadata(self, package_config: Dict) -> Dict:
        """Create plugin metadata for KiCad"""
        return {
            "$schema": "https://go.kicad.org/pcm/schemas/v1",
            "name": package_config['name'],
            "description": package_config['description'],
            "description_full": f"{package_config['description']} - GPU-accelerated PCB autorouting with real-time visualization",
            "identifier": package_config['name'].replace('-', '_'),
            "type": "plugin",
            "author": {
                "name": "OrthoRoute Team",
                "contact": {
                    "github": "https://github.com/bbenchoff/OrthoRoute"
                }
            },
            "maintainer": {
                "name": "OrthoRoute Team",
                "contact": {
                    "github": "https://github.com/bbenchoff/OrthoRoute"
                }
            },
            "license": "MIT",
            "resources": {
                "homepage": "https://github.com/bbenchoff/OrthoRoute",
                "icon": "resources/icon64.png"
            },
            "tags": [
                "pcb",
                "routing",
                "autorouting",
                "gpu",
                "automation",
                "visualization"
            ],
            "keep_on_update": [],
            "versions": [
                {
                    "version": self.version,
                    "status": "stable",
                    "kicad_version": "9.0",
                    "platforms": ["windows", "macos", "linux"]
                }
            ],
            "runtime": "ipc"
        }
    
    def copy_core_files(self, package_dir: Path, package_config: Dict):
        """Copy core plugin files"""
        logger.info(f"📂 Copying core files for {package_config['name']}...")

        # For PCM IPC plugins, everything goes in the "plugins" subdirectory
        # which will be extracted to 3rdparty/plugins/orthoroute/
        plugins_dir = package_dir / "plugins"
        plugins_dir.mkdir(exist_ok=True)

        # Copy the orthoroute package directory directly into plugins/
        orthoroute_dir = self.project_root / "orthoroute"
        if orthoroute_dir.exists():
            package_orthoroute = plugins_dir / "orthoroute"
            shutil.copytree(orthoroute_dir, package_orthoroute)
            py_files = len(list(orthoroute_dir.rglob('*.py')))
            logger.info(f"✓ Copied orthoroute package: {py_files} Python files")

        # Copy main.py entry point directly to plugins/ (not nested)
        main_file = self.project_root / "main.py"
        if main_file.exists():
            shutil.copy2(main_file, plugins_dir / "main.py")
            logger.info(f"✓ Copied main.py entry point")

        # Create plugin.json directly in plugins/ (not nested)
        # Icon path is relative from here
        plugin_json = {
            "identifier": "com.orthoroute.autorouter",
            "name": "OrthoRoute",
            "version": self.version,
            "description": "GPU-accelerated PCB autorouter with Manhattan routing",
            "runtime": "python",
            "actions": [
                {
                    "id": "orthoroute.route",
                    "label": "OrthoRoute",
                    "icon": "resources/icon24.png",  # Changed: relative from plugins/
                    "entry_point": "main.py"
                }
            ]
        }

        with open(plugins_dir / "plugin.json", 'w', encoding='utf-8') as f:
            json.dump(plugin_json, f, indent=2)
        logger.info(f"✓ Created plugin.json for IPC registration")

        # Copy resources directly to plugins/resources/ (sibling to plugin.json)
        resources_src = self.project_root / "graphics"
        if resources_src.exists():
            resources_dir = plugins_dir / "resources"
            resources_dir.mkdir(exist_ok=True)
            for icon_file in ["icon24.png", "icon64.png", "icon200.png"]:
                src = resources_src / icon_file
                if src.exists():
                    shutil.copy2(src, resources_dir / icon_file)
            logger.info(f"✓ Copied icons to plugins/resources/")
        
        # Requirements
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            shutil.copy2(requirements_file, package_dir)
            logger.info("✓ Copied requirements.txt")
    
    def copy_optional_files(self, package_dir: Path, package_config: Dict):
        """Copy optional files based on package configuration"""
        
        # Documentation
        if package_config.get('include_docs', False):
            docs_dir = self.project_root / "docs"
            if docs_dir.exists():
                package_docs = package_dir / "docs"
                shutil.copytree(docs_dir, package_docs)
                logger.info(f"✓ Copied documentation: {len(list(docs_dir.glob('*.md')))} files")
            
            # README
            readme_file = self.project_root / "README.md"
            if readme_file.exists():
                shutil.copy2(readme_file, package_dir)
                logger.info("✓ Copied README.md")
        
        # Tests
        if package_config.get('include_tests', False):
            tests_dir = self.project_root / "tests"
            if tests_dir.exists():
                package_tests = package_dir / "tests"
                shutil.copytree(tests_dir, package_tests)
                logger.info(f"✓ Copied tests: {len(list(tests_dir.glob('*.py')))} files")
        
        # GPU acceleration files
        if package_config.get('include_gpu', True):
            # GPU files are already in src/, just log
            logger.info("✓ GPU acceleration included")
        else:
            # Remove GPU-specific files for lite version
            gpu_file = package_dir / "src" / "gpu_routing_engine.py"
            if gpu_file.exists():
                gpu_file.unlink()
                logger.info("✓ Removed GPU files for lite version")
    
    def create_package_zip(self, package_dir: Path, package_config: Dict) -> Path:
        """Create ZIP package for PCM installation"""
        zip_name = f"{package_config['name']}-{self.version}.zip"
        zip_path = self.build_dir / zip_name

        logger.info(f"📦 Creating PCM plugin ZIP package: {zip_name}")

        # For PCM packages, files must be at the root of the ZIP (no subdirectory)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in package_dir.rglob('*'):
                if file_path.is_file():
                    # Create archive path relative to package_dir (files at root)
                    arcname = str(file_path.relative_to(package_dir))
                    zipf.write(file_path, arcname)

        # Calculate size
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ PCM plugin package created: {zip_name} ({size_mb:.2f} MB)")

        return zip_path
    
    def build_package(self, package_type: str) -> Optional[Path]:
        """Build a specific package type"""
        if package_type not in self.packages:
            logger.error(f"Unknown package type: {package_type}")
            return None
        
        package_config = self.packages[package_type]
        logger.info(f"🏗️ Building {package_type} package: {package_config['name']}")
        
        # Create package directory
        package_dir = self.build_dir / package_config['name']
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        self.copy_core_files(package_dir, package_config)
        self.copy_optional_files(package_dir, package_config)

        # Create metadata.json for PCM installation
        metadata = self.create_plugin_metadata(package_config)
        metadata_file = package_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info("✓ Created metadata.json for PCM")

        # Create README for installation instructions
        readme_content = f"""# OrthoRoute {self.version} - IPC Plugin

## Installation via Plugin Manager (Recommended)

1. Open KiCad
2. Go to Plugin and Content Manager
3. Click "Install from File..."
4. Select orthoroute-{self.version}.zip
5. Restart KiCad
6. The OrthoRoute button should appear in the PCB Editor toolbar

## Manual Installation

1. Extract this ZIP file
2. Copy the extracted folder to your KiCad plugins directory
3. Restart KiCad

## Requirements

- KiCad 9.0 or later with IPC API enabled
- Python 3.8 or later
- See requirements.txt for Python dependencies

For more information, visit: https://github.com/bbenchoff/OrthoRoute
"""
        with open(package_dir / "README.txt", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        logger.info("✓ Created README.txt")

        # Create ZIP package
        zip_path = self.create_package_zip(package_dir, package_config)
        
        logger.info(f"✅ {package_type.title()} package complete: {zip_path}")
        return zip_path
    
    def build_all_packages(self) -> List[Path]:
        """Build the default package"""
        logger.info("🚀 Starting OrthoRoute build process...")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Version: {self.version}")

        self.clean_build_directory()

        # Build only the default package
        try:
            zip_path = self.build_package('default')
            if zip_path:
                size_mb = zip_path.stat().st_size / (1024 * 1024)
                logger.info("="*60)
                logger.info("🎉 BUILD COMPLETE!")
                logger.info(f"  📦 {zip_path.name} ({size_mb:.2f} MB)")
                logger.info(f"Build directory: {self.build_dir}")
                logger.info("="*60)
                return [zip_path]
        except Exception as e:
            logger.error(f"Failed to build package: {e}")
            return []

        return []

def main():
    """Main build script entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OrthoRoute Build System')
    parser.add_argument('--version', default='1.0.0', help='Version number')
    parser.add_argument('--clean', action='store_true', help='Clean build directory only')

    args = parser.parse_args()

    builder = OrthoRouteBuildSystem()
    builder.version = args.version

    if args.clean:
        builder.clean_build_directory()
        return 0

    builder.build_all_packages()

    return 0

if __name__ == "__main__":
    sys.exit(main())
