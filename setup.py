#!/usr/bin/env python3
"""Setup script for OrthoRoute."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Advanced PCB Autorouter with Manhattan routing and GPU acceleration"

setup(
    name="orthoroute",
    version="1.0.0",
    author="OrthoRoute Team",
    author_email="team@orthoroute.com",
    description="Advanced PCB Autorouter with Manhattan routing and GPU acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/orthoroute/orthoroute",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "gui": [
            "PyQt6>=6.0.0",
        ],
        "gpu": [
            "cupy>=13",  # CUDA 12 era; matches requirements-kicad.txt's cupy-cuda12x>=13.4
        ],
        "metal": [
            (
                "mlx>=0.32.0; platform_system == 'Darwin' "
                "and platform_machine == 'arm64'"
            ),
        ],
        "kicad": [
            # KiCad Python API dependencies would go here
            # These are typically provided by KiCad installation
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "orthoroute=main:main",
            "orthoroute-gui=main:run_gui",
            "orthoroute-cli=main:run_cli",
        ],
    },
    include_package_data=True,
    package_data={
        "orthoroute": [
            "presentation/plugin/*.png",  # Plugin icons
            "shared/configuration/*.json",  # Default configs
        ],
    },
    zip_safe=False,
)
