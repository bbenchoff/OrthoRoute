from pathlib import Path

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def _accelerator_requirements():
    return [
        Requirement(line)
        for line in (ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line and not line.startswith("#") and (
            line.startswith("cupy") or line.startswith("mlx")
        )
    ]


def test_cuda_and_metal_requirements_are_platform_separated():
    requirements = {
        requirement.name: requirement
        for requirement in _accelerator_requirements()
    }

    assert set(requirements) == {"cupy", "mlx"}
    assert str(requirements["cupy"].marker) == 'platform_system != "Darwin"'
    assert 'platform_system == "Darwin"' in str(
        requirements["mlx"].marker
    )
    assert 'platform_machine == "arm64"' in str(
        requirements["mlx"].marker
    )
