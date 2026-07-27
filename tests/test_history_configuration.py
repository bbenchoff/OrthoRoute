import pytest

from orthoroute.algorithms.manhattan.unified_pathfinder import (
    PathFinderConfig,
    resolve_history_decay,
)


def test_history_decay_respects_derived_config(monkeypatch):
    monkeypatch.delenv("ORTHO_HISTORY_DECAY", raising=False)
    config = PathFinderConfig()
    config.history_decay = 1.0

    assert resolve_history_decay(config) == 1.0


def test_history_decay_supports_valid_environment_override(monkeypatch):
    monkeypatch.setenv("ORTHO_HISTORY_DECAY", "0.995")

    assert resolve_history_decay(PathFinderConfig()) == pytest.approx(0.995)


@pytest.mark.parametrize("value", ["-0.1", "1.1"])
def test_history_decay_rejects_invalid_override(monkeypatch, value):
    monkeypatch.setenv("ORTHO_HISTORY_DECAY", value)

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        resolve_history_decay(PathFinderConfig())
