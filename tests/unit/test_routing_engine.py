"""
Unit tests for the UnifiedPathFinder config and basic initialisation.

These tests do NOT run the full routing algorithm (that belongs in regression/).
They verify the public API surface used by the pipeline.
"""
import pytest


class TestPathFinderConfig:
    def test_default_config_instantiation(self):
        from orthoroute.algorithms.manhattan.unified_pathfinder import PathFinderConfig
        cfg = PathFinderConfig()
        assert cfg is not None

    def test_config_has_hotset_cap(self):
        from orthoroute.algorithms.manhattan.unified_pathfinder import PathFinderConfig
        cfg = PathFinderConfig()
        # hotset_cap must be a positive integer
        assert hasattr(cfg, "hotset_cap"), "PathFinderConfig missing hotset_cap"
        assert cfg.hotset_cap > 0


def _gpu_available() -> bool:
    try:
        import cupy as cp
        cp.cuda.Device(0).use()
        cp.array([1]).sum()  # force a real compute
        return True
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _gpu_available(), reason="No CUDA GPU available")


class TestUnifiedPathFinderInit:
    def test_instantiation_cpu_mode(self):
        from orthoroute.algorithms.manhattan.unified_pathfinder import (
            UnifiedPathFinder,
            PathFinderConfig,
        )
        router = UnifiedPathFinder(config=PathFinderConfig(), use_gpu=False)
        assert router is not None

    @requires_gpu
    def test_instantiation_gpu_mode(self):
        from orthoroute.algorithms.manhattan.unified_pathfinder import (
            UnifiedPathFinder,
            PathFinderConfig,
        )
        router = UnifiedPathFinder(config=PathFinderConfig(), use_gpu=True)
        assert router is not None

    @requires_gpu
    def test_cuda_dijkstra_importable(self):
        from orthoroute.algorithms.manhattan.pathfinder.cuda_dijkstra import CUDADijkstra
        assert CUDADijkstra is not None

    def test_has_required_pipeline_methods(self):
        from orthoroute.algorithms.manhattan.unified_pathfinder import (
            UnifiedPathFinder,
            PathFinderConfig,
        )
        router = UnifiedPathFinder(config=PathFinderConfig(), use_gpu=False)
        assert callable(getattr(router, "initialize_graph", None)), \
            "Missing initialize_graph()"
        assert callable(getattr(router, "map_all_pads", None)), \
            "Missing map_all_pads()"
        assert callable(getattr(router, "route_multiple_nets", None)), \
            "Missing route_multiple_nets()"


class TestRouteMultipleNetsReturnShape:
    """Verify the return dict from route_multiple_nets has the correct structure."""

    REQUIRED_KEYS = {
        "success",
        "converged",
        "nets_routed",
        "total_nets",
        "iterations",
        "total_time_s",
        "iteration_metrics",
        "failed_nets",
        "overuse_sum",
        "overuse_edges",
        "barrel_conflicts",
        "excluded_nets",
        "excluded_net_ids",
        "error_code",
        "message",
    }

    def test_empty_request_returns_minimal_dict(self):
        """Empty net list: route_multiple_nets returns {} (no crash)."""
        from orthoroute.algorithms.manhattan.unified_pathfinder import (
            UnifiedPathFinder,
            PathFinderConfig,
        )
        router = UnifiedPathFinder(config=PathFinderConfig(), use_gpu=False)
        result = router.route_multiple_nets([])
        assert isinstance(result, dict)

    def test_required_keys_are_defined(self):
        """route_multiple_nets with a real board returns a dict with all REQUIRED_KEYS.

        Uses KiCadFileParser to load TestBackplane.kicad_pcb. Skips if the
        parser can't load enough pads. The empty-input case intentionally
        returns {} — verified by test_empty_request_returns_minimal_dict.
        """
        from pathlib import Path
        board_file = Path(__file__).parent.parent.parent / "TestBoards" / "TestBackplane.kicad_pcb"
        if not board_file.exists():
            pytest.skip("TestBackplane.kicad_pcb not found")
        try:
            from orthoroute.infrastructure.kicad.file_parser import KiCadFileParser
            board = KiCadFileParser().load_board(str(board_file))
        except Exception:
            pytest.skip("KiCadFileParser could not load board")
        if board is None:
            pytest.skip("board loaded as None")
        total_pads = sum(len(getattr(n, "pads", [])) for n in getattr(board, "nets", []))
        if total_pads < 10:
            pytest.skip(f"Only {total_pads} pads loaded — insufficient for routing test")

        from orthoroute.algorithms.manhattan.unified_pathfinder import (
            UnifiedPathFinder,
            PathFinderConfig,
        )
        router = UnifiedPathFinder(config=PathFinderConfig(), use_gpu=False)
        router.initialize_graph(board)
        router.map_all_pads(board)
        result = router.route_multiple_nets(board.nets[:1])
        missing = self.REQUIRED_KEYS - result.keys()
        assert not missing, f"route_multiple_nets result missing keys: {sorted(missing)}"
