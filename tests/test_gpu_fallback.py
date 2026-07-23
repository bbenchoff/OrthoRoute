"""Regression coverage for the full-graph GPU seed fallback."""

import sys
import types

import numpy as np
import pytest

from conftest import make_two_pad_board
from orthoroute.algorithms.manhattan.unified_pathfinder import (
    PathFinderConfig,
    UnifiedPathFinder,
)
from orthoroute.algorithms.manhattan.pathfinder.cuda_dijkstra import CUDADijkstra


class _DeviceCosts(np.ndarray):
    """NumPy costs that advertise GPU residency to exercise the fast-path seam."""

    @property
    def device(self):
        return object()


class _FailingGpuSolver:
    def __init__(self, failure):
        self.failure = failure
        self.calls = 0
        self.last_kwargs = None

    def find_path_fullgraph_gpu_seeds(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if isinstance(self.failure, Exception):
            raise self.failure
        return self.failure


@pytest.mark.parametrize(
    "gpu_failure",
    [None, RuntimeError("synthetic kernel failure")],
    ids=["no-path", "exception"],
)
def test_gpu_seed_failure_falls_back_to_cost_based_routing(monkeypatch, gpu_failure):
    """A GPU miss or exception must not silently drop an otherwise routable net."""
    board = make_two_pad_board(layer_count=4)
    config = PathFinderConfig()
    config.portal_x_snap_max = 0.75
    router = UnifiedPathFinder(config=config, use_gpu=False)

    router.initialize_graph(board)
    router.map_all_pads(board)
    router.precompute_all_pad_escapes(board)
    router.prepare_routing_runtime()
    tasks = router._parse_requests(board.nets)

    router.accounting.total_cost = np.asarray(
        router.graph.base_costs
    ).copy().view(_DeviceCosts)
    gpu_solver = _FailingGpuSolver(gpu_failure)
    router.solver.gpu_solver = gpu_solver

    # The production branch imports CuPy only to confirm the GPU path is active.
    # A stub keeps this regression runnable on CPU-only contributor machines.
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))

    routed, failed = router._route_all(tasks, all_tasks=tasks, iteration=1)

    assert gpu_solver.calls == 1
    assert np.count_nonzero(gpu_solver.last_kwargs["src_seed_costs"]) > 0
    assert np.count_nonzero(gpu_solver.last_kwargs["dst_target_costs"]) > 0
    assert (routed, failed) == (1, 0)
    assert len(router.net_paths["TEST_NET"]) > 1


def test_gpu_seed_failure_skips_cpu_fullgraph_on_huge_graph(monkeypatch):
    """Large CUDA misses must remain negotiated failures, not CPU searches."""
    board = make_two_pad_board(layer_count=4)
    config = PathFinderConfig()
    config.portal_x_snap_max = 0.75
    config.gpu_fullgraph_fail_fast_nodes = 1
    router = UnifiedPathFinder(config=config, use_gpu=False)

    router.initialize_graph(board)
    router.map_all_pads(board)
    router.precompute_all_pad_escapes(board)
    router.prepare_routing_runtime()
    tasks = router._parse_requests(board.nets)

    router.accounting.total_cost = np.asarray(
        router.graph.base_costs
    ).copy().view(_DeviceCosts)
    gpu_solver = _FailingGpuSolver(None)
    router.solver.gpu_solver = gpu_solver
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))

    routed, failed = router._route_all(tasks, all_tasks=tasks, iteration=1)

    assert gpu_solver.calls == 1
    assert (routed, failed) == (0, 1)
    assert router.net_paths["TEST_NET"] == []


def test_gpu_roi_csr_prices_the_destination_node():
    """ROI extraction must preserve the CPU ownership-as-cost semantics."""
    solver = object.__new__(CUDADijkstra)
    solver.indptr = np.array([0, 1, 2], dtype=np.int32)
    solver.indices = np.array([1, 0], dtype=np.int32)

    _, _, weights = solver._extract_roi_csr(
        roi_nodes=np.array([0, 1], dtype=np.int32),
        global_to_roi=np.array([0, 1], dtype=np.int32),
        global_costs=np.array([2.0, 3.0], dtype=np.float32),
        node_penalty=np.array([0.0, 5.0], dtype=np.float32),
    )

    assert weights.tolist() == [7.0, 3.0]
