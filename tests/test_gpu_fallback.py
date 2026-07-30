"""Regression coverage for the full-graph GPU seed fallback."""

import sys
import types

import numpy as np
import pytest

from tests.conftest import make_two_pad_board
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


def test_gpu_fullgraph_prices_source_seed_node():
    """A contracted portal seed must pay for ownership at its entry node."""
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception:
        pytest.skip("CUDA runtime unavailable")

    graph = types.SimpleNamespace(
        indptr=cp.asarray([0, 1, 2, 2, 2], dtype=cp.int32),
        indices=cp.asarray([3, 3], dtype=cp.int32),
    )
    solver = CUDADijkstra(graph=graph)
    path = solver.find_path_fullgraph_gpu_seeds(
        costs=cp.asarray([1.0, 2.0], dtype=cp.float32),
        src_seeds=np.asarray([0, 1], dtype=np.int32),
        dst_targets=np.asarray([3], dtype=np.int32),
        src_seed_costs=np.zeros(2, dtype=np.float32),
        dst_target_costs=np.zeros(1, dtype=np.float32),
        node_penalty=cp.asarray(
            [100.0, 0.0, 0.0, 0.0], dtype=cp.float32
        ),
    )

    assert path == [1, 3]


def test_gpu_fullgraph_high_cost_rounding_does_not_cycle_parents():
    """Equal float32 distances must not replace parents by lower node ID."""
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception:
        pytest.skip("CUDA runtime unavailable")

    # At this magnitude, adding a 0.4 mm edge does not change a float32.
    # Whole-key (distance, parent) minimization used to rewrite the source
    # parent and create a cycle while walking this bidirectional chain.
    graph = types.SimpleNamespace(
        indptr=cp.asarray([0, 1, 3, 5, 6], dtype=cp.int32),
        indices=cp.asarray([1, 0, 2, 1, 3, 2], dtype=cp.int32),
    )
    solver = CUDADijkstra(graph=graph)
    path = solver.find_path_fullgraph_gpu_seeds(
        costs=cp.full(6, 0.4, dtype=cp.float32),
        src_seeds=np.asarray([3], dtype=np.int32),
        dst_targets=np.asarray([0], dtype=np.int32),
        src_seed_costs=np.asarray([16_000_000.0], dtype=np.float32),
        dst_target_costs=np.zeros(1, dtype=np.float32),
    )

    assert path == [3, 2, 1, 0]


def test_gpu_fullgraph_matches_reference_dijkstra_on_random_costs():
    """CUDA SSSP must minimize edge plus entered-node costs."""
    import heapq

    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception:
        pytest.skip("CUDA runtime unavailable")

    width = 6
    height = 6
    node_count = width * height
    rows = [[] for _ in range(node_count)]
    for y in range(height):
        for x in range(width):
            node = y * width + x
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    rows[node].append(ny * width + nx)

    indptr = np.zeros(node_count + 1, dtype=np.int32)
    for node, neighbors in enumerate(rows):
        indptr[node + 1] = indptr[node] + len(neighbors)
    indices = np.asarray(
        [neighbor for row in rows for neighbor in row],
        dtype=np.int32,
    )
    graph = types.SimpleNamespace(
        indptr=cp.asarray(indptr),
        indices=cp.asarray(indices),
    )
    solver = CUDADijkstra(graph=graph)
    rng = np.random.default_rng(20260727)

    def reference_cost(costs, penalty, source, target):
        distance = np.full(node_count, np.inf, dtype=np.float64)
        distance[source] = float(penalty[source])
        queue = [(distance[source], source)]
        while queue:
            current, node = heapq.heappop(queue)
            if current != distance[node]:
                continue
            if node == target:
                return current
            for edge in range(indptr[node], indptr[node + 1]):
                neighbor = int(indices[edge])
                candidate = (
                    current
                    + float(costs[edge])
                    + float(penalty[neighbor])
                )
                if candidate < distance[neighbor]:
                    distance[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        return np.inf

    def returned_cost(path, costs, penalty):
        total = float(penalty[path[0]])
        for source, target in zip(path, path[1:]):
            start, end = indptr[source], indptr[source + 1]
            offset = np.flatnonzero(indices[start:end] == target)
            assert offset.size == 1
            total += (
                float(costs[start + int(offset[0])])
                + float(penalty[target])
            )
        return total

    for _ in range(8):
        costs = rng.uniform(0.05, 8.0, len(indices)).astype(np.float32)
        penalty = rng.uniform(0.0, 3.0, node_count).astype(np.float32)
        penalty[rng.random(node_count) < 0.7] = 0.0
        source, target = rng.choice(
            node_count, size=2, replace=False
        ).astype(np.int32)
        path = solver.find_path_fullgraph_gpu_seeds(
            costs=cp.asarray(costs),
            src_seeds=np.asarray([source], dtype=np.int32),
            dst_targets=np.asarray([target], dtype=np.int32),
            src_seed_costs=np.zeros(1, dtype=np.float32),
            dst_target_costs=np.zeros(1, dtype=np.float32),
            node_penalty=cp.asarray(penalty),
        )

        assert path
        assert returned_cost(path, costs, penalty) == pytest.approx(
            reference_cost(costs, penalty, int(source), int(target)),
            rel=1e-5,
            abs=1e-5,
        )
