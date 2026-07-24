"""Metal (MLX/Apple Silicon) solver parity tests.

Skip everywhere except a Mac with an Apple GPU. The contract under test:
MetalDijkstra must return paths of EQUAL COST to SimpleDijkstra (paths may
differ on equal-cost ties), and the try_attach_metal graft must retain the
CPU fallback.
"""

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core", reason="MLX not installed (not a Mac?)")

from orthoroute.algorithms.manhattan.pathfinder.metal_dijkstra import (
    METAL_AVAILABLE, MetalDijkstra, try_attach_metal,
)

if not METAL_AVAILABLE:
    pytest.skip("MLX present but no Metal GPU device", allow_module_level=True)

from orthoroute.algorithms.manhattan.unified_pathfinder import (
    Lattice3D, SimpleDijkstra,
)


@pytest.fixture(scope="module")
def solvers():
    lat = Lattice3D((0.0, 0.0, 8.0, 8.0), 0.4, layers=4)
    g = lat.build_graph(via_cost=0.7)
    indptr = np.asarray(g.indptr)
    indices = np.asarray(g.indices)
    costs = np.asarray(g.base_costs, dtype=np.float32)
    cpu = SimpleDijkstra(g, lattice=lat)
    gpu = MetalDijkstra(indptr, indices, lat.num_nodes,
                        plane_size=lat.x_steps * lat.y_steps)
    return lat, g, indptr, indices, costs, cpu, gpu


def _path_cost(path, indptr, indices, costs):
    if not path:
        return None
    total = 0.0
    for a, b in zip(path, path[1:]):
        s, e = indptr[a], indptr[a + 1]
        hits = np.nonzero(indices[s:e] == b)[0]
        assert len(hits), f"illegal hop {a}->{b}"
        total += float(costs[s + hits[0]])
    return round(total, 3)


def test_full_graph_parity(solvers):
    lat, g, indptr, indices, costs, cpu, gpu = solvers
    rng = np.random.default_rng(7)
    roi = np.arange(lat.num_nodes, dtype=np.int32)
    g2r = roi.copy()
    for _ in range(15):
        src = int(rng.integers(0, lat.num_nodes))
        dst = int(rng.integers(0, lat.num_nodes))
        c_cpu = _path_cost(cpu.find_path_roi(src, dst, costs, roi, g2r),
                           indptr, indices, costs)
        c_gpu = _path_cost(gpu.find_path_roi(src, dst, costs, roi, g2r),
                           indptr, indices, costs)
        assert c_cpu == c_gpu


def test_roi_and_penalty_parity(solvers):
    lat, g, indptr, indices, costs, cpu, gpu = solvers
    rng = np.random.default_rng(11)
    # Restrict to a vertical slab and price every 7th node
    roi = np.nonzero((np.arange(lat.num_nodes) % lat.x_steps) < 15)[0].astype(np.int32)
    g2r = np.full(lat.num_nodes, -1, dtype=np.int32)
    g2r[roi] = np.arange(len(roi))
    pen = np.zeros(len(roi), dtype=np.float32)
    pen[::7] = 5.0
    for _ in range(8):
        src = int(roi[rng.integers(0, len(roi))])
        dst = int(roi[rng.integers(0, len(roi))])
        c_cpu = _path_cost(
            cpu.find_path_roi(src, dst, costs, roi, g2r, node_penalty=pen),
            indptr, indices, costs)
        c_gpu = _path_cost(
            gpu.find_path_roi(src, dst, costs, roi, g2r, node_penalty=pen),
            indptr, indices, costs)
        assert c_cpu == c_gpu


def test_multisource_returns_valid_path(solvers):
    lat, g, indptr, indices, costs, cpu, gpu = solvers
    roi = np.arange(lat.num_nodes, dtype=np.int32)
    g2r = roi.copy()
    plane = lat.x_steps * lat.y_steps
    seeds = [(plane + 5, 0.0), (plane + 6, 0.5)]
    targets = [(2 * plane + 200, 0.0), (2 * plane + 201, 0.25)]
    result = gpu.find_path_multisource_multisink(seeds, targets, costs, roi, g2r)
    assert result is not None
    path, entry_layer, exit_layer = result
    assert path[0] in (seeds[0][0], seeds[1][0])
    assert path[-1] in (targets[0][0], targets[1][0])
    assert entry_layer == path[0] // plane
    assert _path_cost(path, indptr, indices, costs) is not None  # legal hops


def test_attach_keeps_cpu_fallback(solvers):
    lat, g, indptr, indices, costs, cpu, gpu = solvers
    fresh = SimpleDijkstra(g, lattice=lat)
    assert try_attach_metal(fresh, g, lat)
    assert hasattr(fresh, "metal_solver")
    roi = np.arange(lat.num_nodes, dtype=np.int32)
    plane = lat.x_steps * lat.y_steps
    src, dst = plane + 10, 2 * plane - 15  # both on In1.Cu (connected layer)
    path = fresh.find_path_roi(src, dst, costs, roi, roi.copy())
    assert path and path[0] == src and path[-1] == dst
