"""Metal (Apple Silicon) SSSP solver via MLX.

Drop-in GPU accelerator for SimpleDijkstra's two solver methods, running
on Apple Silicon's unified memory: the whole graph lives in one address
space shared by CPU and GPU, so there is no host<->device transfer layer
at all - the thing that dominates the CUDA backend's complexity.

Design:
- The CSR graph is converted once into a dense padded neighbor table
  (N x K, K = max out-degree, small and constant on a Manhattan lattice:
  2 lateral + adjacent vias). Missing slots point at node 0 with +inf
  cost, so they never win a relaxation.
- SSSP is frontier-driven fixpoint relaxation with packed keys: each
  node's state is one uint64 = (monotonic float32 dist bits << 32) | parent
  slot, so a single scatter-min atomically updates distance AND parent -
  the same trick as the CUDA backend's best_key pool.
- ROI restriction and ownership penalties are dense masks over N; on a
  48GB unified-memory machine full-N work arrays for the 9M-node monster
  board are ~40MB each - negligible.

Falls back to None (caller uses CPU) on any error: this solver must never
make a net fail that the CPU path could route.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    _dev = mx.default_device()
    METAL_AVAILABLE = _dev.type == mx.DeviceType.gpu
except Exception:  # mlx not installed or no GPU
    mx = None
    METAL_AVAILABLE = False

# Padded-slot sentinel cost: never wins a min-relaxation
_INF = np.float32(np.inf)


def try_attach_metal(solver, graph, lattice) -> bool:
    """Graft Metal acceleration onto a SimpleDijkstra instance.

    Wraps the solver's two path methods to try the Metal solver first and
    fall back to the original CPU implementation on None/failure - Metal
    must never make a net fail that the CPU could route. Returns True if
    attached.
    """
    if not METAL_AVAILABLE:
        logger.info("[METAL] MLX Metal GPU not available - staying on CPU")
        return False
    try:
        indptr = graph.indptr.get() if hasattr(graph.indptr, "get") else graph.indptr
        indices = graph.indices.get() if hasattr(graph.indices, "get") else graph.indices
        metal = MetalDijkstra(indptr, indices, lattice.num_nodes,
                              plane_size=lattice.x_steps * lattice.y_steps)
    except Exception as e:
        logger.warning(f"[METAL] init failed: {e} - staying on CPU")
        return False

    cpu_roi = solver.find_path_roi
    cpu_multi = solver.find_path_multisource_multisink

    def roi(src, dst, costs, roi_nodes, global_to_roi, node_penalty=None):
        path = metal.find_path_roi(src, dst, costs, roi_nodes, global_to_roi,
                                   node_penalty=node_penalty)
        if path is not None:
            return path
        return cpu_roi(src, dst, costs, roi_nodes, global_to_roi,
                       node_penalty=node_penalty)

    def multi(src_seeds, dst_targets, costs, roi_nodes, global_to_roi,
              node_penalty=None):
        result = metal.find_path_multisource_multisink(
            src_seeds, dst_targets, costs, roi_nodes, global_to_roi,
            node_penalty=node_penalty)
        if result is not None:
            return result
        return cpu_multi(src_seeds, dst_targets, costs, roi_nodes,
                         global_to_roi, node_penalty=node_penalty)

    solver.find_path_roi = roi
    solver.find_path_multisource_multisink = multi
    solver.metal_solver = metal
    logger.info("[METAL] Metal acceleration attached (CPU fallback retained)")
    return True


class MetalDijkstra:
    """Metal SSSP over the routing lattice via MLX.

    Mirrors SimpleDijkstra's find_path_roi / find_path_multisource_multisink
    signatures so the router can treat either interchangeably.
    """

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, num_nodes: int,
                 plane_size: Optional[int] = None,
                 min_roi_nodes: Optional[int] = None):
        if not METAL_AVAILABLE:
            raise RuntimeError("MLX Metal GPU not available")
        if min_roi_nodes is not None:
            self.MIN_ROI_NODES = min_roi_nodes

        self.N = int(num_nodes)
        self.plane_size = plane_size

        indptr = np.asarray(indptr)
        indices = np.asarray(indices)
        degrees = np.diff(indptr)
        self.K = int(degrees.max()) if len(degrees) else 0

        # Dense padded INCOMING-neighbor tables: in_nbr[v, k] = source node u
        # of the k-th edge ARRIVING at v, in_eid[v, k] = that edge's CSR
        # index (for cost lookup). Relaxation is then pure gather:
        #   dist'[v] = min(dist[v], min_k(dist[in_nbr[v,k]] + cost[in_eid[v,k]]))
        # No scatter, no per-round host sync, and the winning parent is just
        # in_nbr[v, argmin_k] - the Metal translation of the CUDA backend's
        # GPU-resident frontier/backtrace design. Padded slots: nbr=0,
        # eid=E (one extra cost slot pinned to +inf).
        E = len(indices)
        edge_src = np.repeat(np.arange(self.N, dtype=np.int64), degrees)
        order = np.argsort(indices, kind="stable")
        dst_sorted = indices[order]
        in_degrees = np.bincount(dst_sorted, minlength=self.N)
        self.K = int(max(self.K, in_degrees.max())) if len(in_degrees) else self.K
        in_start = np.zeros(self.N + 1, dtype=np.int64)
        np.cumsum(in_degrees, out=in_start[1:])
        in_nbr = np.zeros((self.N, self.K), dtype=np.int64)
        in_eid = np.full((self.N, self.K), E, dtype=np.int64)
        for k in range(self.K):
            has_k = in_degrees > k
            rows = np.nonzero(has_k)[0]
            pos = in_start[:-1][has_k] + k
            in_nbr[rows, k] = edge_src[order[pos]]
            in_eid[rows, k] = order[pos]

        self.in_nbr = mx.array(in_nbr)
        self.in_eid = mx.array(in_eid)
        self.E = E

        # Cost cache: rebuilt only when the (in-place mutated) cost array's
        # cheap fingerprint changes.
        self._cost_fp = None
        self._cost_mx = None  # (E+1,) with [+inf] appended for padded slots

        logger.info(f"[METAL] Solver ready: N={self.N:,} K={self.K} E={E:,} "
                    f"on {mx.default_device()}")

    # ------------------------------------------------------------------ #

    def _costs_to_gpu(self, costs) -> "mx.array":
        c = costs.get() if hasattr(costs, "get") else costs
        c = np.asarray(c, dtype=np.float32)
        fp = (float(c[:16].sum()), float(c[-16:].sum()), float(c.sum()))
        if fp != self._cost_fp:
            self._cost_fp = fp
            self._cost_mx = mx.array(np.append(c, _INF))
        return self._cost_mx

    # Below this ROI size the CPU heap Dijkstra wins: dense GPU relaxation
    # does O(N*K) work per round regardless of wave width, which only pays
    # off on big solves (same tradeoff as the CUDA backend's
    # gpu_roi_min_nodes threshold).
    MIN_ROI_NODES = 150_000

    def _run_sssp(self, seed_nodes: np.ndarray, seed_costs: np.ndarray,
                  costs, roi_nodes, node_penalty,
                  target_nodes: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Fixpoint dense relaxation. Returns (dist_f32[N], parent_i64[N])."""
        roi_size = len(roi_nodes) if roi_nodes is not None else self.N
        if roi_size < self.MIN_ROI_NODES:
            return None  # CPU fallback handles small solves faster
        cost_gpu = self._costs_to_gpu(costs)

        # ROI mask over N (True = routable). None -> everything allowed.
        if roi_nodes is not None and len(roi_nodes) < self.N:
            roi = roi_nodes.get() if hasattr(roi_nodes, "get") else roi_nodes
            allowed_np = np.zeros(self.N, dtype=bool)
            allowed_np[np.asarray(roi)] = True
            # Penalty is ROI-local per the CPU contract: scatter to full-N
            pen_np = np.zeros(self.N, dtype=np.float32)
            if node_penalty is not None:
                pen_np[np.asarray(roi)] = np.asarray(node_penalty, dtype=np.float32)
        else:
            allowed_np = None
            pen_np = None
            if node_penalty is not None:
                pen_np = np.asarray(node_penalty, dtype=np.float32)

        INF = mx.array(np.float32(np.inf))
        # Per-node arrival penalty and ROI restriction folded into a single
        # additive vector: +inf outside the ROI kills those candidates.
        extra_np = np.zeros(self.N, dtype=np.float32)
        if pen_np is not None:
            extra_np += pen_np
        if allowed_np is not None:
            extra_np[~allowed_np] = np.inf
        extra = mx.array(extra_np)[:, None] if extra_np.any() else None

        dist = mx.full((self.N,), np.float32(np.inf), dtype=mx.float32)
        parent = mx.full((self.N,), -1, dtype=mx.int32)
        seeds = mx.array(seed_nodes.astype(np.int64))
        dist = dist.at[seeds].minimum(mx.array(seed_costs.astype(np.float32)))
        mx.eval(dist)

        # Dense gather relaxation, fully GPU-resident. Host sync only every
        # CHECK_EVERY rounds for the fixpoint test.
        CHECK_EVERY = 8
        checkpoint = dist
        for rnd in range(1, self.N + 1):
            gathered = dist[self.in_nbr] + cost_gpu[self.in_eid]  # (N, K)
            if extra is not None:
                gathered = gathered + extra  # arrival penalty / ROI wall
            best = mx.min(gathered, axis=1)
            best_k = mx.argmin(gathered, axis=1)
            improved = best < dist
            new_parent = mx.take_along_axis(
                self.in_nbr, best_k[:, None].astype(mx.int64), axis=1
            ).squeeze(1).astype(mx.int32)
            parent = mx.where(improved, new_parent, parent)
            dist = mx.where(improved, best, dist)

            if rnd % CHECK_EVERY == 0:
                mx.eval(dist, parent)
                if bool(mx.all(dist == checkpoint)):  # no change in window
                    break
                checkpoint = dist

        dist_np = np.array(dist, dtype=np.float32)
        parent_np = np.array(parent, dtype=np.int64)
        return dist_np, parent_np

    # ------------------------------------------------------------------ #

    def find_path_multisource_multisink(self, src_seeds: List[Tuple[int, float]],
                                        dst_targets: List[Tuple[int, float]],
                                        costs, roi_nodes, global_to_roi,
                                        node_penalty=None):
        """Metal twin of SimpleDijkstra.find_path_multisource_multisink."""
        try:
            seed_nodes = np.array([n for n, _ in src_seeds], dtype=np.int64)
            seed_costs = np.array([c for _, c in src_seeds], dtype=np.float32)
            tgt_nodes = np.array([n for n, _ in dst_targets], dtype=np.int64)
            tgt_costs = np.array([c for _, c in dst_targets], dtype=np.float32)
            if len(seed_nodes) == 0 or len(tgt_nodes) == 0:
                return None

            out = self._run_sssp(seed_nodes, seed_costs, costs, roi_nodes,
                                 node_penalty, tgt_nodes)
            if out is None:
                return None
            dist, parent = out

            totals = dist[tgt_nodes] + tgt_costs
            best = int(np.argmin(totals))
            if not np.isfinite(totals[best]):
                return None
            end = int(tgt_nodes[best])

            path = self._backtrace(end, parent, set(int(n) for n in seed_nodes))
            if path is None or len(path) < 2:
                return None
            if self.plane_size:
                entry_layer = path[0] // self.plane_size
                exit_layer = path[-1] // self.plane_size
            else:
                entry_layer = exit_layer = None
            return path, entry_layer, exit_layer
        except Exception as e:
            logger.warning(f"[METAL] multisource solve failed: {e} - CPU will handle")
            return None

    def find_path_roi(self, src: int, dst: int, costs, roi_nodes, global_to_roi,
                      node_penalty=None) -> Optional[List[int]]:
        """Metal twin of SimpleDijkstra.find_path_roi."""
        try:
            out = self._run_sssp(np.array([src], dtype=np.int64),
                                 np.zeros(1, dtype=np.float32),
                                 costs, roi_nodes, node_penalty,
                                 np.array([dst], dtype=np.int64))
            if out is None:
                return None
            dist, parent = out
            if not np.isfinite(dist[dst]):
                return None
            return self._backtrace(int(dst), parent, {int(src)})
        except Exception as e:
            logger.warning(f"[METAL] ROI solve failed: {e} - CPU will handle")
            return None

    @staticmethod
    def _backtrace(end: int, parent: np.ndarray, seed_set) -> Optional[List[int]]:
        path = [end]
        cur = end
        for _ in range(len(parent)):
            if cur in seed_set:
                path.reverse()
                return path
            cur = int(parent[cur])
            if cur < 0:
                return None
            path.append(cur)
        return None  # cycle guard
