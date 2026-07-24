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

        # OUT-neighbor padded tables (uint32, flat) for the frontier scatter
        # kernel: threads relax the outgoing edges of frontier nodes only,
        # so per-round work is O(frontier * K), not O(N * K) like the dense
        # path. Padded slots: eid == E sentinel (skipped in-kernel).
        out_nbr = np.zeros((self.N, self.K), dtype=np.uint32)
        out_eid = np.full((self.N, self.K), E, dtype=np.uint32)
        for k in range(self.K):
            has_k = degrees > k
            rows = np.nonzero(has_k)[0]
            pos = indptr[:-1][has_k] + k
            out_nbr[rows, k] = indices[pos]
            out_eid[rows, k] = pos
        self.out_nbr = mx.array(out_nbr.reshape(-1))
        self.out_eid = mx.array(out_eid.reshape(-1))
        self._zero_penalty = mx.zeros((self.N,), dtype=mx.float32)

        # Frontier relax kernel. Distances are uint32-encoded float bits
        # (monotonic for non-negative floats) and INVERTED (UINT_MAX - bits)
        # so a single init_value=0 works for all atomic outputs: relaxation
        # is atomic_fetch_max on inverted encodings == min on distances,
        # and the frontier count naturally starts at 0.
        self._relax_kernel = mx.fast.metal_kernel(
            name="orthoroute_frontier_relax",
            input_names=["dist", "frontier", "out_nbr", "out_eid",
                         "costs", "penalty", "params"],
            output_names=["new_dist_inv", "next_frontier", "next_count"],
            source="""
    uint tid = thread_position_in_grid.x;
    uint K = params[0];
    uint E_PAD = params[1];
    uint CAP = params[2];
    uint F = frontier_shape[0];
    if (tid >= F * K) return;
    uint f = frontier[tid / K];
    uint slot = f * K + (tid % K);
    uint eid = out_eid[slot];
    if (eid == E_PAD) return;
    uint v = out_nbr[slot];
    float du = as_type<float>(dist[f]);
    float cand = du + costs[eid] + penalty[v];
    if (!(cand < INFINITY)) return;
    uint enc = as_type<uint>(cand);
    if (enc >= dist[v]) return;  // not an improvement on current state
    uint inv = 0xFFFFFFFFu - enc;
    uint old = atomic_fetch_max_explicit(&new_dist_inv[v], inv,
                                         memory_order_relaxed);
    if (inv > old) {
        uint pos = atomic_fetch_add_explicit(&next_count[0], 1u,
                                             memory_order_relaxed);
        if (pos < CAP) {
            atomic_store_explicit(&next_frontier[pos], v,
                                  memory_order_relaxed);
        }
    }
""",
            atomic_outputs=True,
        )

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

        # Penalty vector for the kernel: arrival penalty, +inf outside ROI
        penalty_full = mx.array(extra_np) if extra is not None else self._zero_penalty

        # Frontier scatter loop: per-round work is O(frontier * K). The
        # kernel atomically maxes INVERTED uint32-encoded distances (== min
        # on real distances) so init_value=0 serves all three outputs.
        INF_ENC = int(np.float32(np.inf).view(np.uint32))
        dist_np32 = np.full(self.N, INF_ENC, dtype=np.uint32)
        dist_np32[seed_nodes] = np.minimum(
            dist_np32[seed_nodes],
            seed_costs.astype(np.float32).view(np.uint32))
        dist_enc = mx.array(dist_np32)

        CAP = min(self.N, 4_000_000)
        params = mx.array(np.array([self.K, self.E, CAP], dtype=np.uint32))
        frontier = mx.array(np.unique(seed_nodes).astype(np.uint32))

        for _ in range(self.N):
            F = frontier.shape[0]
            new_inv, next_f, next_c = self._relax_kernel(
                inputs=[dist_enc, frontier, self.out_nbr, self.out_eid,
                        cost_gpu, penalty_full, params],
                grid=(F * self.K, 1, 1),
                threadgroup=(min(256, F * self.K), 1, 1),
                output_shapes=[(self.N,), (CAP,), (1,)],
                output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
                init_value=0,
            )
            dist_enc = mx.minimum(dist_enc,
                                  mx.array(np.uint32(0xFFFFFFFF)) - new_inv)
            count = int(next_c[0])  # the one host sync per round
            if count == 0:
                break
            takes = min(count, CAP)
            frontier = mx.array(
                np.unique(np.array(next_f[:takes], dtype=np.uint32)))

        dist_np = np.array(dist_enc, dtype=np.uint32).view(np.float32)

        # Race-free parent resolution: ONE dense in-table pass against the
        # final distances (same float op order as the kernel, so equality
        # is bitwise-safe). Seeds keep parent -1.
        dist = mx.array(dist_np)
        gathered = dist[self.in_nbr] + cost_gpu[self.in_eid]
        if extra is not None:
            gathered = gathered + extra
        best = mx.min(gathered, axis=1)
        best_k = mx.argmin(gathered, axis=1)
        cand_parent = mx.take_along_axis(
            self.in_nbr, best_k[:, None].astype(mx.int64), axis=1
        ).squeeze(1).astype(mx.int32)
        parent = mx.where(best == dist, cand_parent,
                          mx.full((self.N,), -1, dtype=mx.int32))
        mx.eval(parent)
        parent_np = np.array(parent, dtype=np.int64)
        parent_np[seed_nodes] = -1  # seeds are roots even on cost ties

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
