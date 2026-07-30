"""
Via accounting and barrel ownership for PathFinderRouter.

Extracted verbatim from unified_pathfinder.py (hardening audit, Phase 8
extraction 2). ViaAccounting is a collaborator operating on the router
instance passed to its constructor; PathFinderRouter keeps thin delegating
methods so every internal and external call site is unchanged.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


class ViaAccounting:
    """Tracks via pool usage, barrel ownership, and via-conflict detection.

    Holds no state of its own beyond the router reference: all reads and
    writes go through the router so behavior is identical to the
    pre-extraction methods.
    """

    def __init__(self, router):
        self._router = router

    def _accumulate_via_usage_for_path(
        self,
        node_path: List[int],
        net_id: str = None,
        *,
        col_use=None,
        seg_use=None,
    ):
        """
        Accumulate via column and segment usage for a committed path.
        Also registers via keepouts to prevent other nets from routing tracks through via locations.
        """
        if not hasattr(self._router, 'via_col_use') and not hasattr(self._router, 'via_seg_use'):
            return  # Pooling not enabled

        # Ensure via_keepouts_map exists
        if not hasattr(self._router, '_via_keepouts_map'):
            self._router._via_keepouts_map = {}

        idx_to_coord = self._router.lattice.idx_to_coord
        if col_use is None and hasattr(self._router, 'via_col_use'):
            col_use = self._router.via_col_use
        if seg_use is None and hasattr(self._router, 'via_seg_use'):
            seg_use = self._router.via_seg_use
        col_pool = col_use is not None
        seg_pool = seg_use is not None

        previous_via_xy = None
        for u, v in zip(node_path, node_path[1:]):
            xu, yu, zu = idx_to_coord(u)
            xv, yv, zv = idx_to_coord(v)

            # Check if it's a vertical transition (same x,y, different z)
            if xu == xv and yu == yv and zu != zv:
                # Adjacent-span graphs encode one physical barrel as a chain
                # of vertical hops. Count that contiguous chain once in the
                # column pool, while retaining per-segment occupancy below.
                via_xy = (xu, yu)
                if col_pool and via_xy != previous_via_xy:
                    col_use[xu, yu] += 1
                previous_via_xy = via_xy

                # Segment pooling: increment each segment crossed
                if seg_pool:
                    z_lo, z_hi = (zu, zv) if zu < zv else (zv, zu)
                    # Clamp to routing layers: 1..Nz-2
                    z_lo = max(1, min(z_lo, self._router._Nz - 2))
                    z_hi = max(1, min(z_hi, self._router._Nz - 2))
                    # Increment each segment z→z+1 in range [z_lo, z_hi)
                    for z in range(z_lo, z_hi):
                        seg_idx = z - 1  # Segment z→z+1 stored at index z-1
                        if 0 <= seg_idx < self._router._segZ:
                            seg_use[xu, yu, seg_idx] += 1

                # Register via keepouts for ALL layers the via touches (including endpoints!)
                # This prevents other nets from routing tracks through via locations
                if net_id:
                    z_lo, z_hi = (zu, zv) if zu < zv else (zv, zu)
                    for z in range(z_lo, z_hi + 1):  # Include both endpoints!
                        key = (z, xu, yu)
                        # First owner wins
                        if key not in self._router._via_keepouts_map:
                            self._router._via_keepouts_map[key] = net_id
            else:
                previous_via_xy = None

    def _rebuild_via_usage_from_committed(self):
        """Incrementally update via column/segment usage from committed net paths.

        Strategy:
        - First call: always full rebuild (no cached state yet).
        - Small dirty set (<= INCREMENTAL_THRESHOLD fraction of committed nets):
          subtract old contributions, add new ones for changed nets only.
        - Large dirty set (> threshold, i.e. most nets rerouted this iteration):
          faster to do a full rebuild than process hundreds of individual keys.

        _via_dirty_nets is populated by every net_paths[net_id] = ... assignment.
        """
        if not hasattr(self._router, 'via_col_use') and not hasattr(self._router, 'via_seg_use'):
            return

        # Accumulate on the host and upload once. Per-via writes into CuPy
        # arrays serialize the CPU and GPU thousands of times on large boards.
        col_use_cpu = (
            np.zeros((self._router._Nx, self._router._Ny), dtype=np.int16)
            if hasattr(self._router, 'via_col_use') else None
        )
        seg_use_cpu = (
            np.zeros((self._router._Nx, self._router._Ny, self._router._segZ), dtype=np.int16)
            if hasattr(self._router, 'via_seg_use') else None
        )

        self._router._via_contribution_cache = {}

        # Preserve portal keepouts (pre-registered escape via columns)
        if hasattr(self._router, '_via_keepouts_map'):
            portal_keepouts = {k: v for k, v in self._router._via_keepouts_map.items()
                               if v not in self._router.net_paths}
            self._router._via_keepouts_map.clear()
            self._router._via_keepouts_map.update(portal_keepouts)
            if portal_keepouts:
                logger.debug(f"[VIA-REBUILD] Preserved {len(portal_keepouts)} portal keepouts")

        for net_id, node_path in self._router.net_paths.items():
            if node_path and len(node_path) > 1:
                self._router._accumulate_via_usage_for_path(
                    node_path,
                    net_id=net_id,
                    col_use=col_use_cpu,
                    seg_use=seg_use_cpu,
                )

        # Escape barrels are attached to every committed routed path. Track
        # only geometry belonging to nets without a path, otherwise the same
        # physical barrel is counted twice.
        committed_nets = {
            net_id for net_id, path in self._router.net_paths.items() if path
        }
        self._router._track_escape_vias_in_via_usage(
            exclude_nets=committed_nets,
            col_use=col_use_cpu,
            seg_use=seg_use_cpu,
        )

        if col_use_cpu is not None:
            self._router.via_col_use[:] = self._router.accounting.xp.asarray(col_use_cpu)
        if seg_use_cpu is not None:
            self._router.via_seg_use[:] = self._router.accounting.xp.asarray(seg_use_cpu)

        if hasattr(self._router, '_via_keepouts_map'):
            logger.debug(f"[VIA-KEEPOUTS] Registered {len(self._router._via_keepouts_map)} via keepout cells")

        self._router._rebuild_node_owner()
        self._router._rebuild_path_node_use()

    def _add_via_contribution(self, net_id: str, path: list):
        """Add one net's via contributions to usage arrays and cache the keys."""
        if not hasattr(self._router, '_via_contribution_cache'):
            self._router._via_contribution_cache = {}
        if not hasattr(self._router, '_via_keepouts_map'):
            self._router._via_keepouts_map = {}

        idx_to_coord = self._router.lattice.idx_to_coord
        col_pool = hasattr(self._router, 'via_col_use')
        seg_pool = hasattr(self._router, 'via_seg_use')
        Nz = getattr(self._router, '_Nz', 0)
        segZ = getattr(self._router, '_segZ', 0)

        col_keys = []
        seg_keys = []
        keepout_keys = []

        for u, v in zip(path, path[1:]):
            xu, yu, zu = idx_to_coord(u)
            xv, yv, zv = idx_to_coord(v)

            if xu == xv and yu == yv and zu != zv:
                if col_pool:
                    self._router.via_col_use[xu, yu] += 1
                    col_keys.append((xu, yu))

                if seg_pool:
                    z_lo = max(1, min(min(zu, zv), Nz - 2))
                    z_hi = max(1, min(max(zu, zv), Nz - 2))
                    for z in range(z_lo, z_hi):
                        seg_idx = z - 1
                        if 0 <= seg_idx < segZ:
                            self._router.via_seg_use[xu, yu, seg_idx] += 1
                            seg_keys.append((xu, yu, seg_idx))

                z_lo2, z_hi2 = (zu, zv) if zu < zv else (zv, zu)
                for z in range(z_lo2, z_hi2 + 1):
                    key = (z, xu, yu)
                    if key not in self._router._via_keepouts_map:
                        self._router._via_keepouts_map[key] = net_id
                        keepout_keys.append(key)

        self._router._via_contribution_cache[net_id] = {
            'col_keys': col_keys,
            'seg_keys': seg_keys,
            'keepout_keys': keepout_keys,
        }

    def _remove_via_contribution(self, net_id: str):
        """Subtract a net's cached via contributions from usage arrays."""
        contrib = self._router._via_contribution_cache.pop(net_id, None)
        if contrib is None:
            return

        if hasattr(self._router, 'via_col_use'):
            for xu, yu in contrib['col_keys']:
                self._router.via_col_use[xu, yu] = max(0, self._router.via_col_use[xu, yu] - 1)

        if hasattr(self._router, 'via_seg_use'):
            for xu, yu, seg_idx in contrib['seg_keys']:
                self._router.via_seg_use[xu, yu, seg_idx] = max(0, self._router.via_seg_use[xu, yu, seg_idx] - 1)

        if hasattr(self._router, '_via_keepouts_map'):
            for key in contrib['keepout_keys']:
                self._router._via_keepouts_map.pop(key, None)

    def _mark_via_barrel_ownership_for_path(self, net_name: str, path: List[int]) -> None:
        """
        Mark via barrel nodes as owned by this net IMMEDIATELY after commit.

        CRITICAL: This must be called AFTER each net commits, not just at iteration start!
        Without this, later nets in the same iteration don't see earlier nets' via barrels.
        """
        if not path or len(path) < 2:
            return

        net_id = self._router._get_net_id(net_name)
        graph_path = self._router._path_without_dynamic_escape_chains(
            net_name, path
        )
        owned_nodes = self._router._via_nodes_for_path(graph_path)
        for node_idx in owned_nodes:
            members = self._router._node_owner_members.setdefault(
                node_idx, set()
            )
            members.add(net_id)
            self._router.node_owner[node_idx] = (
                net_id if len(members) == 1 else -2
            )
        portal_nodes = self._router._selected_portal_clearance_nodes(net_name)
        for node_idx in portal_nodes:
            members = self._router._portal_clearance_owner_members.setdefault(
                node_idx, set()
            )
            members.add(net_id)
            self._router.portal_clearance_owner[node_idx] = (
                net_id if len(members) == 1 else -2
            )

        if owned_nodes and self._router.node_owner_gpu is not None:
            owned_nodes_gpu = cp.asarray(
                owned_nodes, dtype=cp.int32
            )
            self._router.node_owner_gpu[owned_nodes_gpu] = cp.asarray(
                self._router.node_owner[owned_nodes], dtype=cp.int32
            )
        if portal_nodes and self._router.portal_clearance_owner_gpu is not None:
            portal_nodes_gpu = cp.asarray(
                portal_nodes, dtype=cp.int32
            )
            self._router.portal_clearance_owner_gpu[
                portal_nodes_gpu
            ] = cp.asarray(
                self._router.portal_clearance_owner[portal_nodes],
                dtype=cp.int32,
            )

    def _clear_via_barrel_ownership_for_path(
        self, net_name: str, path: List[int]
    ) -> None:
        """Remove a ripped-up path without leaving ghost barrel owners."""
        if not path or len(path) < 2:
            return
        net_id = self._router._get_net_id(net_name)
        graph_path = self._router._path_without_dynamic_escape_chains(
            net_name, path
        )
        changed_nodes = self._router._via_nodes_for_path(graph_path)
        for node_idx in changed_nodes:
            members = self._router._node_owner_members.get(node_idx)
            if not members:
                self._router.node_owner[node_idx] = -1
                continue
            members.discard(net_id)
            if not members:
                self._router._node_owner_members.pop(node_idx, None)
                self._router.node_owner[node_idx] = -1
            elif len(members) == 1:
                self._router.node_owner[node_idx] = next(iter(members))
            else:
                self._router.node_owner[node_idx] = -2
        portal_nodes = self._router._selected_portal_clearance_nodes(net_name)
        for node_idx in portal_nodes:
            members = self._router._portal_clearance_owner_members.get(node_idx)
            if not members:
                self._router.portal_clearance_owner[node_idx] = -1
                continue
            members.discard(net_id)
            if not members:
                self._router._portal_clearance_owner_members.pop(node_idx, None)
                self._router.portal_clearance_owner[node_idx] = -1
            elif len(members) == 1:
                self._router.portal_clearance_owner[node_idx] = next(iter(members))
            else:
                self._router.portal_clearance_owner[node_idx] = -2

        if changed_nodes and self._router.node_owner_gpu is not None:
            nodes_gpu = cp.asarray(changed_nodes, dtype=cp.int32)
            self._router.node_owner_gpu[nodes_gpu] = cp.asarray(
                self._router.node_owner[changed_nodes], dtype=cp.int32
            )
        if portal_nodes and self._router.portal_clearance_owner_gpu is not None:
            nodes_gpu = cp.asarray(portal_nodes, dtype=cp.int32)
            self._router.portal_clearance_owner_gpu[nodes_gpu] = cp.asarray(
                self._router.portal_clearance_owner[portal_nodes],
                dtype=cp.int32,
            )

    def _via_nodes_for_hop(self, u: int, v: int) -> List[int]:
        xu, yu, zu = self._router.lattice.idx_to_coord(u)
        xv, yv, zv = self._router.lattice.idx_to_coord(v)
        if xu != xv or yu != yv or zu == zv:
            return []
        z_lo, z_hi = sorted((zu, zv))
        return [
            self._router.lattice.node_idx(xu, yu, z)
            for z in range(z_lo, z_hi + 1)
        ]

    def _via_nodes_for_path(self, path: List[int]) -> List[int]:
        nodes = set()
        for u, v in zip(path, path[1:]):
            nodes.update(self._router._via_nodes_for_hop(u, v))
        return sorted(nodes)

    def _apply_via_pooling_penalties(self, pres_fac: float):
        """
        Apply via column and segment pooling penalties to vertical edge costs.

        Uses GPU-accelerated CUDA kernel when available (800ms → <2ms speedup!)
        Falls back to CPU vectorized implementation if GPU unavailable.
        """
        import time
        t0 = time.perf_counter()

        if not hasattr(self._router, 'via_col_pres') and not hasattr(self._router, 'via_seg_pres'):
            return

        # Check if metadata is available
        if not hasattr(self._router, '_via_edge_metadata') or self._router._via_edge_metadata is None:
            logger.warning("[VIA-POOL] Metadata not built, falling back to sequential")
            self._router._apply_via_pooling_penalties_sequential(pres_fac)
            return

        # Try GPU kernel first if available
        if hasattr(self._router, 'via_kernel_manager') and self._router.via_kernel_manager.use_gpu:
            try:
                # Check if there are any penalties to apply (GPU can check this fast)
                xp = cp if hasattr(self._router.via_col_pres, 'device') else np
                if hasattr(self._router, 'via_col_pres'):
                    col_max = float(xp.max(self._router.via_col_pres))
                    if col_max == 0 and hasattr(self._router, 'via_seg_pres'):
                        seg_max = float(xp.max(self._router.via_seg_pres))
                        if seg_max == 0:
                            return  # No penalties needed

                col_weight = float(getattr(self._router.config, "via_column_weight", 1.0))
                seg_weight = float(getattr(self._router.config, "via_segment_weight", 1.0))

                penalty_count = self._router.via_kernel_manager.apply_via_penalties(
                    via_metadata=self._router._via_edge_metadata,
                    via_col_pres_gpu=self._router.via_col_pres,
                    via_seg_pres_gpu=self._router.via_seg_pres if hasattr(self._router, 'via_seg_pres') else None,
                    col_weight=col_weight * pres_fac,
                    seg_weight=seg_weight * pres_fac,
                    total_cost_gpu=self._router.accounting.total_cost,
                    Ny=self._router._Ny,
                    segZ=self._router._segZ if hasattr(self._router, '_segZ') else 0
                )
                return  # GPU kernel succeeded
            except Exception as e:
                logger.warning(f"[VIA-POOL] GPU kernel failed: {e}, falling back to CPU")

        # CPU fallback - original vectorized implementation
        col_weight = float(getattr(self._router.config, "via_column_weight", 1.0))
        seg_weight = float(getattr(self._router.config, "via_segment_weight", 1.0))

        # Get cost array
        total_cost = self._router.accounting.total_cost
        if self._router.accounting.use_gpu:
            total_cost_cpu = total_cost.get()
        else:
            total_cost_cpu = total_cost

        # Get precomputed metadata
        via_edge_indices = self._router._via_edge_metadata['indices']
        via_xy_coords = self._router._via_edge_metadata['xy_coords']
        z_lo = self._router._via_edge_metadata['z_lo']
        z_hi = self._router._via_edge_metadata['z_hi']

        num_via_edges = len(via_edge_indices)
        if num_via_edges == 0:
            return

        # Initialize penalties array
        penalties = np.zeros(num_via_edges, dtype=np.float32)

        # Vectorized column penalty computation
        if hasattr(self._router, 'via_col_pres'):
            col_penalties = self._router.via_col_pres[via_xy_coords[:, 0], via_xy_coords[:, 1]]
            penalties += col_weight * col_penalties

        # Vectorized segment penalty computation (using prefix sums)
        if hasattr(self._router, 'via_seg_prefix'):
            # Compute prefix indices for range queries
            # Segment index mapping: z-1→z is stored at index z-2
            hi_idx = z_hi - 2  # Index for upper bound
            lo_idx = z_lo - 2  # Index for lower bound

            # Create masks for valid indices
            valid_mask = z_hi > z_lo  # Only process edges spanning multiple layers
            hi_valid = (hi_idx >= 0) & (hi_idx < self._router._segZ)
            lo_valid = (lo_idx >= 0) & (lo_idx < self._router._segZ)

            # Fetch prefix values with bounds checking
            pref_hi = np.zeros(num_via_edges, dtype=np.float32)
            pref_lo = np.zeros(num_via_edges, dtype=np.float32)

            # Use advanced indexing for valid entries
            if np.any(hi_valid):
                valid_hi_edges = hi_valid
                pref_hi[valid_hi_edges] = self._router.via_seg_prefix[
                    via_xy_coords[valid_hi_edges, 0],
                    via_xy_coords[valid_hi_edges, 1],
                    hi_idx[valid_hi_edges]
                ]

            if np.any(lo_valid):
                valid_lo_edges = lo_valid
                pref_lo[valid_lo_edges] = self._router.via_seg_prefix[
                    via_xy_coords[valid_lo_edges, 0],
                    via_xy_coords[valid_lo_edges, 1],
                    lo_idx[valid_lo_edges]
                ]

            # Compute segment penalties: prefix[hi] - prefix[lo]
            seg_penalties = (pref_hi - pref_lo) * valid_mask
            penalties += seg_weight * seg_penalties

        # STEP 2.7: Apply "leave-hot-layer" via discount using layer bias
        if hasattr(self._router, 'layer_bias'):
            k = float(getattr(self._router.config, 'via_hot_layer_discount', 0.20))

            # Get source and destination layer biases for each via edge
            src_bias = self._router.layer_bias[z_lo]
            dst_bias = self._router.layer_bias[z_hi]

            # Cheaper to leave hot layers, more expensive to land on hot layers
            via_discount = (1.0 - k * np.maximum(src_bias, 0.0)) * (1.0 + 0.5 * k * np.maximum(dst_bias, 0.0))
            penalties *= via_discount  # Apply discount/markup to penalties

            # Log discount statistics if significant
            avg_discount = float(np.mean(via_discount))
            if abs(avg_discount - 1.0) > 0.05:
                logger.debug(f"[VIA-DISCOUNT] Average via discount factor: {avg_discount:.3f}")

        # Apply penalties to cost array (vectorized)
        penalty_mask = penalties > 0
        total_cost_cpu[via_edge_indices[penalty_mask]] += pres_fac * penalties[penalty_mask]
        penalties_applied = np.sum(penalty_mask)

        # Update GPU if needed
        if self._router.accounting.use_gpu:
            self._router.accounting.total_cost[:] = cp.asarray(total_cost_cpu)

        elapsed = time.perf_counter() - t0
        if penalties_applied > 0:
            logger.debug(f"[VIA-POOL-PERF] Vectorized penalty application: {num_via_edges} edges, {penalties_applied} penalties in {elapsed:.3f}s")
        else:
            logger.debug(f"[VIA-POOL-PERF] No penalties applied ({num_via_edges} edges checked in {elapsed:.3f}s)")

    def _apply_via_pooling_penalties_sequential(self, pres_fac: float):
        """Sequential fallback for via pooling penalties (for debugging/comparison)"""
        col_weight = float(getattr(self._router.config, "via_column_weight", 1.0))
        seg_weight = float(getattr(self._router.config, "via_segment_weight", 1.0))

        # Get cost array
        total_cost = self._router.accounting.total_cost
        if self._router.accounting.use_gpu:
            total_cost_cpu = total_cost.get()
        else:
            total_cost_cpu = total_cost

        # Get graph data
        indptr = self._router.graph.indptr.get() if hasattr(self._router.graph.indptr, 'get') else self._router.graph.indptr
        indices = self._router.graph.indices.get() if hasattr(self._router.graph.indices, 'get') else self._router.graph.indices

        idx_to_coord = self._router.lattice.idx_to_coord
        penalties_applied = 0

        # Find via edge indices (where _via_edges is True)
        via_edge_indices = np.where(self._router._via_edges)[0]

        for ei in via_edge_indices:
            u = int(np.searchsorted(indptr, ei, side='right') - 1)
            if 0 <= u < len(indptr) - 1 and indptr[u] <= ei < indptr[u + 1]:
                v = int(indices[ei])
                xu, yu, zu = idx_to_coord(u)
                xv, yv, zv = idx_to_coord(v)

                penalty = 0.0

                # Column penalty
                if hasattr(self._router, 'via_col_pres'):
                    penalty += col_weight * self._router.via_col_pres[xu, yu]

                # Segment penalty (use prefix for fast range sum)
                if hasattr(self._router, 'via_seg_prefix'):
                    z_lo, z_hi = (zu, zv) if zu < zv else (zv, zu)
                    z_lo = max(1, min(z_lo, self._router._Nz - 2))
                    z_hi = max(1, min(z_hi, self._router._Nz - 2))
                    if z_hi >= z_lo:  # Allow equal (single-segment vias)
                        hi_idx = z_hi - 2
                        lo_idx = z_lo - 2
                        pref_hi = self._router.via_seg_prefix[xu, yu, hi_idx] if 0 <= hi_idx < self._router._segZ else 0.0
                        pref_lo = self._router.via_seg_prefix[xu, yu, lo_idx] if 0 <= lo_idx < self._router._segZ else 0.0
                        seg_sum = pref_hi - pref_lo
                        penalty += seg_weight * seg_sum

                if penalty > 0:
                    total_cost_cpu[ei] += pres_fac * penalty
                    penalties_applied += 1

        # Update GPU if needed
        if self._router.accounting.use_gpu:
            self._router.accounting.total_cost[:] = cp.asarray(total_cost_cpu)

        if penalties_applied > 0:
            logger.debug(f"[VIA-POOL] Sequential: Applied pooling penalties to {penalties_applied} via edges")

    def _spatial_via_overuse_total(self) -> int:
        """Return over-capacity via-column and via-segment occupancy."""
        total = 0
        for use_name, cap_name in (
            ("via_col_use", "via_col_cap"),
            ("via_seg_use", "via_seg_cap"),
        ):
            if not hasattr(self._router, use_name) or not hasattr(self._router, cap_name):
                continue
            use = getattr(self._router, use_name)
            cap = getattr(self._router, cap_name)
            xp = (
                cp
                if GPU_AVAILABLE and isinstance(use, cp.ndarray)
                else np
            )
            subtotal = xp.maximum(0, use - cap).sum()
            if hasattr(subtotal, "get"):
                subtotal = subtotal.get()
            total += int(subtotal)
        return total

    def _block_via_edges_with_collisions(self):
        """
        Hard-block via edges with spatial collisions by setting costs to infinity.

        This prevents PathFinder from using via edges where the column or any
        spanned segment is already at capacity. Combined with soft penalties,
        this ensures no via spatial violations in the final routing.

        Uses GPU-accelerated CUDA kernel when available (30s → <1ms speedup!)
        """
        if not hasattr(self._router, '_via_edge_metadata') or self._router._via_edge_metadata is None:
            logger.warning("[HARD-BLOCK] No via edge metadata, skipping")
            return

        if not hasattr(self._router, 'via_col_use') and not hasattr(self._router, 'via_seg_use'):
            # No via spatial tracking enabled
            return

        # Try GPU kernel first if available
        if hasattr(self._router, 'via_kernel_manager') and self._router.via_kernel_manager.use_gpu:
            try:
                blocked_count = self._router.via_kernel_manager.hard_block_via_edges(
                    via_metadata=self._router._via_edge_metadata,
                    via_col_use_gpu=self._router.via_col_use,
                    via_col_cap_gpu=self._router.via_col_cap,
                    via_seg_use_gpu=self._router.via_seg_use if hasattr(self._router, 'via_seg_use') else None,
                    via_seg_cap_gpu=self._router.via_seg_cap if hasattr(self._router, 'via_seg_cap') else None,
                    total_cost_gpu=self._router.accounting.total_cost,
                    Ny=self._router._Ny,
                    segZ=self._router._segZ if hasattr(self._router, '_segZ') else 0
                )
                return  # GPU kernel succeeded
            except Exception as e:
                logger.warning(f"[HARD-BLOCK] GPU kernel failed: {e}, falling back to CPU")

        # CPU fallback
        via_edges = self._router._via_edge_metadata
        edge_indices = via_edges['indices']
        xy_coords = via_edges['xy_coords']
        z_lo = via_edges['z_lo']
        z_hi = via_edges['z_hi']

        # Convert from GPU to CPU if needed
        if hasattr(edge_indices, 'get'):
            edge_indices = edge_indices.get()
        if hasattr(xy_coords, 'get'):
            xy_coords = xy_coords.get()
        if hasattr(z_lo, 'get'):
            z_lo = z_lo.get()
        if hasattr(z_hi, 'get'):
            z_hi = z_hi.get()

        # Get cost array
        total_cost = self._router.accounting.total_cost
        if self._router.accounting.use_gpu:
            total_cost_cpu = total_cost.get()
        else:
            total_cost_cpu = total_cost

        # Get via arrays (convert from GPU if needed)
        via_col_use = self._router.via_col_use.get() if hasattr(self._router.via_col_use, 'get') else self._router.via_col_use
        via_col_cap = self._router.via_col_cap.get() if hasattr(self._router.via_col_cap, 'get') else self._router.via_col_cap
        via_seg_use = self._router.via_seg_use.get() if hasattr(self._router, 'via_seg_use') and hasattr(self._router.via_seg_use, 'get') else getattr(self._router, 'via_seg_use', None)
        via_seg_cap = self._router.via_seg_cap.get() if hasattr(self._router, 'via_seg_cap') and hasattr(self._router.via_seg_cap, 'get') else getattr(self._router, 'via_seg_cap', None)

        blocked_count = 0

        for i in range(len(edge_indices)):
            xu, yu = int(xy_coords[i, 0]), int(xy_coords[i, 1])
            z_start, z_end = int(z_lo[i]), int(z_hi[i])
            edge_idx = edge_indices[i]

            # Check column capacity
            col_blocked = False
            if via_col_use is not None and via_col_cap is not None:
                if via_col_use[xu, yu] >= via_col_cap[xu, yu]:
                    col_blocked = True

            # Check segment capacity for all spanned segments
            seg_blocked = False
            if via_seg_use is not None and via_seg_cap is not None:
                for z in range(z_start, z_end):
                    seg_idx = z - 1  # Segments indexed from 0
                    if 0 <= seg_idx < self._router._segZ:
                        if via_seg_use[xu, yu, seg_idx] >= via_seg_cap[xu, yu, seg_idx]:
                            seg_blocked = True
                            break

            # Hard-block if at capacity
            if col_blocked or seg_blocked:
                total_cost_cpu[edge_idx] = np.float32('inf')
                blocked_count += 1

        # Copy back to GPU if needed
        if self._router.accounting.use_gpu:
            total_cost[:len(total_cost_cpu)] = cp.asarray(total_cost_cpu)

        if blocked_count > 0:
            logger.info(f"[HARD-BLOCK-CPU] Blocked {blocked_count} via edges at capacity (using CPU fallback)")

    def _identify_via_edges(self):
        """Mark which edges are vias (vertical transitions between layers)"""
        if getattr(self._router.graph, "edge_kind", None) is not None:
            # CSR finalization has already computed this classification.  Do
            # not download and rescan the multi-gigabyte CSR on GPU builds.
            self._router._via_edges = self._router.graph.edge_kind.astype(np.bool_, copy=True)
            logger.info(
                f"Identified {int(self._router._via_edges.sum())} via edges "
                "from cached edge metadata"
            )
            return

        indptr = self._router.graph.indptr.get() if hasattr(self._router.graph.indptr, 'get') else self._router.graph.indptr
        indices = self._router.graph.indices.get() if hasattr(self._router.graph.indices, 'get') else self._router.graph.indices

        # Use numpy boolean array instead of Python set for memory efficiency
        # With 27M edges, this uses ~30MB instead of ~750MB
        num_edges = int(indptr[-1])
        self._router._via_edges = np.zeros(num_edges, dtype=bool)

        # Use arithmetic instead of idx_to_coord for speed
        plane_size = self._router.lattice.x_steps * self._router.lattice.y_steps

        for u in range(len(indptr) - 1):
            uz = u // plane_size  # Fast arithmetic instead of idx_to_coord
            for ei in range(int(indptr[u]), int(indptr[u+1])):
                v = int(indices[ei])
                vz = v // plane_size
                # Via edge: different layer (same x,y is implicit in Manhattan CSR construction)
                self._router._via_edges[ei] = (uz != vz)

        logger.info(f"Identified {int(self._router._via_edges.sum())} via edges")

    def _build_via_edge_metadata(self):
        """
        Precompute via edge metadata for vectorized penalty application.

        Keeps metadata on GPU if available for zero-copy kernel execution.
        """
        import time
        t0 = time.perf_counter()
        use_via_gpu = self._router.config.use_gpu and GPU_AVAILABLE

        if use_via_gpu:
            # Build directly from the device-resident CSR.  Pulling indices
            # back to host and uploading four derived arrays temporarily
            # duplicates several gigabytes on large backplanes.
            via_edge_indices = cp.flatnonzero(
                self._router.graph.edge_kind_gpu
            ).astype(cp.int32)
            num_via_edges = int(via_edge_indices.size)
            if num_via_edges == 0:
                self._router._via_edge_metadata = None
                return

            u_indices = (
                cp.searchsorted(
                    self._router.graph.indptr, via_edge_indices, side='right'
                ) - 1
            ).astype(cp.int32)
            v_indices = self._router.graph.indices[via_edge_indices]
            plane_size = self._router.lattice.x_steps * self._router.lattice.y_steps

            xu = (u_indices % plane_size) % self._router.lattice.x_steps
            yu = (u_indices % plane_size) // self._router.lattice.x_steps
            zu = u_indices // plane_size
            zv = v_indices // plane_size

            via_xy_coords = cp.stack([xu, yu], axis=1).astype(cp.int32)
            z_lo = cp.clip(
                cp.minimum(zu, zv), 1, self._router.lattice.layers - 2
            ).astype(cp.int32)
            z_hi = cp.clip(
                cp.maximum(zu, zv), 1, self._router.lattice.layers - 2
            ).astype(cp.int32)

            self._router._via_edge_metadata = {
                'indices': via_edge_indices,
                'xy_coords': via_xy_coords,
                'z_lo': z_lo,
                'z_hi': z_hi,
            }
            logger.info(
                f"[VIA-METADATA] Built metadata for {num_via_edges} "
                f"via edges on GPU in {time.perf_counter() - t0:.3f}s"
            )
            return

        # Get via edge indices
        via_edge_indices = np.where(self._router._via_edges)[0]
        num_via_edges = len(via_edge_indices)

        if num_via_edges == 0:
            self._router._via_edge_metadata = None
            return

        # Get graph data
        indptr = self._router.graph.indptr.get() if hasattr(self._router.graph.indptr, 'get') else self._router.graph.indptr
        indices = self._router.graph.indices.get() if hasattr(self._router.graph.indices, 'get') else self._router.graph.indices

        # Precompute u (source node) for each via edge using searchsorted
        u_indices = np.searchsorted(indptr, via_edge_indices, side='right') - 1

        # Get v (destination node) for each via edge
        v_indices = indices[via_edge_indices]

        # Convert to coordinates (vectorized)
        plane_size = self._router.lattice.x_steps * self._router.lattice.y_steps

        # u coordinates
        xu = (u_indices % plane_size) % self._router.lattice.x_steps
        yu = (u_indices % plane_size) // self._router.lattice.x_steps
        zu = u_indices // plane_size

        # v coordinates
        xv = (v_indices % plane_size) % self._router.lattice.x_steps
        yv = (v_indices % plane_size) // self._router.lattice.x_steps
        zv = v_indices // plane_size

        # For via edges, x,y should be same (sanity check in debug mode)
        # Store just one (x,y) coordinate per via edge
        via_xy_coords = np.stack([xu, yu], axis=1).astype(np.int32)

        # Store z ranges (lo, hi) for each via edge
        z_lo = np.minimum(zu, zv)
        z_hi = np.maximum(zu, zv)

        # Clamp z values to valid routing layers (1..Nz-2)
        z_lo = np.clip(z_lo, 1, self._router.lattice.layers - 2)
        z_hi = np.clip(z_hi, 1, self._router.lattice.layers - 2)

        self._router._via_edge_metadata = {
            'indices': via_edge_indices.astype(np.int32),
            'xy_coords': via_xy_coords,
            'z_lo': z_lo.astype(np.int32),
            'z_hi': z_hi.astype(np.int32),
        }
        logger.info(f"[VIA-METADATA] Built metadata for {num_via_edges} via edges on CPU in {time.perf_counter() - t0:.3f}s")

    def _detect_barrel_conflicts(self) -> Tuple[np.ndarray, int]:
        """
        Detect via barrel conflicts across all committed paths (GPU-accelerated).

        This is THE critical fix for shorting_items violations!
        Detects when committed edges touch via barrel nodes owned by other nets.

        Returns:
            (conflict_edge_indices, conflict_count)
        """
        import numpy as np

        logger.debug("[BARREL-CONFLICT] Checking for via barrel conflicts...")
        self._router._barrel_conflict_nets = set()
        self._router._barrel_owner_nets = set()
        self._router._barrel_victim_nets = set()
        self._router._barrel_owner_portal_keys = set()
        self._router._last_exact_barrel_conflict_count = 0
        self._router._last_path_node_conflict_count = 0
        self._router._last_escape_conflict_count = 0
        self._router._last_portal_grid_conflict_count = 0
        self._router._portal_grid_owner_nets = set()
        self._router._portal_grid_victim_nets = set()
        self._router._portal_grid_pairs = set()
        self._router._escape_conflict_pairs = set()
        self._router._exact_barrel_pairs = set()
        self._router._path_node_conflict_pairs = set()
        self._router._path_node_conflict_scores = {}
        self._router._physical_conflict_scores = defaultdict(int)
        self._router._last_exact_barrel_details = []

        # Bail out if node_owner not initialized
        if not hasattr(self._router, 'node_owner') or self._router.node_owner is None:
            logger.info("[BARREL-CONFLICT] Skipping: node_owner not initialized")
            return np.array([], dtype=np.int32), 0

        # Ensure edge_src mapping exists
        self._router._ensure_edge_src_map()

        # Use _net_paths (with underscore) - this is what the negotiation loop uses!
        paths_dict = getattr(self._router, '_net_paths', {})
        if not paths_dict:
            # Fallback to net_paths if _net_paths doesn't exist
            paths_dict = getattr(self._router, 'net_paths', {})

        if not paths_dict:
            logger.info("[BARREL-CONFLICT] Skipping: no committed paths found")
            return np.array([], dtype=np.int32), 0

        logger.info(f"[BARREL-CONFLICT] Found {len(paths_dict)} committed paths")

        # Collect cached committed edges in vectorized chunks. Recomputing
        # every path and extending Python lists one integer at a time made
        # this audit dominate large-board iterations.
        edge_chunks = []
        net_id_chunks = []

        for net_name, path in paths_dict.items():
            if not path or len(path) < 2:
                continue

            net_id = self._router._get_net_id(net_name)
            resource_edges = self._router._net_to_edges.get(net_name)
            if resource_edges is None:
                graph_path = self._router._path_without_dynamic_escape_chains(
                    net_name, path
                )
                # Exact physical-conflict measurement needs each traversal
                # once; _net_to_edges contains both directional CSR arcs.
                net_edges = self._router._path_to_directed_edges(graph_path)
                edge_chunk = np.asarray(net_edges, dtype=np.int32)
            else:
                edge_chunk = np.asarray(
                    resource_edges, dtype=np.int32
                )[:len(resource_edges) // 2]
            if edge_chunk.size == 0:
                continue
            edge_chunks.append(edge_chunk)
            net_id_chunks.append(
                np.full(edge_chunk.size, net_id, dtype=np.int32)
            )

        if not edge_chunks:
            logger.info("[BARREL-CONFLICT] No edges found in paths")
            return np.array([], dtype=np.int32), 0

        edge_indices = np.concatenate(edge_chunks)
        edge_net_ids = np.concatenate(net_id_chunks)
        logger.info(f"[BARREL-CONFLICT] Checking {len(edge_indices)} edges across {len(paths_dict)} nets")

        # SimpleDijkstra already owns the CPU CSR arrays. Reuse them instead
        # of downloading the full destination array from the GPU per audit.
        graph_indices_cpu = self._router.solver.indices

        # VECTORIZED CONFLICT DETECTION (GPU-accelerated!)
        # Get source and destination nodes for all edges at once
        src_nodes = self._router._edge_src[edge_indices]  # Vectorized lookup
        dst_nodes = graph_indices_cpu[edge_indices]  # Vectorized lookup

        # Get ownership for all nodes at once
        src_owners = self._router.node_owner[src_nodes]  # Vectorized lookup
        dst_owners = self._router.node_owner[dst_nodes]  # Vectorized lookup

        # Vectorized conflict check:
        # Conflict if src owned by different net OR dst owned by different net
        src_conflict = (src_owners != -1) & (src_owners != edge_net_ids)
        dst_conflict = (dst_owners != -1) & (dst_owners != edge_net_ids)
        conflict_mask = src_conflict | dst_conflict  # Element-wise OR

        logger.info(f"[BARREL-CONFLICT] Vectorized check of {len(edge_indices)} edges completed")

        # Get the actual edge indices that have conflicts
        conflict_edge_indices = edge_indices[conflict_mask]
        conflict_count = len(conflict_edge_indices)
        self._router._last_exact_barrel_conflict_count = conflict_count

        if conflict_count > 0:
            conflict_positions = np.flatnonzero(conflict_mask)
            victim_net_ids = set(
                map(int, edge_net_ids[conflict_positions])
            )
            owner_net_ids = set()
            conflict_node_chunks = []
            if np.any(src_conflict):
                conflict_node_chunks.append(
                    src_nodes[np.flatnonzero(src_conflict)]
                )
            if np.any(dst_conflict):
                conflict_node_chunks.append(
                    dst_nodes[np.flatnonzero(dst_conflict)]
                )
            conflict_nodes = np.unique(np.concatenate(
                conflict_node_chunks
            ))
            self._router._accumulate_node_conflict_history(conflict_nodes)
            for node_idx in conflict_nodes:
                owner_net_ids.update(
                    self._router._node_owner_members.get(int(node_idx), ())
                )
            id_to_name = {
                numeric_id: name
                for name, numeric_id in self._router.net_id_map.items()
            }
            exact_pairs = set()
            for position in conflict_positions:
                victim_id = int(edge_net_ids[position])
                conflicting_nodes = []
                if src_conflict[position]:
                    conflicting_nodes.append(int(src_nodes[position]))
                if dst_conflict[position]:
                    conflicting_nodes.append(int(dst_nodes[position]))
                for node_idx in conflicting_nodes:
                    for owner_id in self._router._node_owner_members.get(
                        node_idx, ()
                    ):
                        if owner_id == victim_id:
                            continue
                        victim_name = id_to_name.get(victim_id)
                        owner_name = id_to_name.get(owner_id)
                        if (
                            victim_name is not None
                            and owner_name is not None
                        ):
                            self._router._physical_conflict_scores[
                                victim_name
                            ] += 1
                            self._router._physical_conflict_scores[
                                owner_name
                            ] += 1
                            exact_pairs.add(tuple(sorted((
                                victim_name, owner_name
                            ))))
            self._router._exact_barrel_pairs = exact_pairs
            for position in conflict_positions[:20]:
                src_node = int(src_nodes[position])
                dst_node = int(dst_nodes[position])
                self._router._last_exact_barrel_details.append({
                    "victim": id_to_name.get(
                        int(edge_net_ids[position]),
                        str(int(edge_net_ids[position])),
                    ),
                    "edge": int(edge_indices[position]),
                    "src": src_node,
                    "dst": dst_node,
                    "src_owners": tuple(sorted(
                        id_to_name.get(owner, str(owner))
                        for owner in self._router._node_owner_members.get(
                            src_node, ()
                        )
                    )),
                    "dst_owners": tuple(sorted(
                        id_to_name.get(owner, str(owner))
                        for owner in self._router._node_owner_members.get(
                            dst_node, ()
                        )
                    )),
                })
            self._router._barrel_victim_nets = {
                id_to_name[numeric_id]
                for numeric_id in victim_net_ids
                if numeric_id in id_to_name
            }
            self._router._barrel_owner_nets = {
                id_to_name[numeric_id]
                for numeric_id in owner_net_ids
                if numeric_id in id_to_name
            }
            conflict_xy = {
                self._router.lattice.idx_to_coord(int(node_idx))[:2]
                for node_idx in conflict_nodes
            }
            for owner_net in self._router._barrel_owner_nets:
                selected = self._router.net_selected_portals.get(owner_net, ())
                pad_ids = self._router.net_pad_ids.get(owner_net, ())
                layers = self._router.net_portal_layers.get(owner_net, ())
                for pad_id, portal, entry_layer in zip(
                    pad_ids, selected, layers
                ):
                    if (
                        portal is not None
                        and (portal.x_idx, portal.y_idx) in conflict_xy
                    ):
                        self._router._barrel_owner_portal_keys.add((
                            pad_id,
                            portal.x_idx,
                            portal.y_idx,
                            int(entry_layer),
                        ))
            self._router._barrel_conflict_nets = {
                id_to_name[numeric_id]
                for numeric_id in (victim_net_ids | owner_net_ids)
                if numeric_id in id_to_name
            }
            logger.info(f"[BARREL-CONFLICT] Detected {conflict_count} conflicts (checked {len(edge_indices)} edges)")
        else:
            logger.info(f"[BARREL-CONFLICT] No conflicts found (checked {len(edge_indices)} edges)")

        (
            path_node_pairs,
            shared_path_nodes,
            path_node_scores,
        ) = self._router._detect_path_node_conflicts()
        self._router._path_node_conflict_pairs = set(path_node_pairs)
        self._router._path_node_conflict_scores = dict(path_node_scores)
        self._router._last_path_node_conflict_count = len(shared_path_nodes)
        if shared_path_nodes:
            for net_name, score in path_node_scores.items():
                self._router._physical_conflict_scores[net_name] += int(score)
            involved_nets = {
                net_name
                for pair in path_node_pairs
                for net_name in pair
            }
            self._router._barrel_conflict_nets.update(involved_nets)
            self._router._accumulate_node_conflict_history(shared_path_nodes)
            logger.info(
                "[PATH-NODE-CONFLICT] Detected %d shared nodes "
                "across %d net pairs",
                len(shared_path_nodes),
                len(path_node_pairs),
            )

        # Escape stubs and portal via bodies are off-lattice physical
        # geometry, so edge accounting cannot see their conflicts. Audit the
        # selected candidates with the same dimensions used for emission and
        # feed offenders into the normal negotiated hotset.
        self._router._rebuild_escape_occupancy()
        escape_pairs, escape_owners, escape_victims = (
            self._router._detect_escape_conflicts()
        )
        self._router._last_escape_conflict_count = len(escape_pairs)
        self._router._escape_conflict_pairs = set(escape_pairs)
        if escape_pairs:
            for first, second in escape_pairs:
                self._router._physical_conflict_scores[first[0]] += 1
                self._router._physical_conflict_scores[second[0]] += 1
            # Escape stubs live outside edge accounting, so their selected
            # candidates need their own Pathfinder history. Penalize both
            # ends of every physical conflict; otherwise these conflicts can
            # reroute forever without making either alternative more costly.
            self._router._barrel_owner_portal_keys.update(
                self._router._escape_conflict_portal_keys(escape_pairs)
            )
            self._router._barrel_owner_nets.update(escape_owners)
            self._router._barrel_victim_nets.update(escape_victims)
            self._router._barrel_conflict_nets.update(
                escape_owners | escape_victims
            )
            logger.info(
                "[ESCAPE-CONFLICT] Detected %d selected escape conflicts",
                len(escape_pairs),
            )

        (
            portal_grid_pairs,
            portal_grid_owners,
            portal_grid_victims,
            portal_grid_keys,
            portal_grid_nodes,
        ) = self._router._detect_portal_grid_conflicts()
        self._router._last_portal_grid_conflict_count = len(
            portal_grid_pairs
        )
        self._router._portal_grid_pairs = set(portal_grid_pairs)
        self._router._portal_grid_owner_nets = set(portal_grid_owners)
        self._router._portal_grid_victim_nets = set(portal_grid_victims)
        if portal_grid_pairs:
            for identity, victim, _kind in portal_grid_pairs:
                self._router._physical_conflict_scores[identity[0]] += 1
                self._router._physical_conflict_scores[victim] += 1
            self._router._barrel_owner_portal_keys.update(portal_grid_keys)
            self._router._barrel_owner_nets.update(portal_grid_owners)
            self._router._barrel_victim_nets.update(portal_grid_victims)
            self._router._barrel_conflict_nets.update(
                portal_grid_owners | portal_grid_victims
            )
            self._router._accumulate_node_conflict_history(
                portal_grid_nodes
            )
            logger.info(
                "[PORTAL-GRID-CONFLICT] Detected %d terminal-via "
                "conflicts (%d owners, %d victims)",
                len(portal_grid_pairs),
                len(portal_grid_owners),
                len(portal_grid_victims),
            )

        return (
            conflict_edge_indices,
            conflict_count
            + len(shared_path_nodes)
            + len(escape_pairs)
            + len(portal_grid_pairs),
        )
