"""
Negotiation Mixin - Extracted from UnifiedPathFinder

This module contains negotiation mixin functionality.
Part of the PathFinder routing algorithm refactoring.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

from types import SimpleNamespace

# Prefer local light interfaces; fall back to monorepo types if available
try:
    from ....domain.models.board import Board as BoardLike, Pad
except Exception:  # pragma: no cover - plugin environment
    from ..types import BoardLike, Pad

logger = logging.getLogger(__name__)


class NegotiationMixin:
    """
    Negotiation functionality for UnifiedPathFinder.

    This mixin is designed to be used with multiple inheritance.
    It expects the following attributes to be available on self:
    - config: PathFinderConfig instance
    - use_gpu: bool indicating GPU availability
    - node_count: int number of nodes
    - edges: list of edges
    - nodes: dict of node data
    """

    def route_multiple_nets(self, route_requests: List[Tuple[str, str, str]], progress_cb=None) -> Dict[str, List[int]]:
        """
        OPTIMIZED PathFinder with fast net parsing and GPU acceleration
        """
        logger.info(f"[UPF] instance=%s enter %s", getattr(self, "_instance_tag", "NO_TAG"), "route_multiple_nets")

        # Initialize failure tracking for adaptive corridor widening
        if not hasattr(self, 'net_fail_streak'):
            self.net_fail_streak = {}  # Dict[str, int] - consecutive failures per net

        # Normalize edge owner types to prevent crashes
        self._normalize_owner_types()

        # Progress-callback shim: prefer 3-arg GUI signature, fallback to 5 if needed
        def _pc(done, total, msg, paths=None, vias=None):
            """Progress-callback shim: prefer 3-arg GUI signature, fallback to 5 if needed."""
            if progress_cb is None:
                return
            try:
                # Try the common 3-arg GUI signature first
                return progress_cb(done, total, msg)
            except TypeError:
                # If the caller expects 5 args, use that
                return progress_cb(done, total, msg, paths or [], vias or [])

        # SURGICAL: Add hard guard at entry of route_multiple_nets
        if not hasattr(self, 'graph_state'):
            raise RuntimeError("[INIT] graph_state missing. Call initialize_graph(board) first.")

        # DEFENSIVE CHECK: Assert live sizes before routing
        self._assert_live_sizes()

        # CRITICAL FIX: Ensure adaptive delta is initialized
        self._ensure_delta()

        gs = self.graph_state

        # SURGICAL STEP 5: Small net limit for immediate copper verification
        import os
        net_limit = NET_LIMIT
        if net_limit > 0:
            logger.info(f"[NET-LIMIT] Limiting to first {net_limit} nets for testing (ORTHO_NET_LIMIT={net_limit})")
            route_requests = route_requests[:net_limit]

        # If caller already provided (name, src_idx, dst_idx) tuples (ints), skip adaptation
        if route_requests and isinstance(route_requests[0], (tuple, list)) \
           and len(route_requests[0]) == 3 and isinstance(route_requests[0][1], int) and isinstance(route_requests[0][2], int):
            logger.info("[NET-ADAPTER] input already adapted; skipping")
        # NET-ADAPTER: Handle List[Net] objects from GUI - map to lattice terminal indices
        elif route_requests and not isinstance(route_requests[0], (tuple, list)):
            logger.info("[NET-ADAPTER] Converting List[Net] objects to lattice terminal indices")

            # Single source for N and counters
            gs = getattr(self, 'graph_state', None)
            N = getattr(self, 'lattice_node_count', 0)
            if gs and not N:
                N = getattr(gs, 'lattice_node_count', 0)

            valid = missing = same = 0
            adapted_requests = []

            for net_obj in route_requests:
                net_name = getattr(net_obj, "name", getattr(net_obj, "id", str(net_obj)))
                pads = getattr(net_obj, "pads", None) or getattr(net_obj, "terminals", None)

                if not pads or len(pads) < 2:
                    logger.warning(f"[NET-ADAPTER] Net {net_name}: not enough terminals ({len(pads) if pads else 0})")
                    missing += 1
                    continue

                # Choose two different pads (prefer different components)
                src_pad, dst_pad = self._choose_two_pads_for_net(net_obj)
                if src_pad is None or dst_pad is None:
                    logger.warning(f"[NET-ADAPTER] Net {net_name}: insufficient pads for routing")
                    missing += 1
                    continue

                # Sanity check: ensure we didn't pick the same pad object
                if id(src_pad) == id(dst_pad):
                    logger.warning(f"[TERMINALS] net={net_name} selected the same pad object; trying alternate approach")
                    # Try to find any two different pads
                    if len(pads) >= 2:
                        for i, p1 in enumerate(pads):
                            for j, p2 in enumerate(pads[i+1:], i+1):
                                if id(p1) != id(p2):
                                    src_pad, dst_pad = p1, p2
                                    break
                            else:
                                continue
                            break

                    if id(src_pad) == id(dst_pad):
                        logger.error(f"[TERMINALS] net={net_name} cannot find two different pad objects")
                        same += 1
                        continue

                # Resolve pads to portal nodes: object identity first, UID second
                src_idx = self._portal_by_pad_id.get(id(src_pad))
                dst_idx = self._portal_by_pad_id.get(id(dst_pad))

                # Fallback to UID-based lookup (use same helpers as registration)
                src_ref = src_lbl = dst_ref = dst_lbl = None
                if src_idx is None or dst_idx is None:
                    src_comp = getattr(src_pad, "component", getattr(src_pad, "footprint", None))
                    dst_comp = getattr(dst_pad, "component", getattr(dst_pad, "footprint", None))
                    src_ref  = self._uid_component(src_comp)
                    dst_ref  = self._uid_component(dst_comp)
                    src_lbl  = self._uid_pad_label(src_pad, src_ref)
                    dst_lbl  = self._uid_pad_label(dst_pad, dst_ref)

                    if src_idx is None:
                        src_idx = self._portal_by_uid.get((src_ref, src_lbl))
                    if dst_idx is None:
                        dst_idx = self._portal_by_uid.get((dst_ref, dst_lbl))

                # Strong type invariant: node indices must be ints
                if not (isinstance(src_idx, int) and isinstance(dst_idx, int)):
                    logger.error(f"[TERMINALS] bad types for {net_name}: src={src_idx!r} ({type(src_idx)}), dst={dst_idx!r} ({type(dst_idx)})")
                    missing += 1
                    continue

                # Check for missing portals
                if src_idx is None or dst_idx is None:
                    if src_ref is None:  # Didn't compute UIDs above
                        src_comp = getattr(src_pad, "component", getattr(src_pad, "footprint", None))
                        dst_comp = getattr(dst_pad, "component", getattr(dst_pad, "footprint", None))
                        src_ref  = self._uid_component(src_comp)
                        dst_ref  = self._uid_component(dst_comp)
                        src_lbl  = self._uid_pad_label(src_pad, src_ref)
                        dst_lbl  = self._uid_pad_label(dst_pad, dst_ref)

                    logger.error(f"[TERMINALS] missing portal net={net_name} "
                               f"src_uid=({src_ref},{src_lbl}) dst_uid=({dst_ref},{dst_lbl})")
                    # Sample available keys for debugging
                    sample = [(r,l) for (r,l) in self._portal_by_uid.keys() if r in (src_ref, dst_ref)][:5]
                    if sample:
                        logger.debug(f"[TERMINALS] sample uids for comps {src_ref}/{dst_ref}: {sample}")
                    missing += 1
                    continue

                # Check for src==dst collapse
                if src_idx == dst_idx:
                    if src_ref is None:  # Didn't compute UIDs above
                        src_comp = getattr(src_pad, "component", getattr(src_pad, "footprint", None))
                        dst_comp = getattr(dst_pad, "component", getattr(dst_pad, "footprint", None))
                        src_ref  = self._uid_component(src_comp)
                        dst_ref  = self._uid_component(dst_comp)
                        src_lbl  = self._uid_pad_label(src_pad, src_ref)
                        dst_lbl  = self._uid_pad_label(dst_pad, dst_ref)

                    logger.warning(f"[TERMINALS] same-node pair net={net_name} src={src_idx} dst={dst_idx} "
                                 f"uid_src=({src_ref},{src_lbl}) uid_dst=({dst_ref},{dst_lbl})")
                    same += 1
                    continue

                # Range check and success
                if src_idx == dst_idx:
                    same += 1
                    continue

                if not (0 <= src_idx < N and 0 <= dst_idx < N):
                    logger.error(f"[TERMINALS] out of range: net={net_name} src={src_idx} dst={dst_idx} N={N}")
                    missing += 1
                    continue

                # Success: add to routing requests
                adapted_requests.append((str(net_name), int(src_idx), int(dst_idx)))
                logger.info(f"[TERMINALS] net={net_name} src={src_idx} dst={dst_idx}")
                valid += 1

            route_requests = adapted_requests
            logger.info(f"[NET-PARSE] Results: {valid} valid, {missing} missing nodes, {same} same-node pairs")

        # TRIPWIRE A: Log what we will actually route (only after adapter runs)
        if route_requests and isinstance(route_requests[0], (tuple, list)):
            logger.info(f"[NET-ADAPTER] incoming={len(route_requests)}")
            logger.info(f"[NET-ADAPTER] adapted={len(route_requests)}")
            for nid, s, t in route_requests[:5]:
                logger.info(f"[TERMINALS] net={nid} src={s} dst={t}")
        else:
            logger.error(f"[NET-ADAPTER] CRITICAL: route_requests still contains non-tuple objects: {type(route_requests[0]) if route_requests else 'empty'}")
            logger.error(f"[NET-ADAPTER] This indicates the adapter logic failed to run properly")
            raise TypeError(f"route_requests contains {type(route_requests[0])} instead of tuples - adapter logic failed")

        logger.info(f"Unified PathFinder: routing {len(route_requests)} nets")
        start_time = time.time()

        # Defense line: ensure route_requests are in correct format
        assert route_requests and isinstance(route_requests[0][1], int) and isinstance(route_requests[0][2], int), \
            "[ADAPTER] invariant failed: route_requests not in (name,int,int) form"

        # OPTIMIZED net parsing with O(1) lookups
        valid_nets = self._parse_nets_fast(route_requests, already_adapted=True)
        if not valid_nets:
            logger.warning("[NET-PARSE] No valid nets found - PathFinder negotiation cannot run")
            # SURGICAL FIX: Set negotiation flag to prove PathFinder setup worked even with no valid nets
            self._negotiation_ran = True
            logger.info("[SURGICAL] _negotiation_ran=True set despite no valid nets (proves PathFinder setup)")
            return {}

        parse_time = time.time() - start_time
        logger.info(f"Net parsing: {len(valid_nets)} nets in {parse_time:.2f}s")

        total = len(valid_nets)

        # Progress update after parsing (3-arg form)
        if progress_cb:
            _pc(0, total, 0.0)

        # TRIPWIRE B: Verify terminals are reachable before PF
        self._assert_terminals_reachable(valid_nets)

        # Belt-and-suspenders tripwire: ensure no out-of-range nodes after adaptation
        if any(not (0 <= s < self.lattice_node_count and 0 <= d < self.lattice_node_count) for _, (s, d) in valid_nets.items()):
            raise AssertionError("[NET-PARSE] found out-of-range node after adaptation (double-parse?)")

        # PathFinder negotiation with congestion
        result = self._pathfinder_negotiation(valid_nets, _pc, total)
        self._routing_result = result

        # Failure: return the structured result; don't count keys like "paths"
        if isinstance(result, dict) and not result.get("success", True):
            logger.warning(f"[PF-RETURN] failed: {result.get('message','routing failed')}")
            return result

        # Success: count actual net paths
        npaths = sum(
            1 for p in getattr(self, "_net_paths", {}).values()
            if p is not None and (len(p) if hasattr(p, "__len__") else 0) > 1
        )
        logger.info(f"[PF-RETURN] paths={npaths}")
        self._committed_paths = result  # Single source of truth

        # Portal usage fingerprint at end of routing
        logger.info("[PORTAL-FINAL] Portal system final status: edges_registered=%d escapes_used=%d",
                   self._metrics.get("portal_edges_registered", 0),
                   self._metrics.get("portal_escapes_used", 0))

        # Final progress update (5-arg form)
        if progress_cb:
            _pc(total, total, "Routing complete")

        return result
    

    def _pathfinder_negotiation(self, valid_nets: Dict[str, Tuple[int, int]], progress_cb=None, total=0) -> Dict[str, List[int]]:
        """PathFinder negotiation loop with proper 4-phase iteration: refresh → cost update → route → commit"""
        cfg = self.config
        pres_fac = cfg.pres_fac_init
        best_unrouted = None
        stagnant_iters = 0

        # Clear per-iter present, but keep STORE (history of the current round)
        self._reset_present_usage()               # present = 0
        # Note: DO NOT clear store usage - it persists between iterations

        # Mark that negotiation is running
        self._negotiation_ran = True
        logger.info(f"[NEGOTIATE] start: iters={cfg.max_iterations} pres={pres_fac:.2f}×{cfg.pres_fac_mult:.2f}")

        self.routed_nets.clear()
        total_nets = len(valid_nets)

        # Build edge→src mapping once for barrel conflict detection (GPU-accelerated)
        print("\n" + "="*80)
        print("[NEGOTIATION-START] Building edge_src_map for barrel conflict detection")
        print("="*80 + "\n")
        logger.info("="*80)
        logger.info("[NEGOTIATION-START] Building edge_src_map for barrel conflict detection")
        logger.info("="*80)
        if hasattr(self, '_ensure_edge_src_map'):
            self._ensure_edge_src_map()
        else:
            logger.error("[NEGOTIATION-START] ERROR: _ensure_edge_src_map method NOT FOUND!")
            print("[ERROR] _ensure_edge_src_map method NOT FOUND!")

        for it in range(1, cfg.max_iterations + 1):
            logger.info("[NEGOTIATE] iter=%d pres_fac=%.2f", it, pres_fac)
            self.current_iteration = it

            # Log iteration 1 always-connect policy
            if it == 1 and cfg.iter1_always_connect:
                logger.info("[ITER-1-POLICY] Always-connect mode: soft costs only (no hard blocks)")

            # Track path changes for stagnation detection
            import numpy as np
            old_paths = {
                net_id: (np.asarray(path, dtype=np.int64).copy()
                         if path is not None else np.empty(0, dtype=np.int64))
                for net_id, path in self._net_paths.items()
            }

            # 1) Pull last iteration's result into PRESENT
            mapped = self._refresh_present_usage_from_store()    # logs how many entries mapped
            self._check_overuse_invariant("iter-start", compare_to_store=True)

            # Sanity check after refresh
            logger.info("[SANITY] iter=%d store_edges=%d present_nonzero=%d",
                        self.current_iteration,
                        len(self._edge_store),
                        int((self.edge_present_usage > 0).sum()))

            # 2) Compute overuse on PRESENT and update costs
            over_sum, over_edges = self._compute_overuse_stats_present()  # must not raise

            # 2b) CRITICAL: Detect via barrel conflicts (GPU-accelerated)
            # This is THE fix for shorting_items! Detects when edges touch via barrels owned by other nets.
            if hasattr(self, '_detect_barrel_conflicts'):
                conflict_edge_indices, conflict_count = self._detect_barrel_conflicts()
                if conflict_count > 0:
                    logger.info(f"[BARREL-CONFLICT] Marking {conflict_count} conflicting edges as overused")
                    # Mark conflicting edges as overused in present usage array (CRITICAL!)
                    # This makes PathFinder see them and reroute them
                    import numpy as np
                    pres = self.edge_present_usage
                    if hasattr(pres, 'get'):
                        # GPU array - need to update on GPU
                        import cupy as cp
                        conflict_indices_gpu = cp.asarray(conflict_edge_indices)
                        pres[conflict_indices_gpu] += 10.0  # Barrel conflict penalty
                    else:
                        # CPU array
                        pres[conflict_edge_indices] += 10.0  # Barrel conflict penalty

                    # Add to overuse stats
                    over_sum += conflict_count
                    over_edges += conflict_count
                    logger.info(f"[BARREL-CONFLICT] Added {conflict_count} barrel conflicts to overuse (new total: {over_sum})")

            self._update_edge_total_costs(pres_fac)

            # 3) Route all nets against current costs (must not throw on single-net failure)
            routed_ct, failed_ct = self._route_all_nets_cpu_in_batches_with_metrics(valid_nets, progress_cb)

            # Calculate how many nets changed paths this iteration
            def _as_array_path(p):
                if p is None:
                    return np.empty(0, dtype=np.int64)
                # already an array?
                if isinstance(p, np.ndarray):
                    return p.astype(np.int64, copy=True)
                # list/tuple
                return np.asarray(p, dtype=np.int64).copy()

            routes_changed = 0
            for net_id in valid_nets:
                old_path = old_paths.get(net_id, np.empty(0, dtype=np.int64))
                new_path = _as_array_path(self._net_paths.get(net_id, []))
                if not np.array_equal(old_path, new_path):
                    routes_changed += 1

            logger.info("[ROUTES-CHANGED] %d nets changed this iter", routes_changed)

            # 4) Commit PRESENT → STORE so next iter sees it
            changed = self._commit_present_usage_to_store()

            # Sanity check after commit
            logger.info("[SANITY] iter=%d store_edges=%d present_nonzero=%d",
                        self.current_iteration,
                        len(self._edge_store),
                        int((self.edge_present_usage > 0).sum()))

            # Note: Buried-via keepouts are now enforced per-net during ROI extraction
            # (owner-aware filtering in roi_extractor_mixin.py), not by globally modifying graph weights

            logger.info("[ITER-RESULT] routed=%d failed=%d overuse_edges=%d over_sum=%d changed=%s",
                        routed_ct, failed_ct, over_edges, over_sum, bool(changed))

            # ===== COLUMN BALANCE METRICS (Fix #9) =====
            # Track vertical traffic distribution across columns to monitor "elevator shaft" problem
            self._log_column_balance_metrics(it)

            # Log top congested nets for diagnostic purposes
            if over_edges > 0:
                self._log_top_congested_nets(k=20)

            # ---- Termination logic ----
            # Success: no overuse and no failures
            if failed_ct == 0 and over_edges == 0:
                logger.info("[NEGOTIATE] Converged: all nets routed with legal usage.")
                result = self._finalize_success()
                return result

            # Track "no progress" to avoid spinning forever
            cur_unrouted = failed_ct + (1 if over_edges > 0 else 0)
            if best_unrouted is None or cur_unrouted < best_unrouted:
                best_unrouted = cur_unrouted
                stagnant_iters = 0
            else:
                stagnant_iters += 1

            # Optional early stop on stagnation
            if stagnant_iters >= cfg.stagnation_patience:
                logger.warning("[NEGOTIATE] Stagnated for %d iters (best_unrouted=%d).",
                               stagnant_iters, best_unrouted)
                break

            # Increase present-cost pressure and loop
            pres_fac *= cfg.pres_fac_mult

        # Fell out of loop: decide the message
        result = self._finalize_insufficient_layers()
        self._routing_result = result  # Store for GUI emission check
        return result


    def _parse_nets_fast(self, route_requests: List[Tuple[str, str, str]], already_adapted=False) -> Dict[str, Tuple[int, int]]:
        """OPTIMIZED O(1) net parsing using pre-built lookups"""

        # Fast path: if we already have (name, src_idx, dst_idx) with ints, just build the dict
        if already_adapted or (
            route_requests and isinstance(route_requests[0], (tuple, list))
            and len(route_requests[0]) == 3 and isinstance(route_requests[0][1], int) and isinstance(route_requests[0][2], int)
        ):
            nets_dict = {str(name): (int(src), int(dst)) for name, src, dst in route_requests}
            logger.info(f"[NET-PARSE] Results: {len(nets_dict)} valid, 0 missing nodes, 0 same-node pairs")
            return nets_dict

        valid_nets = {}

        logger.debug(f"[NET-PARSE] Processing {len(route_requests)} route requests")
        logger.debug(f"[NET-PARSE] Node lookup has {len(self._node_lookup)} entries")

        missing_nodes = 0
        same_node_pairs = 0

        for net_id, source_node_id, sink_node_id in route_requests:
            # O(1) lookup instead of O(n) search
            if source_node_id in self._node_lookup and sink_node_id in self._node_lookup:
                source_idx = self._node_lookup[source_node_id]
                sink_idx = self._node_lookup[sink_node_id]

                if source_idx != sink_idx:
                    valid_nets[net_id] = (source_idx, sink_idx)
                else:
                    same_node_pairs += 1
            else:
                missing_nodes += 1
                if missing_nodes <= 3:  # Only log first few to avoid spam
                    logger.debug(f"[NET-PARSE] Missing node: src={source_node_id} sink={sink_node_id}")

        logger.info(f"[NET-PARSE] Results: {len(valid_nets)} valid, {missing_nodes} missing nodes, {same_node_pairs} same-node pairs")
        return valid_nets


    def _route_all_nets_cpu_in_batches_with_metrics(self, nets, progress_ctx):
        routed_ct = 0
        failed_ct = 0
        import numpy as np

        for batch in self._batched(list(nets.items()), self.config.batch_size):
            results, metrics = self._route_batch_cpu_with_metrics(batch, progress_ctx)
            for res in results:
                if res.success:
                    routed_ct += 1
                    csr_idx = res.csr_edge_indices
                    if csr_idx is None:
                        # fallback: build indices from path nodes
                        # (u,v) → edge_idx via self.edge_lookup
                        csr_idx = self._edge_indices_from_node_path(res.node_path)
                    # PRESENT += 1 at these edges
                    if isinstance(csr_idx, np.ndarray):
                        np.add.at(self.edge_present_usage, csr_idx, 1)
                    else:
                        for e in csr_idx:
                            self.edge_present_usage[int(e)] += 1
                else:
                    failed_ct += 1
        logger.info("[BATCH] routed=%d failed=%d", routed_ct, failed_ct)
        return routed_ct, failed_ct


    def _route_batch_cpu_with_metrics(self, batch, progress_cb):
        """Route a batch of nets with rip-up/search/restore-on-fail logic."""
        results = []
        metrics = {}
        routed_ct = 0
        failed_ct = 0

        import numpy as np

        for net_id, (src, dst) in batch:
            prev_idx = self._prepare_net_for_reroute(net_id)

            # Apply round-robin layer bias for first iterations (when symmetry matters most)
            # DISABLED AGAIN: Still hitting slow CPU fallback (7-8s per net)
            # TODO: Implement as kernel-side bias (expert suggestion A) for true speed
            if False and self.current_iteration <= 3:
                try:
                    self._apply_roundrobin_layer_bias_fast(net_id, src)
                except Exception as e:
                    logger.warning(f"[ROUNDROBIN] Failed to apply bias for {net_id}: {e}")

            res = self._route_single_net_cpu(net_id, src, dst)  # your existing call
            if res.success:
                csr_idx = res.csr_edge_indices or self._edge_indices_from_node_path(res.node_path)
                csr_idx = np.asarray(csr_idx, dtype=np.int64)
                np.add.at(self.edge_present_usage, csr_idx, 1)
                self._net_paths[net_id] = csr_idx
                if hasattr(self, "edge_owners"):
                    for e in csr_idx.tolist():
                        self.edge_owners[e] = net_id
                routed_ct += 1

                # Update failure tracking: success resets streak
                self._update_failure_tracking(net_id, success=True)

                result = type('RouteResult', (), {
                    'success': True,
                    'net_id': net_id,
                    'csr_edge_indices': csr_idx
                })()
            else:
                # failed: put the old path back so we don't lose capacity accounting
                self._restore_net_after_failed_reroute(net_id, prev_idx)
                failed_ct += 1

                # Update failure tracking: failure increments streak
                self._update_failure_tracking(net_id, success=False)

                result = type('RouteResult', (), {
                    'success': False,
                    'net_id': net_id,
                    'csr_edge_indices': None
                })()

            results.append(result)

        return results, metrics


    def _route_single_net_cpu(self, net_id, src, dst):
        """Route a single net and return result with success info."""
        try:
            path = self._cpu_dijkstra_fallback(src, dst)
            if path and len(path) > 1:
                csr_edges = self._path_nodes_to_csr_edges(path)
                return type('RouteResult', (), {
                    'success': True,
                    'node_path': path,
                    'csr_edge_indices': csr_edges
                })()
            else:
                return type('RouteResult', (), {
                    'success': False,
                    'node_path': [],
                    'csr_edge_indices': None
                })()
        except Exception:
            return type('RouteResult', (), {
                'success': False,
                'node_path': [],
                'csr_edge_indices': None
            })()


    def _update_failure_tracking(self, net_id: str, success: bool) -> None:
        """Update per-net failure tracking for adaptive corridor widening (Fix #7).

        Args:
            net_id: Network identifier
            success: True if routing succeeded, False if failed
        """
        if not hasattr(self, 'net_fail_streak'):
            from collections import defaultdict
            self.net_fail_streak = defaultdict(int)

        if success:
            # Success resets the failure streak
            self.net_fail_streak[net_id] = 0
        else:
            # Failure increments the streak
            self.net_fail_streak[net_id] = self.net_fail_streak.get(net_id, 0) + 1
            logger.debug(f"[FAIL-TRACK] {net_id}: failure streak now {self.net_fail_streak[net_id]}")


    def _update_edge_total_costs(self, pres_fac: float) -> None:
        """Update total edge costs for PathFinder negotiation using present cost factor.

        Args:
            pres_fac: Present cost factor for penalizing overused edges
        """
        import numpy as np
        usage = self.edge_present_usage
        cap   = self.edge_capacity
        hist  = self.edge_history
        base  = self.edge_base_cost

        # ITER-1 FIX: Ignore direction mask entirely in iteration 1
        # The lattice enforces H/V discipline at graph construction (asserts 0 wrong-direction edges)
        # So we can't relax it with soft costs - the edges don't exist!
        # In iter-1, treat ALL edges as legal for maximum connectivity
        if self.config.iter1_always_connect and self.current_iteration == 1:
            legal = np.ones_like(base, dtype=bool)  # All edges legal in iter-1
            logger.debug(f"[ITER-1-DISCIPLINE] Ignoring direction mask (all {legal.sum()} edges treated as legal)")
        else:
            legal = getattr(self, "edge_dir_mask", None)
            # normalize to a NumPy bool array (CPU or GPU)
            if legal is None:
                legal = np.ones_like(base, dtype=bool)
            else:
                if hasattr(legal, "get"):  # CuPy → NumPy
                    legal = legal.get()
                legal = legal.astype(bool, copy=False)

        # Ensure numpy, not device arrays
        if hasattr(usage, "get"): usage = usage.get()
        if hasattr(cap,   "get"): cap   = cap.get()
        if hasattr(hist,  "get"): hist  = hist.get()
        if hasattr(base,  "get"): base  = base.get()

        over  = usage.astype(np.float32) - cap.astype(np.float32)
        over[over < 0.0] = 0.0

        # Calculate total cost: base + present_penalty + historical
        total = base + pres_fac * usage + self.config.hist_cost_weight * hist

        # FIX #6: Column Occupancy Soft-Cap
        # Apply soft-cap multiplier to vertical edges based on column usage
        column_softcap_applied = self._apply_column_occupancy_softcap(total)

        # PATHFINDER ITERATION 1 POLICY: Always-connect (soft costs only)
        # Iteration 1 should route ~100% of nets using geometric paths
        # Hard blocks (infinity costs) should only apply in iterations 2+
        if self.config.iter1_always_connect and self.current_iteration == 1:
            # Iteration 1: Use high soft penalties instead of hard blocks
            # This allows ANY geometric path to be found
            illegal_penalty = 1000.0  # High cost but not infinite

            # Soft-block illegal edges (e.g., wrong-direction edges)
            total[~legal] *= illegal_penalty

            # Soft-block overused edges (capacity violations)
            over_mask = usage > cap
            total[over_mask] *= illegal_penalty

            # INSTRUMENTATION: Count hard walls (must be 0 in iter-1)
            inf_count = int(np.isinf(total).sum())
            if not hasattr(self, '_iter1_inf_writes'):
                self._iter1_inf_writes = 0
            self._iter1_inf_writes += inf_count
            if inf_count > 0:
                logger.warning(f"[ITER-1-HARDWALLS] BUG: Found {inf_count} infinite costs in iteration 1!")
        else:
            # Iterations 2+: Use hard blocks (infinity costs) to enforce constraints
            # Hard-block illegal edges
            total[~legal] = np.inf

            # Strict DRC: also block explicit overuse immediately (only in HARD phase)
            if self.current_iteration >= self.config.phase_block_after and self.config.strict_overuse_block:
                over_mask = usage > cap
                total[over_mask] = np.inf

        self.edge_total_cost = total

        # 🚫 Do NOT mutate ownership/present usage here by default
        if getattr(self.config, "peel_in_cost", False):
            # If this flag is enabled, peeling logic would go here
            # Currently disabled to prevent cost update side effects
            pass


    def _apply_column_occupancy_softcap(self, total: 'np.ndarray') -> int:
        """Apply soft-cap penalty to vertical edges based on (layer, x) column usage.

        This prevents "elevator shaft" congestion by adding immediate pricing feedback
        when multiple nets share the same vertical column before hard overuse occurs.

        Args:
            total: Edge total cost array to modify in-place

        Returns:
            Number of vertical edges with soft-cap applied
        """
        import numpy as np

        # Count nets using each (layer, x) column for vertical edges
        col_use = {}  # (layer, x) -> count of nets using this column

        # Iterate over all nets with committed paths
        for net_id, edge_indices in self._net_paths.items():
            if edge_indices is None or len(edge_indices) == 0:
                continue

            # Convert to numpy array if needed
            if not isinstance(edge_indices, np.ndarray):
                edge_indices = np.asarray(edge_indices, dtype=np.int64)

            # Track which columns this net uses (to count once per net per column)
            net_columns = set()

            # Check each edge in this net's path
            for edge_idx in edge_indices:
                # Get source and destination nodes for this edge
                # Use CSR structure: indptr and indices
                if not hasattr(self, 'indptr_cpu') or not hasattr(self, 'indices_cpu'):
                    continue

                # Find which node this edge belongs to by searching indptr
                u = 0
                for node_idx in range(len(self.indptr_cpu) - 1):
                    if self.indptr_cpu[node_idx] <= edge_idx < self.indptr_cpu[node_idx + 1]:
                        u = node_idx
                        break

                # Get destination node
                v = self.indices_cpu[edge_idx]

                # Get coordinates for both nodes
                u_coord = self._idx_to_coord(u)
                v_coord = self._idx_to_coord(v)

                if u_coord is None or v_coord is None:
                    continue

                u_x, u_y, u_layer = u_coord
                v_x, v_y, v_layer = v_coord

                # Check if this is a vertical edge (same x, different y, same layer)
                if u_x == v_x and u_y != v_y and u_layer == v_layer:
                    col_key = (u_layer, u_x)
                    net_columns.add(col_key)

            # Count this net for each column it uses
            for col_key in net_columns:
                col_use[col_key] = col_use.get(col_key, 0) + 1

        # Apply soft-cap multiplier to vertical edges
        beta = self.config.column_present_beta
        vertical_edges_modified = 0
        max_col_usage = 0

        # Iterate through all edges and apply soft-cap to vertical ones
        for node_idx in range(len(self.indptr_cpu) - 1):
            start_idx = self.indptr_cpu[node_idx]
            end_idx = self.indptr_cpu[node_idx + 1]

            for edge_idx in range(start_idx, end_idx):
                v = self.indices_cpu[edge_idx]

                # Get coordinates
                u_coord = self._idx_to_coord(node_idx)
                v_coord = self._idx_to_coord(v)

                if u_coord is None or v_coord is None:
                    continue

                u_x, u_y, u_layer = u_coord
                v_x, v_y, v_layer = v_coord

                # Check if vertical edge
                if u_x == v_x and u_y != v_y and u_layer == v_layer:
                    col_key = (u_layer, u_x)
                    col_usage = col_use.get(col_key, 0)
                    max_col_usage = max(max_col_usage, col_usage)

                    # Apply soft-cap: multiply cost by (1 + beta * max(0, count - 1))
                    if col_usage > 1:
                        col_pres = 1.0 + beta * (col_usage - 1)
                        total[edge_idx] *= col_pres
                        vertical_edges_modified += 1

        # Log results
        if vertical_edges_modified > 0:
            logger.info(f"[COLUMN-SOFTCAP] Applied to {vertical_edges_modified} vertical edges, max_col_usage={max_col_usage}")

        return vertical_edges_modified


    def _compute_overuse_stats(self) -> tuple[int, int]:
        """Compute overuse statistics from present usage arrays"""
        import numpy as np
        usage = self.edge_present_usage
        cap   = self.edge_capacity

        if hasattr(usage, "get"): usage = usage.get()
        if hasattr(cap, "get"):   cap   = cap.get()

        if usage is None or cap is None:
            return 0, 0

        over = usage.astype(np.float32) - cap.astype(np.float32)
        over[over < 0.0] = 0.0
        return int(over.sum()), int((over > 0.0).sum())


    def _update_edge_history_gpu(self):
        """Update historical congestion on device"""
        if self.use_gpu:
            # Vectorized update on GPU
            overuse = cp.maximum(self.edge_present_usage - self.edge_capacity, 0.0)
            self.edge_history += overuse * 0.1  # Historical accumulation factor
        else:
            # CPU fallback
            overuse = np.maximum(self.edge_present_usage - self.edge_capacity, 0.0)
            self.edge_history += overuse * 0.1

        # FIX #4: Diffused History Cost on Vertical Columns (CRITICAL)
        # Make long-term congestion pressure BLEED sideways so a single column can't stay cheapest forever
        self._apply_column_spread_diffusion(overuse)


    def _apply_column_spread_diffusion(self, overuse: 'np.ndarray') -> None:
        """Apply diffused history cost to vertical columns to prevent "elevator shaft" congestion.

        VPR-style history cost is local to exact edges, which causes one "elevator shaft"
        to stay marginally cheapest while getting overused. Diffusing history across
        adjacent columns makes them attractive alternatives.

        Args:
            overuse: Overuse array for all edges (already computed)
        """
        import numpy as np
        from scipy.ndimage import convolve1d

        # Get config parameters
        alpha = self.config.column_spread_alpha  # 0.35 - fraction that diffuses sideways
        radius = self.config.column_spread_radius  # 2 - columns ±2 get blurred cost

        if alpha <= 0.0:
            return  # Feature disabled

        # Get lattice dimensions
        if not hasattr(self, 'lattice') or self.lattice is None:
            logger.debug("[COLUMN-SPREAD] No lattice available, skipping diffusion")
            return

        lattice = self.lattice
        num_layers = lattice.layers
        x_steps = lattice.x_steps
        y_steps = lattice.y_steps

        # Identify vertical routing layers (even-numbered internal layers)
        vertical_layers = [l for l in range(num_layers) if lattice.get_legal_axis(l) == 'v']

        if not vertical_layers:
            return

        # 1. Accumulate per-(layer, x) vertical overuse
        # Shape: (num_vertical_layers, x_steps)
        vert_overuse = np.zeros((len(vertical_layers), x_steps), dtype=np.float32)

        # Map layer index to position in vert_overuse array
        layer_to_idx = {layer: idx for idx, layer in enumerate(vertical_layers)}

        # Iterate through all edges and accumulate overuse for vertical edges
        if not hasattr(self, 'indptr_cpu') or not hasattr(self, 'indices_cpu'):
            logger.debug("[COLUMN-SPREAD] CSR arrays not available, skipping diffusion")
            return

        # Ensure overuse is on CPU
        if hasattr(overuse, 'get'):
            overuse = overuse.get()

        for node_idx in range(len(self.indptr_cpu) - 1):
            start_idx = self.indptr_cpu[node_idx]
            end_idx = self.indptr_cpu[node_idx + 1]

            for edge_idx in range(start_idx, end_idx):
                # Skip edges with no overuse
                if overuse[edge_idx] <= 0:
                    continue

                v = self.indices_cpu[edge_idx]

                # Get coordinates for both nodes
                u_coord = self._idx_to_coord(node_idx)
                v_coord = self._idx_to_coord(v)

                if u_coord is None or v_coord is None:
                    continue

                u_x, u_y, u_layer = u_coord
                v_x, v_y, v_layer = v_coord

                # Check if this is a vertical edge (same x, different y, same layer)
                if u_x == v_x and u_y != v_y and u_layer == v_layer:
                    # This is a vertical edge
                    if u_layer in layer_to_idx:
                        z_index = layer_to_idx[u_layer]
                        if 0 <= u_x < x_steps:
                            vert_overuse[z_index, u_x] += overuse[edge_idx]

        # 2. Convolve with Gaussian kernel to blur across columns
        kernel = np.exp(-np.abs(np.arange(-radius, radius + 1)))
        kernel = kernel / kernel.sum()

        # Apply 1D convolution along x-axis (axis=1) for each layer
        vert_overuse_blurred = np.zeros_like(vert_overuse)
        for z_idx in range(len(vertical_layers)):
            vert_overuse_blurred[z_idx, :] = convolve1d(
                vert_overuse[z_idx, :], kernel, axis=0, mode='nearest'
            )

        # 3. Add blurred cost to vertical edge history
        total_added = 0.0
        edges_updated = 0

        for node_idx in range(len(self.indptr_cpu) - 1):
            start_idx = self.indptr_cpu[node_idx]
            end_idx = self.indptr_cpu[node_idx + 1]

            for edge_idx in range(start_idx, end_idx):
                v = self.indices_cpu[edge_idx]

                # Get coordinates
                u_coord = self._idx_to_coord(node_idx)
                v_coord = self._idx_to_coord(v)

                if u_coord is None or v_coord is None:
                    continue

                u_x, u_y, u_layer = u_coord
                v_x, v_y, v_layer = v_coord

                # Check if vertical edge
                if u_x == v_x and u_y != v_y and u_layer == v_layer:
                    if u_layer in layer_to_idx:
                        z_index = layer_to_idx[u_layer]
                        if 0 <= u_x < x_steps:
                            # Add diffused cost
                            diffused_cost = alpha * vert_overuse_blurred[z_index, u_x]
                            self.edge_history[edge_idx] += diffused_cost
                            total_added += diffused_cost
                            edges_updated += 1

        # Logging
        if edges_updated > 0:
            mean_per_col = total_added / edges_updated
            logger.info(f"[COLUMN-SPREAD] Diffused history: total_added={total_added:.1f}, "
                       f"mean_per_col={mean_per_col:.3f}, edges_updated={edges_updated}")
        else:
            logger.debug("[COLUMN-SPREAD] No vertical edges to diffuse")


    def _compute_overuse_stats_present(self) -> tuple[int, int]:
        """Compute overuse statistics from present usage arrays (alias for _compute_overuse_stats)"""
        return self._compute_overuse_stats()


    def _compute_overuse_from_present(self) -> tuple[int, int]:
        """Compute overuse from present usage arrays (alias for _compute_overuse_stats)"""
        return self._compute_overuse_stats()


    def _compute_overuse_from_store(self) -> tuple[int, int]:
        """Compute overuse from edge store (alias for _compute_overuse_from_edge_store)"""
        return self._compute_overuse_from_edge_store()


    def _refresh_present_usage_from_store(self) -> int:
        """Rebuild present usage arrays from canonical edge store accounting data."""
        import numpy as np
        # Ensure arrays sized to live CSR
        E = self._live_edge_count() if hasattr(self, '_live_edge_count') else len(self.edge_present_usage)
        if getattr(self, 'edge_present_usage', None) is None or len(self.edge_present_usage) != E:
            self.edge_present_usage = np.zeros(E, dtype=np.float32)
        else:
            self.edge_present_usage.fill(0.0)

        store = getattr(self, "_edge_store", None) or getattr(self, 'edge_store', None) or {}
        mapped = 0
        for ei, usage in store.items():
            if isinstance(ei, int) and 0 <= ei < E and int(usage) > 0:
                self.edge_present_usage[ei] = float(usage)
                mapped += 1
        logger.debug("[UPF] Usage refresh: mapped %d store entries to present usage", mapped)
        return mapped


    def _commit_present_usage_to_store(self) -> bool:
        """Commit present usage to edge store."""
        import numpy as np
        store = getattr(self, "_edge_store", None) or {}
        if not hasattr(self, "_edge_store"):
            self._edge_store = {}
            store = self._edge_store

        changed = False
        usage = self.edge_present_usage
        if hasattr(usage, 'get'):
            usage = usage.get()

        for edge_idx in range(len(usage)):
            usage_val = int(usage[edge_idx])
            if usage_val > 0:
                if store.get(edge_idx, 0) != usage_val:
                    store[edge_idx] = usage_val
                    changed = True
            elif edge_idx in store:
                del store[edge_idx]
                changed = True

        return changed


    def _live_edge_count(self) -> int:
        """Return the number of edges in the graph."""
        if hasattr(self, 'indptr_cpu') and hasattr(self, 'indices_cpu'):
            return len(self.indices_cpu)
        elif hasattr(self, 'edge_present_usage'):
            return len(self.edge_present_usage)
        else:
            return 0


    def rip_up_net(self, net_id: str) -> None:
        """Remove a previously committed path for net_id from congestion accounting"""
        # look up edges this net owns
        edges = self.net_edge_paths.get(net_id, [])
        if not edges:
            return

        for k in edges:
            # Convert key to CSR index for int-only store
            idx = None
            if isinstance(k, int):
                idx = k
            elif hasattr(self, 'edge_lookup'):
                # Try to find CSR index from key
                idx = self.edge_lookup.get(k)

            if idx is not None and idx in self._edge_store:
                # Decrement usage (int-only store)
                self._edge_store[idx] = max(0, self._edge_store[idx] - 1)
                # Handle owners separately
                if hasattr(self, 'edge_owners') and idx in self.edge_owners:
                    self.edge_owners[idx].discard(net_id)

            # Update edge_owners map for capacity filter tracking
            # Extract node indices from the key to find the edge index
            if len(k) >= 3:
                layer, a, b = k[0], k[1], k[2]
                idx = self._edge_index.get((a, b)) or self._edge_index.get((b, a))
                if idx is not None and hasattr(self, "edge_owners"):
                    s = self.edge_owners.get(idx)
                    if s is not None:
                        s.discard(net_id)
                        if not s:
                            self.edge_owners.pop(idx, None)

        # clear maps
        self.net_edge_paths.pop(net_id, None)
        self.routed_nets.pop(net_id, None)
        self.committed_paths.pop(net_id, None)

        # optional: if you keep a present-usage bitmap, call a dec helper
        # self._dec_edge_usage_for_net(net_id)


    def commit_net_path(self, net_id: str, path_node_indices: List[int]) -> None:
        """
        Commit a found path: update canonical edge accounting, ownership,
        present-usage counters, route maps, and spatial indexes.
        """
        # Edge list for this net
        edge_keys_for_net: List[tuple] = []
        layer_pitch = getattr(self.geometry, "pitch", 0.4)
        half = layer_pitch / 2.0

        # walk consecutive node pairs
        for i in range(len(path_node_indices) - 1):
            a = path_node_indices[i]
            b = path_node_indices[i + 1]
            ax, ay, az = self._idx_to_coord(a)
            bx, by, bz = self._idx_to_coord(b)

            # Use node-index-based key for consistent lookup
            # Store format: (layer, node_a, node_b) with canonical ordering
            if b < a:
                a, b = b, a  # canonical ordering
            k = (az, a, b)  # layer, node1, node2

            # Build CSR index and update present + staged deltas (store merged at [COMMIT])
            # Use authoritative CSR edge_lookup, not the legacy _edge_index
            if not getattr(self, 'edge_lookup', None) or getattr(self, '_edge_lookup_size', 0) != self._live_edge_count():
                self._build_edge_lookup_from_csr()
            idx = self.edge_lookup.get((a, b)) or self.edge_lookup.get((b, a))
            if idx is not None:
                if not hasattr(self, "edge_usage_count") or self.edge_usage_count is None:
                    self.edge_usage_count = {}
                self.edge_usage_count[idx] = 1 + int(self.edge_usage_count.get(idx, 0))

                # Update edge_owners map for capacity filter tracking
                if not hasattr(self, "edge_owners") or self.edge_owners is None:
                    self.edge_owners = {}
                owners = self.edge_owners.get(idx)
                if owners is None or not isinstance(owners, set):
                    owners = set()
                    self.edge_owners[idx] = owners
                owners.add(net_id)
                # Update present usage and batch deltas for commit
                try:
                    self.edge_present_usage[idx] += 1
                except Exception:
                    import numpy as _np
                    a_arr = _np.asarray(self.edge_present_usage, dtype=_np.float32)
                    a_arr[idx] += 1
                    self.edge_present_usage = a_arr
                if not hasattr(self, '_batch_deltas') or self._batch_deltas is None:
                    self._batch_deltas = {}
                self._batch_deltas[idx] = int(self._batch_deltas.get(idx, 0)) + 1

            edge_keys_for_net.append(k)

            # R-tree / spatial index for clearance (only once per unique seg)
            if self._clearance_enabled:
                (x1, y1) = self.geometry.lattice_to_world(ax, ay)
                (x2, y2) = self.geometry.lattice_to_world(bx, by)
                try:
                    idx = self._clearance_rtrees.get(az)
                    if idx is not None:
                        xmin, xmax = (min(x1, x2) - half), (max(x1, x2) + half)
                        ymin, ymax = (min(y1, y2) - half), (max(y1, y2) + half)
                        idx.insert(hash((k, net_id)) & 0x7FFFFFFF, (xmin, ymin, xmax, ymax), obj=("track", net_id))
                except Exception:
                    # best-effort; don't crash on R-tree oddities
                    pass

        # record net -> edge list
        self.net_edge_paths[net_id] = edge_keys_for_net

        # **critical**: this is what _pathfinder_negotiation reports as "routed"
        self.routed_nets[net_id] = path_node_indices

        # convenience: also mirror in committed_paths for GUI
        self.committed_paths[net_id] = path_node_indices

        # mark that we actually routed something this session
        self._negotiation_ran = True


    def update_congestion_costs(self, pres_fac_mult: float = None):
        """Update PathFinder costs after iteration with proper capping"""
        if pres_fac_mult is None:
            pres_fac_mult = self._pres_mult

        overfull_count = 0

        store = self._edge_store
        for key, usage_count in store.items():
            extra = max(0, int(usage_count) - self._edge_capacity)
            overfull_count += extra

            # Historical cost tracking (separate from store)
            if extra > 0 and hasattr(self, 'edge_history'):
                if key not in self.edge_history:
                    self.edge_history[key] = 0.0
                self.edge_history[key] = min(self.edge_history[key] + self._hist_inc * extra, self._hist_cap)
            # Note: present cost is computed per-move as pres_fac * extra, not stored per-edge

        store = self._edge_store
        logger.info(f"[PF-COSTS] Updated {len(store)} edge costs, {overfull_count} overfull")
        return overfull_count


    def _build_ripup_queue(self, valid_net_ids: List[str]) -> List[str]:
        """Build congestion-driven rip-up queue targeting nets on overfull edges"""
        offenders = {}

        # Find nets sitting on overfull edges
        store = self._edge_store
        owners_map = getattr(self, 'edge_owners', {})
        for key, usage_count in store.items():
            extra = int(usage_count) - self._edge_capacity
            if extra > 0:
                # Get owners from separate owners map
                owners = owners_map.get(key, set())
                for owner in owners:
                    offenders[owner] = offenders.get(owner, 0) + 1

        # Failed nets must be in the queue (high priority)
        for nid in self._failed_nets_last_iter:
            if nid in valid_net_ids:  # Only include valid nets
                offenders[nid] = offenders.get(nid, 0) + 10  # boost priority

        # If no offenders, fall back to all nets
        if not offenders:
            import random
            queue = valid_net_ids.copy()
            random.shuffle(queue)
            return queue

        # Order by severity (most offenses first), break ties randomly
        import random
        ordered = sorted(offenders.items(), key=lambda kv: (-kv[1], random.random()))

        # Add remaining nets that aren't offenders
        queue = [nid for nid, _ in ordered if nid in valid_net_ids]
        remaining = [nid for nid in valid_net_ids if nid not in queue]
        random.shuffle(remaining)
        queue.extend(remaining)

        # Enhanced logging for verification
        offender_count = len([x for x in offenders.values() if x > 0])
        logger.info(f"[RIPUP-QUEUE] {offender_count} offenders identified")
        if ordered:
            logger.info(f"[RIPUP] top offenders: {ordered[:10]}")

        return queue

    def _pick_preferred_even_layer(self, net_id: str) -> int:
        """Hash net_id to preferred even layer for round-robin load balancing.

        Uses deterministic hash with good distribution to assign each net
        a preferred even (vertical routing) layer among L0, L2, L4, L6, etc.
        This breaks symmetry and prevents all nets from choosing the same column.
        """
        layer_count = getattr(self.config, 'layer_count', 18)
        even_layers = [z for z in range(layer_count) if (z & 1) == 0]
        if not even_layers:
            return 0

        # Deterministic hash with good distribution
        h = (hash(net_id) ^ 0x9E3779B9) & 0xffffffff
        return even_layers[h % len(even_layers)]

    def _apply_roundrobin_layer_bias_fast(self, net_id: str, source_idx: int) -> None:
        """Fast round-robin layer bias using column multiplier table.

        This optimized version avoids the GPU↔CPU bottleneck by:
        1. Creating a small column multiplier table on CPU (O(L×W))
        2. Applying it to vertical edges on GPU in vectorized fashion (O(E))
        3. No large array transfers - only tiny metadata

        Original version took 7-8s per net, this takes <10ms per net.
        """
        import numpy as np

        # Get grid dimensions
        x_steps = getattr(self.lattice, 'x_steps', 0) or getattr(self.geometry, 'x_steps', 0)
        y_steps = getattr(self.lattice, 'y_steps', 0) or getattr(self.geometry, 'y_steps', 0)
        layer_count = getattr(self.config, 'layer_count', 0) or getattr(self.geometry, 'layer_count', 6)

        if x_steps == 0 or y_steps == 0:
            return

        plane_size = x_steps * y_steps

        # Get source coordinates
        src_z = source_idx // plane_size
        src_remainder = source_idx % plane_size
        src_y = src_remainder // x_steps
        src_x = src_remainder % x_steps

        # Get even layers (routing layers)
        even_layers = [l for l in range(layer_count) if l % 2 == 0]
        if not even_layers:
            return

        # Hash net_name to pick preferred layer
        h = (hash(net_id) ^ 0x9E3779B9) & 0xffffffff
        preferred_z = even_layers[h % len(even_layers)]

        logger.info(f"[ROUNDROBIN] net={net_id} preferred_even_layer=L{preferred_z}")

        # Parameters
        alpha = self.config.first_vertical_roundrobin_alpha  # 0.15 default
        window_cols = 32  # ~8mm at 0.25mm pitch, will scale with actual pitch
        pitch = self.config.grid_pitch
        window_cols = int(8.0 / pitch)  # 8mm window

        # Create column multiplier table: shape [layer_count, x_steps]
        # Default: all ones (no modification)
        col_mult = np.ones((layer_count, x_steps), dtype=np.float32)

        # Apply bias only to even (vertical routing) layers within window of source
        x_min = max(0, src_x - window_cols)
        x_max = min(x_steps - 1, src_x + window_cols)

        for z in even_layers:
            if z == preferred_z:
                # Preferred layer gets discount
                col_mult[z, x_min:x_max+1] *= (1.0 - alpha)
            else:
                # Other even layers get penalty
                col_mult[z, x_min:x_max+1] *= (1.0 + alpha)

        # Apply multipliers to vertical edges on GPU (vectorized)
        self._apply_column_multipliers_gpu(col_mult, plane_size, x_steps, y_steps, layer_count)

        #         logger.debug(f"[ROUNDROBIN] net={net_id} applied bias via column multipliers (window={window_cols} cols)")

    def _apply_column_multipliers_gpu(self, col_mult: np.ndarray, plane_size: int,
                                      x_steps: int, y_steps: int, layer_count: int) -> None:
        """Apply column multipliers to vertical edges on GPU in vectorized fashion."""
        try:
            import cupy as cp
        except ImportError:
            # Fallback to CPU if no GPU
            self._apply_column_multipliers_cpu(col_mult, plane_size, x_steps, y_steps, layer_count)
            return

        # Get edge costs (should be on GPU)
        if not hasattr(self.edge_total_cost, 'get'):
            # Already CPU array, use CPU path
            self._apply_column_multipliers_cpu(col_mult, plane_size, x_steps, y_steps, layer_count)
            return

        # Get adjacency matrix
        adj_indptr = self.adjacency_matrix.indptr
        adj_indices = self.adjacency_matrix.indices

        # Upload column multiplier table to GPU (tiny - only L×X floats)
        col_mult_gpu = cp.asarray(col_mult)

        # Get via edges mask (if available)
        if hasattr(self, '_via_edges') and self._via_edges is not None:
            via_edges = cp.asarray(self._via_edges) if not hasattr(self._via_edges, 'get') else self._via_edges
        else:
            # Need to identify vertical edges on the fly
            # For now, skip if we don't have the mask
            return

        # Create custom CUDA kernel for applying multipliers
        # This is still a simple vectorized operation on GPU
        apply_mult_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void apply_column_multipliers(
            float* edge_costs,
            const int* indptr,
            const int* indices,
            const unsigned char* via_edges,
            const float* col_mult,
            int num_nodes,
            int plane_size,
            int x_steps,
            int layer_count
        ) {
            int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (node_idx >= num_nodes) return;

            // Decode node to (x, y, z)
            int z = node_idx / plane_size;
            int remainder = node_idx % plane_size;
            int y = remainder / x_steps;
            int x = remainder % x_steps;

            // Apply to outgoing via edges
            for (int e_idx = indptr[node_idx]; e_idx < indptr[node_idx + 1]; e_idx++) {
                if (via_edges[e_idx]) {
                    int neighbor_idx = indices[e_idx];
                    int neighbor_z = neighbor_idx / plane_size;

                    // Vertical edge - use source layer's column multiplier
                    if (neighbor_z != z) {
                        edge_costs[e_idx] *= col_mult[z * x_steps + x];
                    }
                }
            }
        }
        ''', 'apply_column_multipliers')

        # Launch kernel
        num_nodes = len(adj_indptr) - 1
        threads_per_block = 256
        num_blocks = (num_nodes + threads_per_block - 1) // threads_per_block

        apply_mult_kernel(
            (num_blocks,), (threads_per_block,),
            (self.edge_total_cost, adj_indptr, adj_indices, via_edges,
             col_mult_gpu.ravel(), num_nodes, plane_size, x_steps, layer_count)
        )

    def _apply_column_multipliers_cpu(self, col_mult: np.ndarray, plane_size: int,
                                      x_steps: int, y_steps: int, layer_count: int) -> None:
        """CPU fallback for applying column multipliers."""
        import numpy as np

        # Get adjacency matrix on CPU
        adj_indptr = self.adjacency_matrix.indptr
        adj_indices = self.adjacency_matrix.indices

        if hasattr(adj_indptr, 'get'):
            adj_indptr = adj_indptr.get()
            adj_indices = adj_indices.get()

        # Get edge costs on CPU
        edge_costs = self.edge_total_cost
        if hasattr(edge_costs, 'get'):
            edge_costs = edge_costs.get()

        # Get via edges mask
        via_edges = getattr(self, '_via_edges', None)
        if via_edges is not None and hasattr(via_edges, 'get'):
            via_edges = via_edges.get()

        num_nodes = len(adj_indptr) - 1
        modified_count = 0

        # Apply multipliers to vertical edges
        for node_idx in range(num_nodes):
            # Decode node to (x, y, z)
            z = node_idx // plane_size
            remainder = node_idx % plane_size
            y = remainder // x_steps
            x = remainder % x_steps

            # Process outgoing edges
            for e_idx in range(adj_indptr[node_idx], adj_indptr[node_idx + 1]):
                # Check if via edge
                is_via = via_edges[e_idx] if via_edges is not None else True

                if is_via:
                    neighbor_idx = adj_indices[e_idx]
                    neighbor_z = neighbor_idx // plane_size

                    # Vertical edge
                    if neighbor_z != z:
                        edge_costs[e_idx] *= col_mult[z, x]
                        modified_count += 1

        # Write back if needed
        if hasattr(self.edge_total_cost, 'get'):
            try:
                import cupy as cp
                self.edge_total_cost = cp.asarray(edge_costs)
            except ImportError:
                self.edge_total_cost = edge_costs
        else:
            self.edge_total_cost = edge_costs

        #         logger.debug(f"[ROUNDROBIN-CPU] Modified {modified_count} via edges")

    def _apply_roundrobin_layer_bias(self, net_id: str, source_idx: int) -> int:
        """Apply round-robin layer preference for first vertical run near source portal.

        This breaks symmetry when nets leave their portals by giving each net a
        preferred even layer (L2/L4/L6...) for its first vertical segment. When all
        nets see identical costs near portals, they all pick the same "cheapest" even
        layer, creating an instant elevator shaft. This fix assigns each net a different
        preferred layer via hashing.

        Args:
            net_id: Net name/identifier
            source_idx: Global source node index

        Returns:
            preferred_layer: The preferred even layer index for this net
        """
        import numpy as np

        # Get even layers (0-indexed where even indices = L1, L3, L5... in KiCad)
        # But we want routing layers which are typically the even indices in 0-indexed
        layer_count = getattr(self.config, 'layer_count', 0)
        if layer_count == 0:
            layer_count = getattr(self.geometry, 'layer_count', 6)

        # Even layer indices for routing (0, 2, 4, ...)
        even_layers = [l for l in range(layer_count) if l % 2 == 0]
        if not even_layers:
            return 0  # Fallback

        # Hash net_name to pick preferred layer
        bucket = hash(net_id) % len(even_layers)
        preferred_z = even_layers[bucket]

        logger.info(f"[ROUNDROBIN] net={net_id} preferred_even_layer=L{preferred_z}")

        # Get grid dimensions for node -> (x, y, z) conversion
        x_steps = getattr(self.lattice, 'x_steps', 0) or getattr(self.geometry, 'x_steps', 0)
        y_steps = getattr(self.lattice, 'y_steps', 0) or getattr(self.geometry, 'y_steps', 0)

        if x_steps == 0 or y_steps == 0:
            logger.warning(f"[ROUNDROBIN] Cannot apply bias: grid dimensions not available")
            return preferred_z

        plane_size = x_steps * y_steps

        # Get source coordinates
        src_z = source_idx // plane_size
        src_remainder = source_idx % plane_size
        src_y = src_remainder // x_steps
        src_x = src_remainder % x_steps

        # Define "near portal" distance (8mm / pitch)
        pitch = self.config.grid_pitch
        distance_mm = 8.0
        distance_cells = int(distance_mm / pitch)

        # Apply penalty to via edges that connect to non-preferred even layers
        # within distance_cells of the source
        alpha = self.config.first_vertical_roundrobin_alpha

        # Get edge costs (need to work with both CPU and GPU arrays)
        if hasattr(self.edge_total_cost, 'get'):
            edge_costs = self.edge_total_cost.get()
        else:
            edge_costs = self.edge_total_cost

        # Get adjacency matrix for edge lookup
        if hasattr(self.adjacency_matrix, 'get'):
            adj_indptr = self.adjacency_matrix.indptr.get()
            adj_indices = self.adjacency_matrix.indices.get()
        else:
            adj_indptr = self.adjacency_matrix.indptr
            adj_indices = self.adjacency_matrix.indices

        # Track how many edges we modify
        modified_count = 0

        # Scan nodes near source and penalize non-preferred vertical transitions
        for dx in range(-distance_cells, distance_cells + 1):
            for dy in range(-distance_cells, distance_cells + 1):
                nx, ny = src_x + dx, src_y + dy
                if not (0 <= nx < x_steps and 0 <= ny < y_steps):
                    continue

                # Check all layers at this (x, y) position
                for nz in range(layer_count):
                    node_idx = nz * plane_size + ny * x_steps + nx

                    # Get all outgoing edges from this node
                    edge_start = adj_indptr[node_idx]
                    edge_end = adj_indptr[node_idx + 1]

                    for edge_idx in range(edge_start, edge_end):
                        neighbor_idx = adj_indices[edge_idx]
                        neighbor_z = neighbor_idx // plane_size

                        # Check if this is a vertical (via) edge to an even layer
                        if neighbor_z != nz and neighbor_z in even_layers:
                            # If neighbor is not the preferred layer, penalize
                            if neighbor_z != preferred_z:
                                edge_costs[edge_idx] *= (1.0 + alpha)
                                modified_count += 1

        # Write back modified costs
        if hasattr(self.edge_total_cost, 'get'):
            # GPU array - need to convert back
            try:
                import cupy as cp
                self.edge_total_cost = cp.asarray(edge_costs)
            except ImportError:
                self.edge_total_cost = edge_costs
        else:
            self.edge_total_cost = edge_costs

        #         logger.debug(f"[ROUNDROBIN] net={net_id} modified {modified_count} via edges near source")

        return preferred_z


    def _select_offenders_for_ripup(self, routing_queue: List[str]) -> List[str]:
        """Select subset of nets to rip up based on congestion blame with freeze logic"""
        from collections import defaultdict

        # Initialize freeze tracking if not exists
        if not hasattr(self, '_frozen_nets'):
            self._frozen_nets = set()
            self._net_clean_iters = defaultdict(int)
            self._freeze_clean_iters = 2  # Freeze nets clean for 2+ iterations

        blame = {}
        touched = []

        # Calculate blame = sum over (usage - cap) on edges a net uses
        for net_id, keys in self.net_edge_paths.items():
            if not keys:  # Skip nets with no committed edges
                continue
            s = 0
            for k in keys:
                usage_count = self._edge_store.get(k)
                if usage_count is not None:
                    s += max(0, int(usage_count) - self._edge_capacity)
            blame[net_id] = s
            if s == 0:
                self._net_clean_iters[net_id] += 1
                if self._net_clean_iters[net_id] >= self._freeze_clean_iters:
                    self._frozen_nets.add(net_id)
            else:
                self._net_clean_iters[net_id] = 0
                if net_id in self._frozen_nets:
                    self._frozen_nets.discard(net_id)  # Unfreeze if now dirty
                touched.append(net_id)

        # Add failed nets from last iteration (high priority)
        for net_id in self._failed_nets_last_iter:
            if net_id not in blame:
                blame[net_id] = 0
            blame[net_id] += 10  # Boost priority
            if net_id not in touched:
                touched.append(net_id)

        if not touched:
            return []

        # Filter out frozen nets
        candidates = [n for n in touched if n not in self._frozen_nets]
        if not candidates:
            return []

        candidates.sort(key=lambda n: blame[n], reverse=True)

        # Tighter limits: 10% max, min 4, max 48
        ripup_fraction = 0.10  # 10% instead of 30%
        min_ripup = 4          # At least 4 nets
        max_ripup = 48         # But not more than 48 to avoid thrash

        num_to_rip = max(min_ripup, int(len(candidates) * ripup_fraction))
        num_to_rip = min(num_to_rip, max_ripup, len(candidates))

        selected = candidates[:num_to_rip]
        logger.debug(f"[OFFENDERS] {len(candidates)} candidates, {len(self._frozen_nets)} frozen, selected {len(selected)}")
        return selected


    def _prepare_net_for_reroute(self, net_id):
        prev = self._net_paths.get(net_id)
        if prev is None or len(prev) == 0:
            return None
        import numpy as np
        idx = np.asarray(prev, dtype=np.int64)
        np.subtract.at(self.edge_present_usage, idx, 1)
        # if you maintain owners, free them for this net:
        if hasattr(self, "edge_owners"):
            for e in idx.tolist():
                if self.edge_owners.get(e) == net_id:
                    del self.edge_owners[e]
        return idx  # return so we can restore on failure


    def _restore_net_after_failed_reroute(self, net_id, prev_idx):
        if prev_idx is None:
            return
        import numpy as np
        np.add.at(self.edge_present_usage, prev_idx, 1)
        if hasattr(self, "edge_owners"):
            for e in prev_idx.tolist():
                self.edge_owners[e] = net_id


    def _emit_capacity_analysis(self, successful: int, total_nets: int, overuse_count: int, failed_nets: int):
        """Emit honest capacity analysis when routing is capacity-limited"""
        logger.info("=" * 60)
        logger.info("CAPACITY ANALYSIS - Why routing failed:")
        logger.info("=" * 60)

        success_rate = (successful / total_nets) * 100 if total_nets > 0 else 0
        logger.info(f"FINAL RESULTS: {successful}/{total_nets} nets routed ({success_rate:.1f}%)")
        logger.info(f"FAILED NETS: {failed_nets}")
        logger.info(f"OVERUSE VIOLATIONS: {overuse_count} edges over capacity")

        # Layer usage analysis
        logger.info("\nLAYER USAGE ANALYSIS:")
        try:
            self._analyze_layer_capacity()
        except Exception as e:
            logger.warning(f"Layer analysis failed: {e}")

        # What-if analysis
        logger.info(f"\nCAPACITY INSIGHT: With current {self.geometry.layer_count if self.geometry else 6} layers:")
        if success_rate < 50:
            logger.info("• Severely capacity-limited - consider adding 2-4 more layers")
        elif success_rate < 80:
            logger.info("• Moderately capacity-limited - consider adding 1-2 more layers")
        else:
            logger.info("• Near capacity limits - one additional layer may resolve remaining conflicts")

        logger.info("=" * 60)


    def _identify_most_congested_nets(self, count: int) -> List[str]:
        """Identify nets contributing most to congestion for capacity-limited removal"""
        net_congestion_score = {}

        # Score each net by how much it contributes to overused edges
        for edge_idx, usage_count in self._edge_store.items():
            if usage_count > 1:  # overused edge
                congestion_contribution = usage_count - 1  # overuse amount
                # Get owners from separate tracking
                owners = self.edge_owners.get(edge_idx, set()) if hasattr(self, 'edge_owners') else set()
                for net_id in owners:
                    net_congestion_score[net_id] = net_congestion_score.get(net_id, 0) + congestion_contribution

        # Return the top N most congested nets
        sorted_nets = sorted(net_congestion_score.items(), key=lambda x: x[1], reverse=True)
        return [net_id for net_id, score in sorted_nets[:count]]


    def _analyze_layer_capacity(self):
        """Analyze per-layer usage and congestion"""
        if not hasattr(self, 'geometry') or self.geometry is None:
            logger.warning("No geometry system available for layer analysis")
            return

        layer_usage = {}
        for layer in range(self.geometry.layer_count):
            layer_usage[layer] = 0

        # Count routed segments per layer
        for net_id, path in self.routed_nets.items():
            if path and len(path) > 1:
                for i in range(len(path) - 1):
                    try:
                        coord1 = self._idx_to_coord(path[i])
                        coord2 = self._idx_to_coord(path[i + 1])
                        if coord1 and coord2 and coord1[2] == coord2[2]:  # Same layer
                            layer_usage[coord1[2]] += 1
                    except Exception:
                        continue

        logger.info("Layer usage distribution:")
        for layer in range(self.geometry.layer_count):
            direction = self.geometry.layer_directions[layer]
            usage = layer_usage.get(layer, 0)
            layer_name = self._map_layer_for_gui(layer)
            logger.info(f"  {layer_name} ({direction}): {usage} segments")


    def _dump_repro_bundle(self, successful: int, total_nets: int, failed_nets: int):
        """Dump small repro bundle for debugging failed routing"""
        import json
        import os
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"orthoroute_repro_{timestamp}.json"

        repro_data = {
            "metadata": {
                "timestamp": timestamp,
                "seed": getattr(self, '_routing_seed', 42),
                "instance_tag": getattr(self, '_instance_tag', 'unknown'),
                "total_nets": total_nets,
                "successful": successful,
                "failed": failed_nets
            },
            "config": {
                "max_iterations": self.config.max_iterations,
                "batch_size": getattr(self.config, 'batch_size', 32),
                "grid_pitch": self.config.grid_pitch,
                "layer_count": getattr(self.config, 'layer_count', 6)
            },
            "bounds": None,
            "grid_dims": None,
            "portals": [],
            "edges_sample": []
        }

        # Add bounds if available
        if hasattr(self, 'geometry') and self.geometry:
            repro_data["bounds"] = {
                "min_x": self.geometry.grid_min_x,
                "min_y": self.geometry.grid_min_y,
                "max_x": self.geometry.grid_max_x,
                "max_y": self.geometry.grid_max_y
            }
            repro_data["grid_dims"] = {
                "x_steps": self.geometry.x_steps,
                "y_steps": self.geometry.y_steps
            }

        # Add first 200 portals
        if hasattr(self, '_pad_portals'):
            portal_items = list(self._pad_portals.items())[:200]
            for pad_id, portal in portal_items:
                repro_data["portals"].append({
                    "pad_id": pad_id,
                    "x": portal.x,
                    "y": portal.y,
                    "layer": portal.layer,
                    "net": portal.net
                })

        # Add first 1k edges
        if hasattr(self, 'edges') and self.edges:
            edge_sample = self.edges[:1000]
            for from_idx, to_idx, cost in edge_sample:
                repro_data["edges_sample"].append([int(from_idx), int(to_idx), float(cost)])

        # Add committed paths for determinism verification
        if hasattr(self, 'routed_nets'):
            paths_sample = {}
            for net_id, path in list(self.routed_nets.items())[:50]:  # First 50 nets
                if path and len(path) > 0:
                    paths_sample[net_id] = [int(node) for node in path]
            repro_data["committed_paths"] = paths_sample

        # Write repro bundle
        try:
            with open(filename, 'w') as f:
                json.dump(repro_data, f, indent=2)
            logger.info(f"[REPRO] Dumped repro bundle: {filename}")
        except Exception as e:
            logger.error(f"[REPRO] Failed to dump repro bundle: {e}")


    def _calculate_iteration_metrics(self, successful: int, failed_nets: int, routes_changed: int,
                                   total_relax_calls: int, relax_calls_per_net: list, 
                                   total_nets: int) -> dict:
        """Calculate comprehensive iteration metrics"""
        metrics = {}
        
        # Basic routing metrics
        metrics['success_rate'] = successful / total_nets * 100 if total_nets > 0 else 0.0
        metrics['failure_rate'] = failed_nets / total_nets * 100 if total_nets > 0 else 0.0
        
        # Relax call statistics
        if relax_calls_per_net:
            metrics['avg_relax_calls'] = sum(relax_calls_per_net) / len(relax_calls_per_net)
            sorted_relax = sorted(relax_calls_per_net)
            metrics['p95_relax_calls'] = sorted_relax[int(0.95 * len(sorted_relax))] if sorted_relax else 0
        else:
            metrics['avg_relax_calls'] = 0.0
            metrics['p95_relax_calls'] = 0.0
        
        # CRITICAL: Use CANONICAL EDGE STORE (authoritative)
        over_capacity_count = 0
        overuse_values = []
        history_total = 0.0

        # Count overuse from canonical edge store
        nets_on_edges = set()
        for edge_idx, usage_count in self._edge_store.items():
            # PathFinder capacity = 1 per edge (no sharing allowed)
            capacity = 1

            # Count edges with usage > capacity (multiple nets using same edge)
            if usage_count > capacity:
                over_capacity_count += 1
                overuse_amount = usage_count - capacity
                overuse_values.append(overuse_amount)

            # Historical costs now tracked separately in edge arrays
            # Skip since we use simple int store now

        metrics['over_capacity_edges'] = over_capacity_count

        if overuse_values:
            metrics['max_overuse'] = max(overuse_values)
            metrics['avg_overuse'] = sum(overuse_values) / len(overuse_values)
        else:
            metrics['max_overuse'] = 0.0
            metrics['avg_overuse'] = 0.0

        metrics['history_total'] = history_total
        
        return metrics
    

    def _route_batch_gpu_with_metrics(self, batch: List[Tuple[str, Tuple[int, int]]]) -> tuple:
        """Route batch of nets using GPU SSSP with detailed metrics"""
        batch_results = []
        batch_metrics = []
        
        logger.info(f"[ROUTING] Batch of {len(batch)} nets...")
        
        # Multi-ROI parallel processing for both "multi_roi" and "multi_roi_bidirectional" modes
        if (self.config.mode in ["multi_roi", "multi_roi_bidirectional"]) and self.config.roi_parallel and len(batch) > 1:
            logger.info(f"DEBUG: Entering _route_multi_roi_batch with {len(batch)} nets using mode: {self.config.mode}")
            multi_results, multi_metrics = self._route_multi_roi_batch(batch)
            logger.info(f"DEBUG: _route_multi_roi_batch completed, got {len(multi_results)} results")
            return multi_results, multi_metrics
        
        # Sequential processing for other modes with batch caps and time budget
        batch_sz = min(len(batch), getattr(self.config, "batch_size", 32))
        # Final batch cap from environment
        batch_sz = min(batch_sz, BATCH_SIZE)
        TIME_BUDGET_S = getattr(self.config, "per_net_budget_s", PER_NET_BUDGET_S)

        for i, (net_id, (source_idx, sink_idx)) in enumerate(batch[:batch_sz]):
            logger.info(f"  Progress: routing net {i+1}/{len(batch[:batch_sz])}: {net_id}")

            import time
            t0 = time.time()

            # Apply round-robin layer bias for first iterations (when symmetry matters most)
            # DISABLED: Performance bottleneck - causes 7-8s per net due to GPU↔CPU transfers
            # TODO: Optimize by doing GPU-side modifications or pre-computing once
            if False and self.current_iteration <= 3:
                try:
                    self._apply_roundrobin_layer_bias(net_id, source_idx)
                except Exception as e:
                    logger.warning(f"[ROUNDROBIN] Failed to apply bias for {net_id}: {e}")

            # PRAGMATIC FIX: Test first few nets on CPU, use GPU only if it proves fast
            emergency_cpu_only = EMERGENCY_CPU_ONLY
            smart_fallback = SMART_FALLBACK  # GPU->CPU fallback

            if emergency_cpu_only or (smart_fallback and i < 3):  # Test first 3 nets on CPU
                logger.info(f"[SMART-FALLBACK] {net_id}: Using CPU (emergency={emergency_cpu_only}, smart={smart_fallback and i < 3})")
                path = self._cpu_dijkstra_fallback(source_idx, sink_idx)
                net_metrics = {'smart_fallback': True, 'reason': 'first_nets_cpu_test'}
            else:
                # Route with chosen algorithm - prefer fast ROI Near-Far; only use delta_stepping when explicitly asked
                if self.config.mode in ("delta_stepping", "fullgraph"):
                    path, net_metrics = self._gpu_delta_stepping_sssp_with_metrics(
                        source_idx, sink_idx, time_budget_s=TIME_BUDGET_S, t0=t0, net_id=net_id
                    )
                else:  # near_far (default) - much faster for typical nets
                    path, net_metrics = self._gpu_roi_near_far_sssp_with_metrics(
                        net_id, source_idx, sink_idx, time_budget_s=TIME_BUDGET_S, t0=t0
                    )

            # If the GPU path returned None or blew the budget, hard fallback to CPU
            if self._deadline_passed(t0, TIME_BUDGET_S) and not hasattr(self.config, 'use_cpu_routing'):
                logger.info(f"[TIME-BUDGET] {net_id}: GPU > {TIME_BUDGET_S:.2f}s → CPU fallback")
                path = self._cpu_dijkstra_fallback(source_idx, sink_idx)
                net_metrics = {'roi_fallback': True, 'reason': 'time_budget'}

            batch_results.append(path)
            batch_metrics.append(net_metrics)

            # Commit path immediately for accounting
            if path and len(path) > 1:
                logger.info(f"[PATH] net={net_id} nodes={len(path)}")
                self.commit_net_path(net_id, path)              # updates edge_store + owners
                self._refresh_present_usage_from_store()   # rebuild usage vector
                pres_fac = getattr(self, '_current_pres_fac', 2.0)  # get current pres_fac
                self._update_edge_total_costs(pres_fac)         # raise costs on used edges
                # Accumulate edge usage on device
                self._accumulate_edge_usage_gpu(path)
        
        return batch_results, batch_metrics
    
    # ===== MULTI-ROI AUTO-TUNING & INSTRUMENTATION =====
    

    def _log_multi_roi_performance(self):
        """Log comprehensive multi-ROI performance statistics"""
        stats = self._multi_roi_stats
        
        logger.info("=" * 60)
        logger.info("MULTI-ROI PERFORMANCE DASHBOARD")
        logger.info("=" * 60)
        logger.info(f"Total chunks processed: {stats['total_chunks']}")
        logger.info(f"Total nets processed: {stats['total_nets']}")
        logger.info(f"Successful nets: {stats['successful_nets']}")
        logger.info(f"Success rate: {stats['successful_nets']/max(1, stats['total_nets'])*100:.1f}%")
        logger.info(f"Average ms per net: {stats['avg_ms_per_net']:.1f}ms")
        logger.info(f"Target ms per net: {self._target_ms_per_net}ms")
        logger.info(f"Performance vs target: {stats['avg_ms_per_net']/self._target_ms_per_net*100:.1f}%")
        logger.info(f"Queue cap hits: {stats['queue_cap_hits']}")
        logger.info(f"Peak memory usage: {stats['memory_usage_peak_mb']:.1f}MB")
        logger.info(f"Current K: {self._current_k}")
        
        if stats['k_adjustments']:
            logger.info("Recent K adjustments:")
            for adj in stats['k_adjustments'][-3:]:  # Show last 3
                logger.info(f"  Chunk {adj['chunk']}: {adj['old_k']}→{adj['new_k']} ({adj['reason']})")
        
        logger.info("=" * 60)
    

    def _refresh_present_usage_from_accounting(self, force_rebuild=False):
        """Rebuild present usage arrays from canonical edge store accounting data.

        Args:
            force_rebuild: Force rebuild even if arrays appear current
        """
        # CSR-only: rebuild present usage directly from _edge_store integer keys
        import numpy as np
        # Ensure arrays sized to live CSR
        self._sync_edge_arrays_to_live_csr()
        E = self._live_edge_count()
        if getattr(self, 'edge_present_usage', None) is None or len(self.edge_present_usage) != E:
            self.edge_present_usage = np.zeros(E, dtype=np.float32)
        else:
            self.edge_present_usage.fill(0.0)

        store = getattr(self, "_edge_store", None) or getattr(self, 'edge_store', None) or {}
        mapped = 0
        for ei, usage in store.items():
            if isinstance(ei, int) and 0 <= ei < E and int(usage) > 0:
                self.edge_present_usage[ei] = float(usage)
                mapped += 1
        logger.info("[UPF] Usage refresh: mapped %d store entries to present usage", mapped)


    def _apply_capacity_limit_after_negotiation(self) -> None:
        """Apply capacity limits to edge usage after PathFinder negotiation completes.

        Ensures that final edge usage respects capacity constraints by clamping
        overused edges and updating present usage arrays accordingly.
        """
        import numpy as np
        # Build usage from committed_paths
        self._refresh_present_usage_from_store()
        usage = self.edge_present_usage
        cap   = self.edge_capacity
        if hasattr(usage, "get"): usage = usage.get()
        if hasattr(cap,   "get"): cap   = cap.get()

        # Fast map: edge_idx -> owner set (kept up-to-date in commit/rip-up)
        owners_map = getattr(self, "edge_owners", None)        # {idx: set(net_id)}
        if owners_map is None:
            owners_map = {}
            # populate once from committed paths
            for nid, nodes in self.committed_paths.items():
                for a, b in zip(nodes[:-1], nodes[1:]):
                    idx = self._edge_index.get((a, b)) or self._edge_index.get((b, a))
                    if idx is not None:
                        owners_map.setdefault(idx, set()).add(nid)
            self.edge_owners = owners_map

        over_idx = np.flatnonzero(usage > cap)
        # Peel offenders until no edge is overfull or nothing left
        while len(over_idx) > 0:
            # Score nets by how many overfull edges they occupy
            score: dict[str,int] = {}
            for e in over_idx:
                for nid in owners_map.get(int(e), ()):
                    score[nid] = score.get(nid, 0) + 1

            if not score:
                break

            # Rip the worst net
            worst = max(score.items(), key=lambda kv: kv[1])[0]
            self.rip_up_net(worst)

            # Update owners_map and usage for next round
            if worst in self.committed_paths:
                del self.committed_paths[worst]
            # rebuild owners_map entries quickly
            for e, s in list(owners_map.items()):
                if worst in s:
                    s.remove(worst)
                    if not s:
                        owners_map.pop(e, None)

            self._refresh_present_usage_from_store()
            usage = self.edge_present_usage
            if hasattr(usage, "get"): usage = usage.get()
            over_idx = np.flatnonzero(usage > cap)

        # routed_nets mirror for GUI/logs
        self.routed_nets = dict(self.committed_paths)


    def _route_batch_gpu(self, batch: List[Tuple[str, Tuple[int, int]]]) -> List[Optional[List[int]]]:
        """Route batch of nets using GPU ∆-stepping SSSP"""
        batch_results = []
        
        for net_id, (source_idx, sink_idx) in batch:
            # Use fast GPU SSSP instead of Python A*
            path = self._gpu_delta_stepping_sssp(source_idx, sink_idx, net_id=net_id)
            batch_results.append(path)
            
            # Accumulate edge usage on device
            if path and len(path) > 1:
                self._accumulate_edge_usage_gpu(path)
        
        return batch_results
    

    def _force_top_k_offenders(self, k: int) -> List[str]:
        """Force selection of top K offenders by blame (guardrail for deadlock)"""
        blame = {}
        for net_id, keys in self.net_edge_paths.items():
            if not keys:
                continue
            s = 0
            for key in keys:
                usage_count = self._edge_store.get(key)
                if usage_count is not None:
                    s += max(0, int(usage_count) - self._edge_capacity)
            if s > 0:
                blame[net_id] = s

        if not blame:
            return []

        sorted_offenders = sorted(blame.items(), key=lambda x: -x[1])
        selected = [net_id for net_id, score in sorted_offenders[:k]]
        logger.info(f"[GUARDRAIL] Forced {len(selected)} top offenders: {selected[:5]}...")
        return selected


    def _compute_overuse_from_edge_store(self) -> tuple[int, int]:
        """Compute current overuse from edge store - single source of truth"""
        overuse_sum = 0
        overuse_edges = 0
        store = self._edge_store
        for key, usage_count in store.items():
            extra = max(0, int(usage_count) - self._edge_capacity)
            overuse_sum += extra
            if extra > 0:
                overuse_edges += 1
        return overuse_sum, overuse_edges


    def _bump_history(self, overuse_sum: int):
        """Update historical costs based on current overuse"""
        if overuse_sum == 0:
            return

        hist_inc = getattr(self, '_hist_inc', 0.4)
        hist_cap = getattr(self, '_hist_cap', 1000.0)

        store = self._edge_store
        # Initialize edge_history dict if needed for historical cost tracking
        if not hasattr(self, 'edge_history'):
            self.edge_history = {}

        for key, usage_count in store.items():
            extra = max(0, int(usage_count) - self._edge_capacity)
            if extra > 0:
                if key not in self.edge_history:
                    self.edge_history[key] = 0.0
                self.edge_history[key] = min(self.edge_history[key] + hist_inc * extra, hist_cap)


    def _get_adaptive_roi_margin(self, net_id: str, base_margin_mm: float = 10.0) -> float:
        """Get adaptive ROI margin based on net failure history"""
        failure_count = self._net_failure_count.get(net_id, 0)

        # Start with base margin, expand by 1-2 grid cells per failure
        grid_pitch = self.config.grid_pitch  # typically 0.4mm
        expansion = failure_count * 2 * grid_pitch  # 2 grid cells per failure

        # Calculate new margin with expansion
        new_margin = base_margin_mm + expansion

        # Store the new margin for this net
        self._net_roi_margin[net_id] = new_margin

        if expansion > 0:
            logger.info(f"[ROI++] net={net_id} expand={new_margin:.1f}mm fail={failure_count}")

        return new_margin


    def _update_net_failure_count(self, net_id: str, failed: bool):
        """Update failure count for ROI expansion"""
        if failed:
            self._net_failure_count[net_id] = self._net_failure_count.get(net_id, 0) + 1
        else:
            # Reset failure count on success
            if net_id in self._net_failure_count:
                del self._net_failure_count[net_id]
            if net_id in self._net_roi_margin:
                del self._net_roi_margin[net_id]


    def _log_column_balance_metrics(self, iteration: int) -> None:
        """
        Log column balance metrics to monitor vertical traffic distribution.

        This tracks how vertical traffic is distributed across columns (x-coordinates)
        on even-numbered vertical routing layers. The Gini coefficient measures
        inequality - lower is better (0 = perfectly balanced, 1 = all traffic in one column).

        This helps diagnose the "elevator shaft" problem where traffic concentrates
        in a few overused columns instead of spreading evenly.
        """
        import numpy as np

        def compute_gini(values):
            """Compute Gini coefficient (0=perfect equality, 1=perfect inequality)"""
            values = np.array(sorted(values), dtype=np.float64)
            n = len(values)
            if n == 0 or values.sum() == 0:
                return 0.0
            index = np.arange(1, n + 1)
            return (2.0 * np.sum(index * values)) / (n * values.sum()) - (n + 1) / n

        # Get lattice dimensions
        if not hasattr(self, 'lattice') or self.lattice is None:
            logger.debug("[COLUMN-BALANCE] No lattice available, skipping metrics")
            return

        lattice = self.lattice
        num_layers = lattice.layers
        x_steps = lattice.x_steps
        y_steps = lattice.y_steps

        # Track column usage per layer
        layer_gini = {}  # layer -> gini coefficient
        layer_top5 = {}  # layer -> top 5 overused columns

        # Analyze vertical routing layers (even-numbered internal layers: 0, 2, 4, 6, ...)
        vertical_layers = [l for l in range(num_layers) if lattice.get_legal_axis(l) == 'v']

        for layer in vertical_layers:
            col_usage = {}  # x -> usage count

            # Count vertical edges per column by examining all routed nets
            for net_id, path_indices in self._net_paths.items():
                if path_indices is None or len(path_indices) == 0:
                    continue

                # Convert to list if it's a numpy array
                if hasattr(path_indices, 'tolist'):
                    path_indices = path_indices.tolist()
                elif not isinstance(path_indices, list):
                    path_indices = list(path_indices)

                # Walk consecutive node pairs in the path
                for i in range(len(path_indices) - 1):
                    try:
                        node_a = int(path_indices[i])
                        node_b = int(path_indices[i + 1])

                        # Get coordinates for both nodes
                        xa, ya, za = lattice.idx_to_coord(node_a)
                        xb, yb, zb = lattice.idx_to_coord(node_b)

                        # Check if this is a vertical edge on the current layer
                        # Vertical edge: same x, different y, same layer
                        if za == zb == layer and xa == xb and ya != yb:
                            # This is a vertical edge at column x
                            col_usage[xa] = col_usage.get(xa, 0) + 1

                    except (ValueError, TypeError, IndexError) as e:
                        # Skip malformed paths
                        continue

            # Compute statistics for this layer
            if col_usage:
                usage_vals = list(col_usage.values())
                gini = compute_gini(usage_vals)
                layer_gini[layer] = gini

                # Get top 5 columns by usage
                top5 = sorted(col_usage.items(), key=lambda kv: -kv[1])[:5]
                layer_top5[layer] = top5

                # Log per-layer details
                logger.info(f"[COLUMN-STATS] Iter={iteration} L{layer}: "
                          f"top-5 columns by usage -> {top5}, gini={gini:.3f} "
                          f"(total_cols={len(col_usage)}, max_usage={max(usage_vals)}, avg_usage={np.mean(usage_vals):.1f})")
            else:
                layer_gini[layer] = 0.0
                layer_top5[layer] = []
                logger.info(f"[COLUMN-STATS] Iter={iteration} L{layer}: no vertical traffic")

        # Summary line showing Gini across all vertical layers
        if layer_gini:
            gini_summary = ", ".join([f"L{layer}: gini={gini:.2f}" for layer, gini in sorted(layer_gini.items())])
            avg_gini = np.mean(list(layer_gini.values()))
            logger.info(f"[COLUMN-BALANCE] Iter={iteration} Summary: {gini_summary} (avg={avg_gini:.2f}, lower is better)")
        else:
            logger.info(f"[COLUMN-BALANCE] Iter={iteration} Summary: No vertical layers with traffic")


    def _assert_terminals_reachable(self, valid_nets: Dict[str, Tuple[int, int]]) -> None:
        """TRIPWIRE: Validate terminal connectivity before negotiation"""
        logger.info(f"[REACHABILITY-TRIPWIRE] Testing {len(valid_nets)} terminals...")

        unreachable = 0
        for net_id, (source_idx, sink_idx) in valid_nets.items():
            # Test basic lattice connectivity
            if source_idx >= self.node_count or sink_idx >= self.node_count:
                logger.error(f"[REACHABILITY] {net_id}: Terminal out of bounds (src={source_idx}, snk={sink_idx}, max={self.node_count})")
                unreachable += 1
                continue

            # Test if terminals exist in portal registry
            source_coord = self._idx_to_coord(source_idx)
            sink_coord = self._idx_to_coord(sink_idx)

            if source_coord is None or sink_coord is None:
                logger.error(f"[REACHABILITY] {net_id}: Invalid coordinate mapping (src_idx={source_idx}→{source_coord}, snk_idx={sink_idx}→{sink_coord})")
                unreachable += 1

        reachable = len(valid_nets) - unreachable
        logger.info(f"[REACHABILITY-TRIPWIRE] Results: {reachable} reachable, {unreachable} failed")

        if unreachable > 0:
            logger.warning(f"[REACHABILITY-TRIPWIRE] {unreachable}/{len(valid_nets)} terminals unreachable - routing will fail")


