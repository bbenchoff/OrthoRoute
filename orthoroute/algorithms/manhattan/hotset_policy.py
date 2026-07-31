"""
Hotset selection and stagnation policy for PathFinderRouter.

Extracted verbatim from unified_pathfinder.py (hardening audit, Phase 8
extraction 3). HotsetPolicy is a collaborator operating on the router
instance passed to its constructor; PathFinderRouter keeps thin delegating
methods so every internal and external call site is unchanged.

HOTSET MECHANISM (PREVENTS THRASHING)
═══════════════════════════════════════════════════════════════════════════════

PROBLEM (without hotsets):
• Re-routing ALL 464 nets every iteration takes minutes
• 90% of nets are clean, re-routing them wastes time and risks new conflicts

SOLUTION (adaptive hotsets):
• Iteration 1: Route all nets (initial solution)
• Iteration 2+: Only re-route nets that touch overused edges

HOTSET BUILDING (O(1) via edge-to-nets tracking):
───────────────────────────────────────────────────────────────────────────────
1. Find overused edges: over_idx = {e | present[e] > capacity[e]}
2. Find offending nets: offenders = ⋃(edge_to_nets[e] for e in over_idx)
3. Score by impact: impact[net] = Σ(overuse[e] for e in net_to_edges[net] ∩ over_idx)
4. Adaptive cap: min(hotset_cap, max(64, 3 × |over_idx|))
   • 26 overused edges → hotset ~78 nets (not 418)
   • 500 overused edges → hotset capped at 150

NET-TO-EDGE TRACKING:
• _net_to_edges: Dict[net_id → [edge_indices]] - cached when paths committed
• _edge_to_nets: Dict[edge_idx → {net_ids}] - reverse mapping
• Updated on: commit, clear, rip operations
• Enables O(1) hotset building instead of O(N×E) path scanning

TYPICAL EVOLUTION:
• Iter 1: Route 464 nets → 81 succeed, 514 overused edges
• Iter 2: Hotset 150 nets → 81 succeed, 275 overused edges
• Iter 7: Hotset 150 nets → 81 succeed, 143 overused edges
• Iter 12: Hotset 96 nets → 61 succeed, 29 overused edges (rip event)
• Iter 27: Hotset 64 nets → 73 succeed, 22 overused edges
• Detail pass: Hotset 8 nets, 6 iters → 0 overuse (SUCCESS)
"""

import logging
import random
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class HotsetPolicy:
    """Selects hotsets and decides stagnation ripping.

    Holds no state of its own beyond the router reference: all reads and
    writes go through the router so behavior is identical to the
    pre-extraction methods.
    """

    def __init__(self, router):
        self._router = router

    @staticmethod
    def _history_hotset_cap(total_overuse: float) -> int:
        """Scale reroute waves without using the tail policy globally.

        Small caps protect a nearly-clean route from destructive churn. A
        monster route with tens of thousands of exact node conflicts is a
        different regime: paying the full-graph accounting cost to move only
        100 of 8,192 nets makes tiny minima look like useful convergence.
        Conflict-aware selection below keeps the larger waves one-sided.
        """
        if total_overuse <= 8:
            return 16
        if total_overuse <= 32:
            return 32
        if total_overuse <= 128:
            return 64
        if total_overuse <= 2_048:
            return 100
        if total_overuse <= 16_384:
            return 180
        return 256

    @staticmethod
    def _rolling_progress_insufficient(
        values,
        window: int = 5,
        minimum_fraction: float = 0.025,
        minimum_overuse: int = 16_384,
    ) -> Tuple[bool, Optional[float]]:
        """Return whether the best rolling descent is operationally slow."""
        window = max(1, int(window))
        if len(values) < window + 1:
            return False, None
        start = float(values[-window - 1])
        if start <= max(0, int(minimum_overuse)):
            return False, None
        best_later = min(map(float, values[-window:]))
        improvement_fraction = max(
            0.0,
            (start - best_later) / max(1.0, start),
        )
        threshold = max(0.0, float(minimum_fraction))
        return improvement_fraction < threshold, improvement_fraction

    @staticmethod
    def _pressure_work_scale(
        routed_task_count: int,
        reference_hotset: int = 100,
        maximum_scale: float = 2.0,
    ) -> float:
        """Return bounded equivalent pressure steps for a selective pass."""
        reference = max(1, int(reference_hotset))
        maximum = max(1.0, float(maximum_scale))
        return min(
            maximum,
            max(1.0, max(0, int(routed_task_count)) / reference),
        )

    def _effective_history_hotset_cap(
        self,
        total_overuse: float,
    ) -> int:
        """Apply the temporary rate-based severe-wave expansion."""
        base = self._router._history_hotset_cap(total_overuse)
        severe_threshold = int(getattr(
            self._router.config,
            "slow_progress_min_overuse",
            16_384,
        ))
        if (
            total_overuse > severe_threshold
            and self._router.iteration <= int(getattr(
                self._router,
                "_hotset_rate_boost_until",
                0,
            ))
        ):
            initial_cap = max(
                1,
                int(getattr(
                    self._router.config,
                    "slow_progress_hotset_cap",
                    512,
                )),
            )
            maximum_cap = max(
                initial_cap,
                int(getattr(
                    self._router.config,
                    "slow_progress_hotset_cap_max",
                    initial_cap,
                )),
            )
            growth_after = max(
                1,
                int(getattr(
                    self._router.config,
                    "slow_progress_hotset_growth_after",
                    2,
                )),
            )
            event_count = max(
                1,
                int(getattr(
                    self._router,
                    "_slow_progress_event_count",
                    1,
                )),
            )
            growth_steps = max(0, event_count - growth_after + 1)
            boosted_cap = min(
                maximum_cap,
                initial_cap * (2 ** growth_steps),
            )
            return max(
                base,
                boosted_cap,
            )
        return base

    def _bounded_history_hotset_cap(
        self,
        total_overuse: float,
    ) -> int:
        """Resolve the ordinary cap and an explicit plateau-recovery wave.

        Board parameter derivation may lower ``config.hotset_cap`` (for
        example, to 20% of the net count).  That is the normal-pass safety
        ceiling, not a second ceiling on the separately bounded recovery
        controller.  When recovery deliberately requests a larger wave, its
        own ``slow_progress_hotset_cap_max`` is the applicable guardrail.
        """
        base_cap = self._router._history_hotset_cap(total_overuse)
        effective_cap = self._router._effective_history_hotset_cap(total_overuse)
        configured_cap = max(1, int(self._router.config.hotset_cap))
        if effective_cap > base_cap:
            configured_cap = max(configured_cap, effective_cap)
        return min(configured_cap, effective_cap)

    @staticmethod
    def _hotset_exploration_fraction(total_overuse: float) -> float:
        """Spend less of a severe-congestion wave on random search."""
        if total_overuse > 16_384:
            return 0.15
        if total_overuse > 2_048:
            return 0.25
        return 0.40

    @staticmethod
    def _select_conflict_aware_hotset(
        ranked_candidates: List[str],
        conflict_pairs,
        cap: int,
        exploration_fraction: float,
        rng,
    ) -> List[str]:
        """Select a conflict-covering wave plus bounded exploration.

        _route_all clears and recommits each selected net sequentially, with
        live edge and node occupancy refreshed between nets.  Selecting a
        global independent set is therefore unnecessary: on a dense conflict
        component it can collapse a nominally large wave to only a handful
        of nets.  Instead, greedily cover the greatest number of still-live
        conflict pairs.  After those pairs are covered, prefer candidates
        that do not duplicate a selected conflict endpoint, then fill the
        requested budget.  The final exploratory slice is shuffled but also
        fills its allocation.
        """
        cap = max(0, int(cap))
        if cap == 0 or not ranked_candidates:
            return []

        candidates = list(dict.fromkeys(ranked_candidates))
        cap = min(cap, len(candidates))
        candidate_set = set(candidates)
        rank = {
            candidate: index
            for index, candidate in enumerate(candidates)
        }
        adjacency = defaultdict(set)
        for first, second in conflict_pairs or ():
            if (
                first == second
                or first not in candidate_set
                or second not in candidate_set
            ):
                continue
            adjacency[first].add(second)
            adjacency[second].add(first)

        fraction = min(1.0, max(0.0, float(exploration_fraction)))
        primary_target = max(
            1,
            min(cap, int(round(cap * (1.0 - fraction)))),
        )
        selected = []
        selected_set = set()
        remaining_adjacency = {
            candidate: set(adjacency.get(candidate, ()))
            for candidate in candidates
        }

        # Greedy vertex-cover approximation.  Lazy heap entries make degree
        # updates proportional to the affected conflict pairs instead of
        # rescanning every candidate for every slot.
        import heapq
        degree_heap = [
            (-len(remaining_adjacency[candidate]), rank[candidate], candidate)
            for candidate in candidates
            if remaining_adjacency[candidate]
        ]
        heapq.heapify(degree_heap)
        while len(selected) < primary_target and degree_heap:
            negative_degree, _, candidate = heapq.heappop(degree_heap)
            if candidate in selected_set:
                continue
            live_degree = len(remaining_adjacency[candidate])
            if -negative_degree != live_degree:
                heapq.heappush(
                    degree_heap,
                    (-live_degree, rank[candidate], candidate),
                )
                continue
            if live_degree == 0:
                break
            selected.append(candidate)
            selected_set.add(candidate)
            for neighbor in tuple(remaining_adjacency[candidate]):
                remaining_adjacency[neighbor].discard(candidate)
                heapq.heappush(
                    degree_heap,
                    (
                        -len(remaining_adjacency[neighbor]),
                        rank[neighbor],
                        neighbor,
                    ),
                )
            remaining_adjacency[candidate].clear()

        # Fill the deterministic part with nonredundant candidates first.
        # Edge-only offenders have no adjacency and therefore stay eligible
        # ahead of the opposite endpoint of an already covered conflict.
        if len(selected) < primary_target:
            nonredundant = [
                candidate
                for candidate in candidates
                if candidate not in selected_set
                and not (
                    adjacency.get(candidate, set()) & selected_set
                )
            ]
            nonredundant_set = set(nonredundant)
            redundant = [
                candidate
                for candidate in candidates
                if candidate not in selected_set
                and candidate not in nonredundant_set
            ]
            for candidate in nonredundant + redundant:
                selected.append(candidate)
                selected_set.add(candidate)
                if len(selected) >= primary_target:
                    break

        if len(selected) < cap:
            exploration = [
                candidate
                for candidate in candidates
                if candidate not in selected_set
            ]
            rng.shuffle(exploration)
            selected.extend(exploration[:cap - len(selected)])

        return selected

    @staticmethod
    def _should_rip_for_stagnation(
        spatial_via_overuse: int,
        tail_threshold: int = 8,
        physical_cleanup_started: bool = False,
    ) -> bool:
        """Allow speculative rip-up only before staged physical cleanup."""
        return (
            not physical_cleanup_started
            and spatial_via_overuse <= max(0, int(tail_threshold))
        )

    def _select_physical_hotset(self) -> Set[str]:
        """Return a bounded, severity-ranked physical-conflict wave."""
        offenders = set(getattr(self._router, "_barrel_conflict_nets", ()))
        cap = self._router._physical_hotset_limit(
            int(getattr(self._router, "_last_barrel_conflict_count", 0)),
            max_cap=int(getattr(
                self._router.config, "physical_hotset_cap", 1024
            )),
            min_cap=int(getattr(
                self._router.config, "physical_hotset_min", 64
            )),
            conflicts_per_net=float(getattr(
                self._router.config,
                "physical_conflicts_per_hotset_net",
                50.0,
            )),
        )
        if len(offenders) <= cap:
            return offenders
        scores = getattr(self._router, "_physical_conflict_scores", {})
        ranked = sorted(
            offenders,
            key=lambda net_id: (
                -int(scores.get(net_id, 0)),
                str(net_id),
            ),
        )
        selected = set(ranked[:cap])
        logger.info(
            "[PHYSICAL-HOTSET] Selected %d/%d offenders "
            "(score range %d..%d)",
            len(selected),
            len(offenders),
            int(scores.get(ranked[0], 0)),
            int(scores.get(ranked[cap - 1], 0)),
        )
        return selected

    def _build_hotset(self, tasks: Dict[str, Tuple[int, int]], ripped: Optional[Set[str]] = None) -> Set[str]:
        """
        Build hotset: ONLY nets touching overused edges, with adaptive capping.
        Prevents thrashing by limiting hotset size based on actual overuse.
        Implements freeze-clean: nets clean for 3+ iterations are excluded from hotset.
        """
        if ripped is None:
            ripped = set()

        present = self._router.accounting.present.get() if self._router.accounting.use_gpu else self._router.accounting.present
        cap = self._router.accounting.capacity.get() if self._router.accounting.use_gpu else self._router.accounting.capacity
        over = np.maximum(0, present - cap)
        # History belongs in the Pathfinder routing cost, not in the
        # definition of a live offender.  Once an edge is no longer
        # oversubscribed, selecting every net that merely touches its retained
        # history wastes the bounded random hotset on clean nets and can make
        # full-memory runs diverge.  Rip up nets using resources that are
        # over capacity now; their next shortest-path search still sees the
        # complete historical cost field.
        over_idx = set(map(int, np.flatnonzero(over > 0)))
        via_pool_offenders = self._router._find_via_pool_offenders()
        path_node_scores = {
            net_id: int(score)
            for net_id, score in getattr(
                self._router, "_path_node_conflict_scores", {}
            ).items()
            if net_id in tasks and int(score) > 0
        }
        path_node_offenders = set(path_node_scores)
        path_node_overuse = 0
        if hasattr(self._router, "path_node_use"):
            path_node_overuse = int(np.maximum(
                0,
                np.asarray(self._router.path_node_use) - 1,
            ).sum())
        total_overuse_with_vias = self._router.accounting.compute_overuse(
            router_instance=self._router
        )[0]
        total_negotiated_overuse = (
            total_overuse_with_vias + path_node_overuse
        )
        cleanup_threshold = int(getattr(
            self._router.config,
            "portal_cleanup_edge_threshold",
            3,
        ))

        # Initialize clean iteration tracking
        if not hasattr(self._router, '_net_clean_iters'):
            self._router._net_clean_iters = {}

        freeze_after_clean = int(getattr(self._router.config, 'freeze_after_clean', 3))

        # No edge overuse can still leave an over-capacity via pool.
        if len(over_idx) == 0:
            unrouted = {nid for nid in tasks.keys() if not self._router.net_paths.get(nid)}
            physical_offenders = (
                self._router._select_physical_hotset()
                if total_negotiated_overuse <= cleanup_threshold
                else set()
            )
            hotset = (
                unrouted
                | ripped
                | via_pool_offenders
                | physical_offenders
            )
            node_cap = self._router._bounded_history_hotset_cap(
                total_negotiated_overuse
            )
            node_exploration_fraction = (
                self._router._hotset_exploration_fraction(
                    total_negotiated_overuse
                )
            )
            import random
            hotset.update(self._router._select_conflict_aware_hotset(
                sorted(
                    path_node_offenders,
                    key=lambda net_id: (
                        -path_node_scores[net_id],
                        str(net_id),
                    ),
                ),
                getattr(self._router, "_path_node_conflict_pairs", ()),
                node_cap,
                node_exploration_fraction,
                random.Random(42 + self._router.iteration),
            ))
            logger.info(
                f"[HOTSET] no-edge-overuse; unrouted={len(unrouted)} "
                f"ripped={len(ripped)} via_pool={len(via_pool_offenders)} "
                f"physical={len(physical_offenders)} "
                f"→ hotset={len(hotset)}"
            )
            self._router._last_hotset_size = len(hotset)
            self._router._last_hotset_cap = node_cap
            self._router._last_hotset_offender_count = len(
                path_node_offenders
            )
            self._router._last_hotset_exploration_fraction = (
                node_exploration_fraction
            )
            self._router._last_hotset_conflict_aware = True
            self._router._record_hotset_conflict_coverage(hotset)
            return hotset

        # OVERUSE EXISTS: collect nets touching overused edges using fast lookup
        offenders = set()
        for ei in over_idx:
            offenders.update(self._router._edge_to_nets.get(ei, set()))
        # Guided H/V routing can cross at a capacity-one lattice node without
        # sharing an edge. Negotiate those node-only offenders concurrently
        # with edge congestion instead of deferring them to final cleanup.
        offenders.update(path_node_offenders)

        # Update clean iteration counters
        for net_id in tasks.keys():
            if net_id in offenders:
                # Net is touching overused edges - reset counter
                self._router._net_clean_iters[net_id] = 0
            else:
                # Net is clean - increment counter
                self._router._net_clean_iters[net_id] = self._router._net_clean_iters.get(net_id, 0) + 1

        # Filter out frozen nets (clean for freeze_after_clean+ iterations)
        frozen_nets = {nid for nid in offenders if self._router._net_clean_iters.get(nid, 0) >= freeze_after_clean}
        offenders -= frozen_nets

        if frozen_nets:
            logger.debug(f"[FREEZE-CLEAN] Excluded {len(frozen_nets)} nets clean for {freeze_after_clean}+ iterations")

        # Add ripped nets
        offenders |= ripped
        offenders |= via_pool_offenders

        # Add unrouted nets (small priority, will be at end after sorting)
        unrouted = {nid for nid in tasks.keys() if not self._router.net_paths.get(nid)}

        # Score offenders by total overuse they contribute
        scores = []
        for net_id in offenders:
            impact = float(path_node_scores.get(net_id, 0))
            if net_id in self._router._net_to_edges:
                impact += sum(
                    float(over[ei])
                    for ei in self._router._net_to_edges[net_id]
                    if ei in over_idx
                )
            scores.append((impact, net_id))

        # Add unrouted with low priority
        for net_id in unrouted:
            if net_id not in offenders:
                scores.append((0.0, net_id))

        # Sort by impact (highest first)
        scores.sort(reverse=True)

        # Scale with the complete edge/via + node residual. A small tail needs
        # cautious waves; severe monster-board congestion needs enough work
        # per pass to amortize full-graph accounting.
        total_overuse = sum(float(over[ei]) for ei in over_idx)

        # Preserve small tail waves, but do not apply the 100-net tail policy
        # to a monster route with tens of thousands of live node conflicts.
        # Exact physical offenders and unrouted nets still bypass this cap.
        adaptive_cap = self._router._bounded_history_hotset_cap(
            total_negotiated_overuse
        )

        # Severe congestion needs predominantly high-impact work. Retain a
        # bounded random fraction to break phase-locking, then select an
        # independent set of current path-node conflicts so both sides do not
        # move together and exchange ownership.
        import random
        rng = random.Random(42 + self._router.iteration)

        # Cooldown: exclude nets rerouted in previous iteration (prevents immediate re-routing)
        if not hasattr(self._router, '_last_reroute_iter'):
            self._router._last_reroute_iter = {}

        ranked_with_cooldown = [
            net_id
            for _, net_id in scores
            if (
                self._router.iteration
                - self._router._last_reroute_iter.get(net_id, -999)
                > 1
            )
        ]
        exploration_fraction = self._router._hotset_exploration_fraction(
            total_negotiated_overuse
        )
        hotset_with_cooldown = self._router._select_conflict_aware_hotset(
            ranked_with_cooldown,
            getattr(self._router, "_path_node_conflict_pairs", ()),
            adaptive_cap,
            exploration_fraction,
            rng,
        )

        # Update last reroute iteration for selected nets
        for nid in hotset_with_cooldown:
            self._router._last_reroute_iter[nid] = self._router.iteration

        hotset = set(hotset_with_cooldown)
        raw_overuse_edges = int(np.count_nonzero(over > 0))
        if (
            raw_overuse_edges <= cleanup_threshold
            and total_negotiated_overuse <= cleanup_threshold
        ):
            physical_offenders = self._router._select_physical_hotset()
            # The adaptive cap is for ordinary edge congestion. Once the graph
            # tail is clean enough for physical cleanup, exact shorts bypass
            # that cap so one-sided repair can move every selected component.
            hotset.update(physical_offenders)
        # A failed full-graph search has no committed edges and therefore no
        # congestion score. It must bypass both caps and cooldowns or it can
        # remain unrouted indefinitely.
        hotset.update(unrouted)

        unique_frac = len(hotset - getattr(self._router, '_prev_hotset', set())) / max(1, len(hotset))
        self._router._prev_hotset = hotset.copy()
        self._router._last_hotset_size = len(hotset)
        self._router._last_hotset_cap = adaptive_cap
        self._router._last_hotset_offender_count = len(offenders)
        self._router._last_hotset_exploration_fraction = (
            exploration_fraction
        )
        self._router._last_hotset_conflict_aware = True
        self._router._record_hotset_conflict_coverage(hotset)

        logger.info(f"[HOTSET] overuse_edges={len(over_idx)} total_overuse={int(total_overuse)}, "
                    f"offenders={len(offenders)}, cap={adaptive_cap} → hotset={len(hotset)}/{len(tasks)} "
                    f"(explore={exploration_fraction:.0%}, "
                    f"conflict-aware, unique={unique_frac:.1%}, "
                    f"pair-cover="
                    f"{self._router._last_hotset_conflict_pair_coverage_fraction:.1%})")

        return hotset

    def _rank_stagnation_offenders(
        self,
        over: np.ndarray,
    ) -> List[Tuple[float, str]]:
        """Rank live offenders by the complete negotiated objective."""
        canonical_mask = self._router._canonical_edge_resource_mask()
        over_idx = set(map(
            int,
            np.flatnonzero((over > 0) & canonical_mask),
        ))
        _, _, measured_node_scores = (
            self._router._detect_path_node_conflicts()
        )
        self._router._path_node_conflict_scores = dict(measured_node_scores)

        candidates = set(measured_node_scores)
        for edge_idx in over_idx:
            candidates.update(self._router._edge_to_nets.get(edge_idx, ()))

        scores = []
        for net_id in candidates:
            if (
                not self._router.net_paths.get(net_id)
                or net_id in self._router.locked_nets
            ):
                continue
            impact = float(measured_node_scores.get(net_id, 0))
            if net_id in self._router._net_to_edges:
                impact += sum(
                    float(over[edge_idx])
                    for edge_idx in self._router._net_to_edges[net_id]
                    if edge_idx in over_idx
                )
            if impact > 0:
                scores.append((impact, net_id))
        return sorted(
            scores,
            key=lambda item: (-item[0], str(item[1])),
        )

    def _rip_top_k_offenders(self, k=20) -> Set[str]:
        """
        Rip only the worst 16-24 nets to break stagnation (not the world).
        Respect locked nets - don't rip unless they touch new overuse.
        Returns the set of ripped net IDs.
        """
        present = self._router.accounting.present.get() if self._router.accounting.use_gpu else self._router.accounting.present
        cap = self._router.accounting.capacity.get() if self._router.accounting.use_gpu else self._router.accounting.capacity
        over = np.maximum(0, present - cap)
        scores = self._router._rank_stagnation_offenders(over)
        victims = self._router._select_stagnation_victims(scores, k)

        for net_id in victims:
            if self._router.net_paths.get(net_id) and net_id in self._router._net_to_edges:
                old_path = self._router.net_paths[net_id]
                # Use cached edges for efficiency
                self._router.accounting.clear_path(self._router._net_to_edges[net_id])
                self._router._clear_via_barrel_ownership_for_path(
                    net_id, old_path
                )
                self._router._clear_path_node_use(old_path)
                self._router._clear_escape_occupancy(net_id)
                self._router.net_paths[net_id] = []
                self._router.net_selected_portals.pop(net_id, None)
                # Clear edge tracking for ripped nets
                self._router._clear_net_edge_tracking(net_id)
                # Reset clean streak so they can't immediately lock again
                self._router.net_clean_streak[net_id] = 0

        logger.info(f"[STAGNATION] Ripped {len(victims)} nets (locked={len(self._router.locked_nets)} preserved)")
        return victims

    def _select_stagnation_victims(
        self,
        scores: Sequence[Tuple[float, str]],
        k: int,
    ) -> Set[str]:
        """Rotate a fixed-width recovery window around one retained best."""
        ranked = [net_id for _, net_id in scores]
        if not ranked or k <= 0:
            return set()

        # A cleared history denotes a newly retained best basin (and remains
        # useful to lightweight test fixtures that do not run __init__).
        if not getattr(self._router, "_stagnation_victim_history", set()):
            self._router._stagnation_victim_cursor = 0

        count = min(int(k), len(ranked))
        start = int(getattr(
            self._router, "_stagnation_victim_cursor", 0
        )) % len(ranked)
        selected = [
            ranked[(start + offset) % len(ranked)]
            for offset in range(count)
        ]
        self._router._stagnation_victim_cursor = (
            start + count
        ) % len(ranked)
        victims = set(selected)
        self._router._stagnation_victim_history = victims
        return victims
