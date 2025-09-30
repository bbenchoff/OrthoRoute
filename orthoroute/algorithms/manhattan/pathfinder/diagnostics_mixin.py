"""
Diagnostics Mixin - Extracted from UnifiedPathFinder

This module contains diagnostics mixin functionality.
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
from ....domain.models.board import Board, Pad

logger = logging.getLogger(__name__)


class DiagnosticsMixin:
    """
    Diagnostics functionality for UnifiedPathFinder.

    This mixin is designed to be used with multiple inheritance.
    It expects the following attributes to be available on self:
    - config: PathFinderConfig instance
    - use_gpu: bool indicating GPU availability
    - node_count: int number of nodes
    - edges: list of edges
    - nodes: dict of node data
    """

    def _csr_smoke_check(self):
        """Validate CSR adjacency matrix structure and dimensions."""
        try:
            N = getattr(self, "lattice_node_count", None) or getattr(self, "node_count", None)
            assert getattr(self, "adjacency_matrix", None) is not None, "[CSR] adjacency_matrix not built"
            if N is not None:
                assert self.adjacency_matrix.shape == (N, N), f"[CSR] shape mismatch: {self.adjacency_matrix.shape} vs nodes={N}"
        except Exception as e:
            logger.error("[CSR] smoke check failed: %s", e)

    # ========================================================================
    # Edge Usage Tracking and Storage
    # ========================================================================


    def _check_overuse_invariant(self, where: str = "", compare_to_store: bool = True):
        """Validate consistency between present usage and store overuse calculations.

        Args:
            where: Context string for debugging if invariant fails
            compare_to_store: Whether to compare present vs store (False during batch commits)

        Returns:
            Tuple of (present_overuse_sum, present_overuse_edges)

        Raises:
            RuntimeError: If overuse calculations are inconsistent
        """
        pres_s, pres_e = self._compute_overuse_from_present()
        logger.info("[UPF] Overuse: sum=%d edges=%d (from %d total edges)", pres_s, pres_e, self._live_edge_count())

        if compare_to_store:
            store_s, store_e = self._compute_overuse_from_store()
            logger.info("[UPF] Store overuse: sum=%d edges=%d (from %d store edges)", store_s, store_e, len(getattr(self, '_edge_store', {}) or {}))
            if pres_s != store_s:
                logger.error("[INVARIANT] Overuse sum mismatch: calc=%d vs reported=%d at %s", store_s, pres_s, where)
                logger.warning("[INVARIANT-RECOVERY] Rebuilding present usage from canonical store")
                self._refresh_present_usage_from_store()
                pres_s2, _ = self._compute_overuse_from_present()
                if pres_s2 != store_s:
                    raise RuntimeError(f"Overuse invariant did not recover at {where}: store={store_s} present={pres_s2}")
                logger.info("[INVARIANT-RECOVERY] Present usage rebuilt successfully")
        return pres_s, pres_e


    def _log_build_sanity_checks(self, layer_count: int):
        """Log critical build parameters for debugging"""
        logger.info("=== SANITY CHECKS/LOGS ===")

        # Layer count, names, and HV mask
        hv_summary = ", ".join([f"{name}={pol.upper()}" for name, pol in zip(self.config.layer_names, self.hv_polarity)])
        logger.info(f"Layer configuration: count={layer_count}, HV_mask=[{hv_summary}]")

        # Allowed vertical pairs (first 10)
        if hasattr(self, 'allowed_layer_pairs'):
            pair_count = len(self.allowed_layer_pairs)
            logger.info(f"Allowed vertical transitions: {pair_count} pairs")
            sample_pairs = list(sorted(self.allowed_layer_pairs))[:5]
            for from_l, to_l in sample_pairs:
                from_name = self.config.layer_names[from_l] if from_l < len(self.config.layer_names) else f"L{from_l}"
                to_name = self.config.layer_names[to_l] if to_l < len(self.config.layer_names) else f"L{to_l}"
                logger.info(f"  Via transition: {from_name} -> {to_name}")
            if pair_count > 5:
                logger.info(f"  ... and {pair_count-5} more transitions")

        # Via cost and caps
        VIA_COST_LOCAL = VIA_COST
        VIA_CAP = VIA_CAPACITY_PER_NET
        logger.info(f"Via configuration: cost={VIA_COST}, cap_per_net={VIA_CAP}")

        # Grid and bounds
        logger.info(f"Grid pitch: {self.config.grid_pitch}mm")
        logger.info(f"Occupancy grids: {len(self.occ)} layers with cell_size={self.occ[0].cell_size:.3f}mm")

        logger.info("=== BUILD COMPLETE ===")

    # ========================================================================
    # Debug and Utility Methods
    # ========================================================================


    def _log_first_routed_nets_debug(self, net_count: int = 3):
        """Log debug info for first few routed nets"""
        if not hasattr(self, '_nets_routed_debug'):
            self._nets_routed_debug = 0

        if self._nets_routed_debug < net_count:
            self._nets_routed_debug += 1
            logger.info(f"=== DEBUG NET {self._nets_routed_debug}/{net_count} ===")
            logger.info("Debug info: pad-stub analysis for first few routed nets")


    def _log_first_illegal_expansion(self, cause: str, net_x: str = "", layer_y: int = -1):
        """Log first illegal expansion cause for debugging"""
        if not hasattr(self, '_logged_first_illegal'):
            self._logged_first_illegal = True
            logger.error(f"FIRST ILLEGAL EXPANSION: {cause}")
            if net_x:
                logger.error(f"  Blocked by net '{net_x}' on layer {layer_y}")


    def _adaptive_delta_tuning(self, iteration_success_rate: float, routing_time_ms: float):
        """Adaptive delta tuning based on performance feedback"""
        if not self.config.adaptive_delta:
            return
        
        # Track performance with current delta
        if not hasattr(self, '_delta_performance_history'):
            self._delta_performance_history = []
        self._delta_performance_history.append({
            'delta_mult': self._adaptive_delta,
            'success_rate': iteration_success_rate,
            'routing_time_ms': routing_time_ms,
            'performance_score': iteration_success_rate / max(1.0, routing_time_ms / 1000.0)  # success per second
        })
        
        # Tune delta every few iterations based on performance trends
        if len(self._delta_performance_history) >= 2:
            current_score = self._delta_performance_history[-1]['performance_score']
            previous_score = self._delta_performance_history[-2]['performance_score']
            
            old_delta = self._adaptive_delta
            
            # Adaptive logic: increase delta if performance is good, decrease if poor
            if current_score > previous_score * 1.1:  # 10% better performance
                self._adaptive_delta = min(self._adaptive_delta * 1.2, 8.0)  # Increase delta (max 8x)
                reason = "performance_improvement"
            elif current_score < previous_score * 0.9:  # 10% worse performance  
                self._adaptive_delta = max(self._adaptive_delta * 0.8, 2.0)  # Decrease delta (min 2x)
                reason = "performance_degradation"
            else:
                return  # No significant change
            
            if old_delta != self._adaptive_delta:
                logger.info(f"[ADAPTIVE DELTA]: {old_delta:.1f}x -> {self._adaptive_delta:.1f}x ({reason})")
                logger.info(f"   Performance score: {current_score:.3f} vs {previous_score:.3f}")
                
                # Keep history manageable
                if len(self._delta_performance_history) > 10:
                    self._delta_performance_history = self._delta_performance_history[-10:]
    

    def _analyze_warp_divergence(self, kernel_metrics: Dict, packed_data: Dict):
        """Analyze warp divergence patterns for optimization"""
        K = kernel_metrics['K']
        block_dim = kernel_metrics['block_dim'][0]  # Threads per block
        
        # Calculate potential divergence sources
        roi_sizes = []
        for i, meta in enumerate(packed_data['roi_metadata']):
            roi_sizes.append(meta['nodes'])
        
        # Analyze size distribution (indicates divergence potential)
        if len(roi_sizes) > 1:
            size_variance = np.var(roi_sizes)
            size_mean = np.mean(roi_sizes) 
            coefficient_of_variation = np.sqrt(size_variance) / size_mean if size_mean > 0 else 0
            
            # Warp efficiency analysis
            threads_per_roi = block_dim
            actual_work_per_roi = [min(threads_per_roi, size) for size in roi_sizes]
            warp_efficiency = np.mean(actual_work_per_roi) / block_dim if block_dim > 0 else 0
            
            warp_analysis = {
                'timestamp': time.time(),
                'roi_size_cv': coefficient_of_variation,
                'warp_efficiency': warp_efficiency,
                'divergence_risk': 'HIGH' if coefficient_of_variation > 0.5 else 'MEDIUM' if coefficient_of_variation > 0.2 else 'LOW',
                'optimization_suggestion': self._suggest_warp_optimization(coefficient_of_variation, warp_efficiency)
            }
            
            self._warp_stats.append(warp_analysis)
            
            logger.debug(f"[WARP ANALYSIS]: efficiency={warp_efficiency:.1%}, "
                        f"divergence_risk={warp_analysis['divergence_risk']}")
            
            if warp_analysis['divergence_risk'] == 'HIGH':
                logger.warning(f"[WARNING]: HIGH warp divergence risk detected (CV={coefficient_of_variation:.2f})")
                logger.info(f"[OPTIMIZATION]: {warp_analysis['optimization_suggestion']}")
    

    def _suggest_warp_optimization(self, cv: float, efficiency: float) -> str:
        """Suggest warp optimization strategies"""
        if cv > 0.5 and efficiency < 0.6:
            return "Consider ROI size balancing or dynamic block sizing"
        elif cv > 0.3:
            return "Consider sorting ROIs by size for better warp utilization"
        elif efficiency < 0.7:
            return "Consider reducing threads per block or increasing work per thread"
        else:
            return "Warp utilization is acceptable"
    

    def _export_instrumentation_csv(self):
        """Export instrumentation data to CSV files for convergence analysis"""
        if not getattr(self, '_instrumentation', None):
            return
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_path = self.config.csv_export_path.replace('.csv', f'_{timestamp}')
            
            # Export iteration-level metrics
            iteration_csv = base_path.replace('.csv', '_iterations.csv')
            with open(iteration_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iteration', 'timestamp', 'success_rate_pct', 'overuse_violations', 
                    'max_overuse', 'avg_overuse', 'pres_fac', 'acc_fac', 'routes_changed',
                    'total_nets', 'successful_nets', 'failed_nets', 'iteration_time_ms',
                    'delta_value', 'congestion_penalty'
                ])
                
                for metric in self._instrumentation.iteration_metrics:
                    writer.writerow([
                        metric.iteration, metric.timestamp, metric.success_rate,
                        metric.overuse_violations, metric.max_overuse, metric.avg_overuse,
                        metric.pres_fac, metric.acc_fac, metric.routes_changed,
                        metric.total_nets, metric.successful_nets, metric.failed_nets,
                        metric.iteration_time_ms, metric.delta_value, metric.congestion_penalty
                    ])
            
            # Export ROI batch metrics
            if self._instrumentation.roi_batch_metrics:
                roi_csv = base_path.replace('.csv', '_roi_batches.csv')
                with open(roi_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'batch_timestamp', 'batch_size', 'avg_roi_nodes', 'avg_roi_edges',
                        'min_roi_size', 'max_roi_size', 'compression_ratio',
                        'memory_efficiency', 'parallel_factor', 'total_processing_time_ms'
                    ])
                    
                    for metric in self._instrumentation.roi_batch_metrics:
                        writer.writerow([
                            metric.batch_timestamp, metric.batch_size, metric.avg_roi_nodes,
                            metric.avg_roi_edges, metric.min_roi_size, metric.max_roi_size,
                            metric.compression_ratio, metric.memory_efficiency,
                            metric.parallel_factor, metric.total_processing_time_ms
                        ])
            
            # Export per-net timing metrics
            if self._instrumentation.net_timing_metrics:
                net_csv = base_path.replace('.csv', '_net_timings.csv')
                with open(net_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'net_id', 'timestamp', 'routing_time_ms', 'success', 'path_length',
                        'iterations_used', 'roi_nodes', 'roi_edges', 'search_nodes_visited'
                    ])
                    
                    for metric in self._instrumentation.net_timing_metrics:
                        writer.writerow([
                            metric.net_id, metric.timestamp, metric.routing_time_ms,
                            metric.success, metric.path_length, metric.iterations_used,
                            metric.roi_nodes, metric.roi_edges, metric.search_nodes_visited
                        ])
            
            # Export session metadata
            metadata_csv = base_path.replace('.csv', '_metadata.csv')
            with open(metadata_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['key', 'value'])
                for key, value in self._instrumentation.session_metadata.items():
                    writer.writerow([key, str(value)])
            
            logger.info(f"[INSTRUMENTATION]: CSV data exported to {iteration_csv} and related files")
            
            # Update GUI with export status
            if self._gui_status_callback:
                self._gui_status_callback(f"CSV metrics exported: {len(self._instrumentation.iteration_metrics)} iterations, {len(self._instrumentation.net_timing_metrics)} nets")
        
        except Exception as e:
            logger.error(f"Failed to export CSV instrumentation: {e}")
    

    def get_instrumentation_summary(self) -> Dict[str, Any]:
        """Get a summary of instrumentation data for display"""
        if not self._instrumentation or not self._instrumentation.iteration_metrics:
            return {}
        
        last_iteration = self._instrumentation.iteration_metrics[-1]
        
        return {
            'session_id': self._current_session_id,
            'total_iterations': len(self._instrumentation.iteration_metrics),
            'final_success_rate': last_iteration.success_rate,
            'final_violations': last_iteration.overuse_violations,
            'total_nets_processed': len(self._instrumentation.net_timing_metrics),
            'successful_nets': sum(1 for net in self._instrumentation.net_timing_metrics if net.success),
            'avg_routing_time_ms': sum(net.routing_time_ms for net in self._instrumentation.net_timing_metrics) / max(1, len(self._instrumentation.net_timing_metrics)),
            'roi_batches_processed': len(self._instrumentation.roi_batch_metrics)
        }
    
    # ============================================================================
    # ZERO-COPY DEVICE-ONLY GPU OPTIMIZATIONS
    # ============================================================================
    

    def _finalize_insufficient_layers(self):
        """Handle routing failure due to insufficient layer count."""
        # Rebuild present usage from store for consistent numbers
        self._refresh_present_usage_from_store()
        over_sum, over_edges = self._compute_overuse_stats_present()
        failed_nets = len([net for net in self.routed_nets.keys() if not self.routed_nets[net]])

        # Compute overuse array for robust shortfall estimation
        import numpy as np
        cap = self.edge_capacity
        usage = self.edge_present_usage
        if hasattr(cap, 'get'): cap = cap.get()
        if hasattr(usage, 'get'): usage = usage.get()
        over_array = np.maximum(0, usage - cap)

        analysis = self._estimate_layer_shortfall(over_array)

        shortfall = analysis["shortfall"]
        error_code = analysis["error_code"]
        via_edges = analysis["via_overuse_edges"]
        hv_edges = analysis["hv_overuse_edges"]
        via_frac = analysis["via_overuse_frac"]

        # Clear logging of what we're analyzing
        logger.info("[CAP-ANALYZE] over_edges=%d via_edges=%d (%.1f%%) hv_edges=%d (%.1f%%) pairs_est=%d layers=%d",
                    int(over_edges), via_edges, via_frac * 100, hv_edges, (1 - via_frac) * 100,
                    analysis["pairs_est"], shortfall)

        # Generate appropriate message based on analysis
        if error_code == "VIA-BOTTLENECK":
            message = ("Routing failed due to via congestion; adding layers won't help. "
                      f"Via bottleneck: {via_edges} overfull vias vs {hv_edges} routing segments. "
                      "Consider smaller drills/annular rings, microvias/HDI, or relaxing via-to-via clearances.")
        elif error_code == "INSUFFICIENT-LAYERS":
            message = (f"[INSUFFICIENT-LAYERS] Unrouted={failed_nets}, "
                      f"overuse_edges={int(over_edges)}, over_sum={int(over_sum)}. "
                      f"Estimated additional layers needed: {shortfall}. "
                      f"Increase layer count or relax design rules.")
        else:
            message = f"Routing failed. Unrouted={failed_nets}, overuse_edges={int(over_edges)}, over_sum={int(over_sum)}."

        rr = {
            "success": False,
            "error_code": error_code or "ROUTING-FAILED",
            "message": message,
            "unrouted": failed_nets,
            "overuse_edges": int(over_edges),
            "overuse_sum": int(over_sum),
            "layer_shortfall": shortfall,
            "via_overuse_edges": via_edges,
            "hv_overuse_edges": hv_edges,
            "via_overuse_frac": via_frac,
            "h_need": analysis["h_need"],
            "v_need": analysis["v_need"]
        }

        logger.warning(rr["message"])
        self._routing_result = rr
        return rr


    def _estimate_layer_shortfall(self, over_array):
        """
        Aggregate overuse per spatial channel across all layers, count vias separately,
        detect via bottlenecks vs layer congestion, return analysis breakdown.

        Returns: dict with shortfall analysis and error classification
        """
        import numpy as np
        from collections import defaultdict

        # Configurable knobs
        pct   = getattr(self.config, "layer_shortfall_percentile", 95)  # 90, 95, …
        cap_n = getattr(self.config, "layer_shortfall_cap", 16)         # hard cap

        over = over_array.get() if hasattr(over_array, "get") else np.asarray(over_array)
        if over.size == 0 or np.count_nonzero(over) == 0:
            return {"shortfall": 0, "error_code": None, "via_overuse_edges": 0, "hv_overuse_edges": 0}

        indptr  = getattr(self, 'indptr_cpu', None)
        indices = getattr(self, 'indices_cpu', None)

        if indptr is None or indices is None:
            logger.warning("[SHORTFALL] CSR arrays not available, using fallback estimate")
            return {"shortfall": 2, "error_code": "INSUFFICIENT-LAYERS", "via_overuse_edges": 0, "hv_overuse_edges": 0}

        # Precompute source row for each edge index once if available
        edge_src = getattr(self, "edge_src_cpu", None)
        if edge_src is None or len(edge_src) != len(indices):
            edge_src = np.repeat(np.arange(len(indptr) - 1, dtype=np.int32),
                                 np.diff(indptr).astype(np.int32))
            self.edge_src_cpu = edge_src

        totals_H = defaultdict(int)
        totals_V = defaultdict(int)
        via_overuse_edges = 0
        hv_overuse_edges = 0

        nz_idx = np.nonzero(over)[0]
        for eidx in nz_idx:
            u = int(edge_src[eidx])
            v = int(indices[eidx])

            # Undirected dedupe: process each physical segment once
            if u > v:
                continue

            x1, y1, z1 = self._idx_to_coord(u)
            x2, y2, z2 = self._idx_to_coord(v)

            val = int(over[eidx])

            # Count via vs H/V breakdown
            if z1 != z2:
                via_overuse_edges += 1
                continue  # Skip vias for layer capacity analysis
            else:
                hv_overuse_edges += 1

            if y1 == y2 and x1 != x2:          # Horizontal
                a = (min(x1, x2), y1)
                b = (max(x1, x2), y1)
                totals_H[(a, b)] += val
            elif x1 == x2 and y1 != y2:        # Vertical
                a = (x1, min(y1, y2))
                b = (x1, max(y1, y2))
                totals_V[(a, b)] += val

        def need_from_totals(totals: dict) -> int:
            if not totals:
                return 0
            arr = np.fromiter(totals.values(), dtype=np.int32)
            return int(np.ceil(np.percentile(arr, pct)))

        need_H = need_from_totals(totals_H)
        need_V = need_from_totals(totals_V)

        pairs = max(need_H, need_V)
        extra_layers = 2 * pairs if pairs > 0 else 0
        extra_layers = int(np.clip(extra_layers, 0, cap_n))

        # Determine error classification
        total_overuse = via_overuse_edges + hv_overuse_edges
        via_overuse_frac = via_overuse_edges / max(1, total_overuse)

        if hv_overuse_edges == 0 and via_overuse_edges > 0:
            error_code = "VIA-BOTTLENECK"
            shortfall = 0
        elif via_overuse_frac > 0.9:  # >90% via bottleneck
            error_code = "VIA-BOTTLENECK"
            shortfall = 0
        else:
            error_code = "INSUFFICIENT-LAYERS" if extra_layers > 0 else None
            shortfall = extra_layers

        return {
            "shortfall": shortfall,
            "error_code": error_code,
            "via_overuse_edges": via_overuse_edges,
            "hv_overuse_edges": hv_overuse_edges,
            "via_overuse_frac": via_overuse_frac,
            "h_need": need_H,
            "v_need": need_V,
            "pairs_est": pairs
        }


    def _finalize_success(self):
        """Handle successful routing completion."""
        logger.info("[NEGOTIATE] Converged: all nets routed with legal usage.")
        self._routing_result = {'success': True, 'needs_more_layers': False}
        return self.routed_nets


    def _count_failed_nets_last_iter(self):
        """Count nets that failed to route in the last iteration."""
        if not hasattr(self, 'routed_nets'):
            return 0
        return len([net for net, path in self.routed_nets.items() if not path])


    def _log_top_congested_nets(self, k=20):
        if not getattr(self, "_net_paths", None):
            return
        cap = int(getattr(self, "_edge_capacity", 1)) or 1
        pressure = []
        for net_id, path in self._net_paths.items():
            if self._is_empty_path(path):
                continue
            over_sum = 0
            for (u, v) in path:
                idx = self.edge_lookup.get((u, v)) or self.edge_lookup.get((v, u))
                if idx is not None:
                    over = max(0, int(self.edge_present_usage[idx]) - cap)
                    over_sum += over
            if over_sum:
                pressure.append((over_sum, net_id))
        pressure.sort(reverse=True)
        top = pressure[:k]
        if top:
            logger.info("[HOT-NETS] top=%d: %s", len(top), ", ".join(f"{n}:{s}" for s, n in top))


