"""
Layer Analyzer - Determines if board is routable with current layer count
"""
import logging
from typing import Tuple, Dict, List
from collections import defaultdict

logger = logging.getLogger(__name__)


class LayerAnalyzer:
    """Analyzes routing layer utilization and provides recommendations"""

    def __init__(self, layer_count: int):
        self.layer_count = layer_count
        self.layer_usage = defaultdict(int)  # edges used per layer
        self.layer_capacity = defaultdict(int)  # total capacity per layer

    def analyze_routing(self, accounting, graph) -> Dict:
        """Analyze current routing state and determine if more layers needed"""

        # Calculate per-layer statistics
        layer_stats = {}
        total_overuse = 0
        total_edges = 0

        for layer_idx in range(self.layer_count):
            # Get edges for this layer (filter by layer in graph structure)
            layer_edges = self._get_layer_edges(layer_idx, graph)
            layer_overuse = 0
            layer_used = 0

            for edge_id in layer_edges:
                usage = accounting.history.get(edge_id, 0)
                if usage > 0:
                    layer_used += 1
                if usage > 1:
                    layer_overuse += (usage - 1)

            layer_stats[layer_idx] = {
                'total_edges': len(layer_edges),
                'used_edges': layer_used,
                'overuse': layer_overuse,
                'utilization': layer_used / len(layer_edges) if layer_edges else 0
            }

            total_overuse += layer_overuse
            total_edges += len(layer_edges)

        # Determine if more layers needed
        needs_more_layers = self._calculate_layer_recommendation(layer_stats, total_overuse)

        return {
            'layer_stats': layer_stats,
            'total_overuse': total_overuse,
            'needs_more_layers': needs_more_layers['needed'],
            'recommended_layers': needs_more_layers['recommended_count'],
            'reason': needs_more_layers['reason']
        }

    def _get_layer_edges(self, layer_idx: int, graph, lattice) -> List:
        """Get all edges belonging to a specific layer"""
        layer_edges = []

        # Iterate through graph CSR structure and filter by layer
        for src_node in range(len(graph.indptr) - 1):
            # Get layer of source node
            if hasattr(lattice, 'idx_to_coord'):
                try:
                    _, _, src_z = lattice.idx_to_coord(src_node)
                    if src_z == layer_idx:
                        # Add horizontal/vertical edges on this layer
                        start_idx = graph.indptr[src_node]
                        end_idx = graph.indptr[src_node + 1]
                        for edge_idx in range(start_idx, end_idx):
                            dst_node = graph.indices[edge_idx]
                            _, _, dst_z = lattice.idx_to_coord(dst_node)
                            # Same-layer edges (H/V tracks)
                            if dst_z == layer_idx:
                                layer_edges.append(edge_idx)
                except:
                    pass

        return layer_edges

    def _calculate_layer_recommendation(self, layer_stats: Dict, total_overuse: int) -> Dict:
        """Calculate if more layers are needed and how many"""

        # Strategy: If overuse is concentrated and all layers are >80% utilized,
        # we likely need more layers

        high_util_layers = sum(1 for stats in layer_stats.values()
                              if stats['utilization'] > 0.8)
        avg_utilization = sum(stats['utilization'] for stats in layer_stats.values()) / len(layer_stats)

        if total_overuse > 100 and high_util_layers > self.layer_count * 0.7:
            # Many layers are saturated and we have significant overuse
            # Estimate additional layers needed based on overuse density
            overuse_per_layer = total_overuse / self.layer_count
            additional_layers = int(overuse_per_layer / 50) + 2  # Conservative estimate

            return {
                'needed': True,
                'recommended_count': self.layer_count + additional_layers,
                'reason': f'High layer utilization ({avg_utilization:.1%} avg) with {total_overuse} overused edges. ' +
                         f'{high_util_layers}/{self.layer_count} layers >80% full.'
            }
        elif total_overuse > 200:
            # Severe overuse even with low utilization suggests routing congestion
            return {
                'needed': True,
                'recommended_count': self.layer_count + 4,
                'reason': f'Severe congestion: {total_overuse} overused edges despite available space. ' +
                         'May need more layers for vertical routing options.'
            }
        else:
            return {
                'needed': False,
                'recommended_count': self.layer_count,
                'reason': f'Current layer count ({self.layer_count}) appears sufficient. ' +
                         f'Overuse: {total_overuse}, avg utilization: {avg_utilization:.1%}'
            }
