"""
Board Analysis Module

Analyzes PCB characteristics to derive optimal routing parameters.
Computes capacity, demand, congestion ratio, and layer assignments.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class BoardCharacteristics:
    """Physical and topological properties of the board"""

    # Layer configuration
    signal_layers: List[int]           # [1, 2, 3, ..., N]
    h_layers: Set[int]                 # Layers designated for horizontal routing
    v_layers: Set[int]                 # Layers designated for vertical routing
    layer_count: int                   # Total signal layers
    plane_layers: Set[int]             # Power/ground planes (not routed)

    # Physical dimensions
    board_width_mm: float
    board_height_mm: float
    usable_area_mm2: float             # After subtracting keepouts
    grid_pitch_mm: float

    # Routing capacity
    total_horizontal_channels: int     # Available H routing channels
    total_vertical_channels: int       # Available V routing channels
    channels_per_h_layer: int          # Avg channels per H layer
    channels_per_v_layer: int          # Avg channels per V layer
    via_capacity_estimate: int         # Estimated via locations

    # Routing demand
    net_count: int
    total_hpwl_mm: float               # Sum of half-perimeter wirelengths
    avg_net_hpwl_mm: float
    demand_horizontal_mm: float        # Sum of horizontal spans
    demand_vertical_mm: float          # Sum of vertical spans
    demand_h_percentage: float         # % of demand that's horizontal

    # Congestion analysis
    congestion_ratio: float            # ρ = demand / capacity (key metric!)
    density_nets_per_mm2: float
    routing_complexity: str            # "SPARSE", "NORMAL", "TIGHT", "DENSE"

    # Via analysis
    allowed_via_spans: Set[Tuple[int, int]]  # Legal via layer pairs
    via_flexibility: float             # % of possible pairs that are legal


def analyze_board_characteristics(
    lattice,
    tasks: Dict,
    board_data: Optional[Dict] = None
) -> BoardCharacteristics:
    """
    Analyze board and compute routing characteristics.

    Args:
        lattice: Routing lattice with dimensions and layer info
        tasks: Dict of nets to route {net_id: (src_node, dst_node)}
        board_data: Optional dict with board metadata

    Returns:
        BoardCharacteristics with all analyzed properties
    """
    logger.info("=" * 80)
    logger.info("ANALYZING BOARD CHARACTERISTICS")
    logger.info("=" * 80)

    # Extract basic board properties
    layer_count = lattice.layers
    grid_pitch_mm = lattice.pitch

    # Calculate board dimensions from bounds or lattice steps
    if hasattr(lattice, 'bounds'):
        bounds = lattice.bounds
        board_width_mm = bounds[2] - bounds[0]
        board_height_mm = bounds[3] - bounds[1]
    else:
        # Fallback: calculate from grid steps
        board_width_mm = lattice.x_steps * grid_pitch_mm
        board_height_mm = lattice.y_steps * grid_pitch_mm

    # All layers except F.Cu (0) and B.Cu (Nz-1)
    signal_layers = list(range(1, layer_count - 1))
    plane_layers = set()  # Could be extracted from board_data if available

    logger.info(f"Board: {board_width_mm:.1f}mm × {board_height_mm:.1f}mm")
    logger.info(f"Layers: {layer_count} total, {len(signal_layers)} signal")
    logger.info(f"Grid pitch: {grid_pitch_mm}mm")

    # Compute usable area (simplified - could subtract keepouts)
    usable_area_mm2 = board_width_mm * board_height_mm

    # Assign H/V layers based on demand analysis
    h_layers, v_layers, demand_h_pct = _assign_hv_layers_by_demand(
        signal_layers, tasks, lattice
    )

    logger.info(f"H-layers: {sorted(h_layers)} ({len(h_layers)} layers)")
    logger.info(f"V-layers: {sorted(v_layers)} ({len(v_layers)} layers)")
    logger.info(f"Demand: {demand_h_pct*100:.1f}% horizontal, {(1-demand_h_pct)*100:.1f}% vertical")

    # Compute routing capacity (how much space we have)
    channels_x = int(board_width_mm / grid_pitch_mm)
    channels_y = int(board_height_mm / grid_pitch_mm)

    channels_per_h_layer = channels_x
    channels_per_v_layer = channels_y

    total_h_channels = len(h_layers) * channels_x * channels_y  # H layer capacity
    total_v_channels = len(v_layers) * channels_y * channels_x  # V layer capacity

    via_capacity_estimate = channels_x * channels_y * (layer_count - 1)  # Rough estimate

    # Compute routing demand from nets
    net_count = len(tasks)
    total_hpwl = 0.0
    demand_h = 0.0
    demand_v = 0.0

    for net_id, (src_node, dst_node) in tasks.items():
        src_x, src_y, src_z = lattice.idx_to_coord(src_node)
        dst_x, dst_y, dst_z = lattice.idx_to_coord(dst_node)

        dx = abs(dst_x - src_x) * grid_pitch_mm
        dy = abs(dst_y - src_y) * grid_pitch_mm
        hpwl = dx + dy

        total_hpwl += hpwl
        demand_h += dx
        demand_v += dy

    avg_net_hpwl = total_hpwl / max(1, net_count)

    logger.info(f"Nets: {net_count}")
    logger.info(f"Total HPWL: {total_hpwl:.0f}mm")
    logger.info(f"Avg net length: {avg_net_hpwl:.1f}mm")

    # Compute congestion ratio ρ = demand / capacity
    # This is THE key metric for parameter derivation
    # Apply detour factor (1.3×) and utilization target (0.75×)
    detour_factor = 1.3
    utilization_target = 0.75

    effective_demand = total_hpwl * detour_factor
    effective_capacity = (total_h_channels + total_v_channels) * grid_pitch_mm * utilization_target

    congestion_ratio = effective_demand / max(1.0, effective_capacity)

    # Classify routing complexity
    if congestion_ratio < 0.6:
        complexity = "SPARSE"
    elif congestion_ratio < 0.9:
        complexity = "NORMAL"
    elif congestion_ratio < 1.2:
        complexity = "TIGHT"
    else:
        complexity = "DENSE"

    logger.info(f"Congestion ratio ρ = {congestion_ratio:.3f} ({complexity})")
    logger.info(f"  Demand: {effective_demand:.0f}mm (with {detour_factor}× detour)")
    logger.info(f"  Capacity: {effective_capacity:.0f}mm (at {utilization_target*100:.0f}% target)")

    # Compute density
    density = net_count / max(1.0, usable_area_mm2)
    logger.info(f"Density: {density:.4f} nets/mm²")

    # Analyze via spans (simplified - assume all-to-all for now)
    allowed_via_spans = set()
    for i in range(layer_count):
        for j in range(i + 1, layer_count):
            allowed_via_spans.add((i, j))

    total_possible = layer_count * (layer_count - 1) // 2
    via_flexibility = len(allowed_via_spans) / max(1, total_possible)

    logger.info(f"Via flexibility: {via_flexibility*100:.0f}% of possible pairs allowed")
    logger.info("=" * 80)

    return BoardCharacteristics(
        signal_layers=signal_layers,
        h_layers=h_layers,
        v_layers=v_layers,
        layer_count=len(signal_layers),
        plane_layers=plane_layers,
        board_width_mm=board_width_mm,
        board_height_mm=board_height_mm,
        usable_area_mm2=usable_area_mm2,
        grid_pitch_mm=grid_pitch_mm,
        total_horizontal_channels=total_h_channels,
        total_vertical_channels=total_v_channels,
        channels_per_h_layer=channels_per_h_layer,
        channels_per_v_layer=channels_per_v_layer,
        via_capacity_estimate=via_capacity_estimate,
        net_count=net_count,
        total_hpwl_mm=total_hpwl,
        avg_net_hpwl_mm=avg_net_hpwl,
        demand_horizontal_mm=demand_h,
        demand_vertical_mm=demand_v,
        demand_h_percentage=demand_h / max(1.0, demand_h + demand_v),
        congestion_ratio=congestion_ratio,
        density_nets_per_mm2=density,
        routing_complexity=complexity,
        allowed_via_spans=allowed_via_spans,
        via_flexibility=via_flexibility,
    )


def _assign_hv_layers_by_demand(
    signal_layers: List[int],
    tasks: Dict,
    lattice,
    anchor_top_is_h: bool = True
) -> Tuple[Set[int], Set[int], float]:
    """
    Assign layers to horizontal or vertical routing based on demand.

    Returns:
        (h_layers, v_layers, demand_h_percentage)
    """
    if not tasks:
        # Fallback: alternating assignment
        h_layers = set(l for l in signal_layers if (l % 2 == 1) == anchor_top_is_h)
        v_layers = set(signal_layers) - h_layers
        return h_layers, v_layers, 0.5

    # Measure demand orientation
    demand_dx = 0.0
    demand_dy = 0.0

    for net_id, (src_node, dst_node) in tasks.items():
        src_x, src_y, _ = lattice.idx_to_coord(src_node)
        dst_x, dst_y, _ = lattice.idx_to_coord(dst_node)

        demand_dx += abs(dst_x - src_x)
        demand_dy += abs(dst_y - src_y)

    # Calculate horizontal demand percentage
    total_demand = demand_dx + demand_dy
    if total_demand < 1e-6:
        demand_h_pct = 0.5  # Equal split if no demand
    else:
        demand_h_pct = demand_dx / total_demand

    # Determine how many H layers to allocate
    n_signal = len(signal_layers)
    h_goal = max(1, min(n_signal - 1, int(round(n_signal * demand_h_pct))))

    # Assign layers alternating, starting from anchor
    h_layers = set()
    v_layers = set()

    want_h = anchor_top_is_h
    for layer in signal_layers:
        if want_h and len(h_layers) < h_goal:
            h_layers.add(layer)
        else:
            v_layers.add(layer)
        want_h = not want_h

    # Adjust if we didn't hit goal exactly (shouldn't happen with good rounding)
    while len(h_layers) < h_goal and v_layers:
        # Move a mid-stack V layer to H
        mid_layer = sorted(v_layers)[len(v_layers) // 2]
        v_layers.remove(mid_layer)
        h_layers.add(mid_layer)

    while len(h_layers) > h_goal and h_layers:
        # Move a mid-stack H layer to V
        mid_layer = sorted(h_layers)[len(h_layers) // 2]
        h_layers.remove(mid_layer)
        v_layers.add(mid_layer)

    return h_layers, v_layers, demand_h_pct
