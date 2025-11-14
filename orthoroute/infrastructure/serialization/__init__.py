"""
OrthoRoute Serialization Module

Provides serialization and deserialization for cloud routing workflows.

Formats:
- ORP (OrthoRoute PCB): Board representation for routing input
- ORS (OrthoRoute Solution): Routing results and metrics

All formats are JSON-based and KiCad-independent.
Coordinates are in PCB space (millimeters).
"""

from pathlib import Path
from typing import Any, Dict
import logging

from .orp_exporter import (
    export_board_to_orp,
    import_board_from_orp,
    convert_orp_to_board,
    convert_orp_to_board_data,
)

from .ors_exporter import (
    export_solution_to_ors,
    import_solution_from_ors,
    convert_ors_to_geometry_payload,
)

logger = logging.getLogger(__name__)

# Compatibility aliases for existing code
export_pcb_to_orp = export_board_to_orp
import_pcb_from_orp = import_board_from_orp


def derive_orp_filename(board_filename: str) -> str:
    """
    Derive .ORP filename from board filename.

    Example: MainController.kicad_pcb -> MainController.ORP
    """
    path = Path(board_filename)
    return str(path.with_suffix('.ORP'))


def derive_ors_filename(orp_filename: str) -> str:
    """
    Derive .ORS filename from .ORP filename.

    Example: MainController.ORP -> MainController.ORS
    """
    path = Path(orp_filename)
    return str(path.with_suffix('.ORS'))


def get_solution_summary(ors_data: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of the routing solution.

    Args:
        ors_data: Parsed ORS data (output from import_solution_from_ors)

    Returns:
        Multi-line summary string for display in GUI
    """
    try:
        summary_lines = []

        # Handle both new format (from our exporter) and old format
        if 'metadata' in ors_data and 'statistics' in ors_data:
            # New format
            metadata = ors_data['metadata']
            stats = ors_data['statistics']

            summary_lines.append(f"Solution Timestamp: {metadata.get('export_timestamp', 'Unknown')}")
            summary_lines.append(f"OrthoRoute Version: {metadata.get('orthoroute_version', 'Unknown')}")
            summary_lines.append("")

            summary_lines.append("=== Routing Quality ===")
            summary_lines.append(f"Convergence: {metadata.get('converged', False)}")
            summary_lines.append(f"Total Iterations: {metadata.get('total_iterations', 0)}")
            summary_lines.append(f"Total Runtime: {metadata.get('total_time_seconds', 0):.1f} seconds")
            summary_lines.append("")

            summary_lines.append(f"Nets Routed: {stats.get('nets_routed', 0)}")
            summary_lines.append(f"Total Wirelength: {stats.get('total_wirelength_mm', 0):.1f} mm")
            summary_lines.append(f"Total Via Count: {stats.get('total_vias', 0)}")
            summary_lines.append(f"Final Overflow: {stats.get('final_overflow_cost', 0)}")
            summary_lines.append("")

            # Geometry details
            geometry = ors_data.get('geometry', {})
            nets = geometry.get('by_net', {})
            summary_lines.append(f"=== Geometry Details ===")
            summary_lines.append(f"Total Nets: {len(nets)}")
            summary_lines.append(f"Total Tracks: {stats.get('total_tracks', 0)}")
            summary_lines.append(f"Total Vias: {stats.get('total_vias', 0)}")

        else:
            # Old format compatibility
            metadata = ors_data.get('metadata', {})
            summary_lines.append(f"Solution Timestamp: {metadata.get('timestamp', 'Unknown')}")
            summary_lines.append(f"OrthoRoute Version: {metadata.get('orthoroute_version', 'Unknown')}")
            summary_lines.append("")

            final = ors_data.get('metrics', {}).get('final', {})
            summary_lines.append("=== Routing Quality ===")
            summary_lines.append(f"Convergence: {final.get('converged', False)}")
            summary_lines.append(f"Total Iterations: {final.get('iterations', 0)}")
            summary_lines.append(f"Total Runtime: {final.get('total_time', 0):.1f} seconds")
            summary_lines.append("")

            summary_lines.append(f"Nets Routed: {final.get('nets_routed', 0)}")
            summary_lines.append(f"Total Wirelength: {final.get('wirelength', 0):.1f} mm")
            summary_lines.append(f"Total Via Count: {final.get('via_count', 0)}")
            summary_lines.append(f"Final Overflow: {final.get('overflow', 0)}")
            summary_lines.append("")

            nets = ors_data.get('nets', {})
            summary_lines.append(f"=== Net Details ===")
            summary_lines.append(f"Total Nets: {len(nets)}")

            total_traces = sum(len(net.get('traces', [])) for net in nets.values())
            total_vias = sum(len(net.get('vias', [])) for net in nets.values())
            summary_lines.append(f"Total Trace Segments: {total_traces}")
            summary_lines.append(f"Total Vias: {total_vias}")

        return "\n".join(summary_lines)

    except Exception as e:
        logger.error(f"Failed to generate solution summary: {e}")
        return f"Error generating summary: {e}"


__all__ = [
    # ORP (Board) format - new names
    "export_board_to_orp",
    "import_board_from_orp",
    "convert_orp_to_board",
    "convert_orp_to_board_data",
    # ORP (Board) format - legacy names for compatibility
    "export_pcb_to_orp",
    "import_pcb_from_orp",
    # ORS (Solution) format
    "export_solution_to_ors",
    "import_solution_from_ors",
    "convert_ors_to_geometry_payload",
    # Utility functions
    "derive_orp_filename",
    "derive_ors_filename",
    "get_solution_summary",
]
