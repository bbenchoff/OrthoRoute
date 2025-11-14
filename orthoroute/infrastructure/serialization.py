#!/usr/bin/env python3
"""
OrthoRoute Serialization Module

Handles export/import of board data and routing solutions for cloud routing workflow.

File Formats:
- .ORP (OrthoRoute PCB): Board geometry, pads, nets, DRC rules
- .ORS (OrthoRoute Solution): Routing solution with traces, vias, and metrics
"""

import json
import gzip
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Format versions for compatibility checking
ORP_FORMAT_VERSION = "1.0"
ORS_FORMAT_VERSION = "1.0"


def export_pcb_to_orp(board_data: Dict[str, Any], output_path: str, compress: bool = True) -> bool:
    """
    Export board data to .ORP file (OrthoRoute PCB format).

    Args:
        board_data: Dictionary containing complete board information
        output_path: Path to save .ORP file
        compress: If True, use gzip compression

    Returns:
        True if export succeeded, False otherwise
    """
    try:
        # Build ORP data structure according to spec
        orp_data = {
            "format": "OrthoRoute PCB",
            "version": ORP_FORMAT_VERSION,
            "timestamp": datetime.now().isoformat(),

            # Board metadata
            "board": {
                "filename": board_data.get('filename', 'unknown.kicad_pcb'),
                "bounds": {
                    "x_min": board_data.get('x_min', 0.0),
                    "y_min": board_data.get('y_min', 0.0),
                    "x_max": board_data.get('x_max', board_data.get('width', 0.0)),
                    "y_max": board_data.get('y_max', board_data.get('height', 0.0)),
                    "width": board_data.get('width', 0.0),
                    "height": board_data.get('height', 0.0),
                },
                "layer_count": board_data.get('layers', 0),
            },

            # Pads
            "pads": _serialize_pads(board_data.get('pads', [])),

            # Nets
            "nets": _serialize_nets(board_data.get('nets', {})),

            # DRC rules
            "drc": {
                "clearance": board_data.get('clearance', 0.2),
                "track_width": board_data.get('track_width', 0.2),
                "via_diameter": board_data.get('via_diameter', 0.8),
                "via_drill": board_data.get('via_drill', 0.4),
                "minimum_drill": board_data.get('minimum_drill', 0.3),
            },

            # Grid parameters
            "grid": {
                "resolution": board_data.get('grid_resolution', 0.1),
            },

            # Components (optional, for reference)
            "components": _serialize_components(board_data.get('components', [])),
        }

        # Write to file
        output_path = Path(output_path)
        json_str = json.dumps(orp_data, indent=2)

        if compress:
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                f.write(json_str)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)

        logger.info(f"Exported board to {output_path} ({len(json_str)} bytes)")
        return True

    except Exception as e:
        logger.error(f"Failed to export PCB to ORP: {e}", exc_info=True)
        return False


def import_pcb_from_orp(orp_path: str):
    """
    Import board data from .ORP file (OrthoRoute PCB format).

    Args:
        orp_path: Path to .ORP file

    Returns:
        Board domain model, or None if import failed
    """
    try:
        from ...domain.models.board import Board, Net, Pad, Component, Coordinate

        orp_path = Path(orp_path)
        logger.info(f"[IMPORT-ORP] Loading board from {orp_path}")

        # Try to read (auto-detect gzip)
        try:
            with gzip.open(orp_path, 'rt', encoding='utf-8') as f:
                orp_data = json.load(f)
        except (gzip.BadGzipFile, OSError):
            with open(orp_path, 'r', encoding='utf-8') as f:
                orp_data = json.load(f)

        # Validate format
        if orp_data.get('format') != 'OrthoRoute PCB':
            logger.error(f"Invalid ORP file format: {orp_data.get('format')}")
            return None

        # Check version compatibility
        version = orp_data.get('version', '0.0')
        if version != ORP_FORMAT_VERSION:
            logger.warning(f"ORP format version mismatch: {version} != {ORP_FORMAT_VERSION}")

        # Create board object
        board_data = orp_data['board']
        board = Board(
            id=board_data.get('filename', 'unknown'),
            name=board_data.get('filename', 'unknown')
        )
        board.layer_count = board_data.get('layer_count', 2)

        # Set bounds
        bounds = board_data.get('bounds', {})
        board.bounds = (
            bounds.get('x_min', 0.0),
            bounds.get('y_min', 0.0),
            bounds.get('x_max', 0.0),
            bounds.get('y_max', 0.0)
        )

        # Import DRC rules
        drc = orp_data.get('drc', {})
        board.clearance = drc.get('clearance', 0.2)
        board.track_width = drc.get('track_width', 0.2)
        board.via_diameter = drc.get('via_diameter', 0.8)
        board.via_drill = drc.get('via_drill', 0.4)
        board.min_drill = drc.get('minimum_drill', 0.3)

        # Import pads
        board.pads = []
        for pad_data in orp_data.get('pads', []):
            pos_data = pad_data.get('position', {})
            pos = Coordinate(
                x=pos_data.get('x', 0.0),
                y=pos_data.get('y', 0.0)
            )

            size_data = pad_data.get('size', {})
            size = (size_data.get('width', 0.0), size_data.get('height', 0.0))

            pad = Pad(
                id=str(len(board.pads)),  # Generate ID
                component_id='',
                position=pos,
                layer=pad_data.get('layer_mask', 0),
                size=size,
                net_id=pad_data.get('net')
            )

            if pad_data.get('drill_size', 0.0) > 0:
                pad.drill = pad_data['drill_size']

            board.pads.append(pad)

        # Import nets
        board.nets = []
        nets_data = orp_data.get('nets', {})
        for net_name, net_info in nets_data.items():
            net = Net(id=net_name, name=net_name)
            # Find pad IDs for this net
            net.pad_ids = set()
            for i, pad in enumerate(board.pads):
                if pad.net_id == net_name:
                    net.pad_ids.add(str(i))
            board.nets.append(net)

        logger.info(f"[IMPORT-ORP] Successfully imported {len(board.nets)} nets, {len(board.pads)} pads")
        return board

    except Exception as e:
        logger.error(f"[IMPORT-ORP] Failed to import board: {e}", exc_info=True)
        return None


def export_solution_to_ors(
    ors_path: str,
    geometry_payload,
    iteration_metrics: List[Dict[str, Any]],
    routing_metadata: Dict[str, Any],
    compress: bool = True
) -> bool:
    """
    Export routing solution to .ORS file (OrthoRoute Solution format).

    Args:
        ors_path: Path to save .ORS file
        geometry_payload: GeometryPayload with tracks and vias
        iteration_metrics: List of per-iteration metric dicts
        routing_metadata: Final routing statistics
        compress: If True, use gzip compression

    Returns:
        True if export succeeded, False otherwise
    """
    try:
        logger.info(f"[EXPORT-ORS] Exporting solution to {ors_path}")

        # Build ORS data structure according to spec
        ors_data = {
            "format": "OrthoRoute Solution",
            "version": ORS_FORMAT_VERSION,
            "timestamp": datetime.now().isoformat(),

            # Per-net geometry
            "nets": _serialize_geometry_by_net(geometry_payload),

            # Per-iteration metrics
            "metrics": {
                "iterations": iteration_metrics,
                "final": routing_metadata,
            },

            # Metadata
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "orthoroute_version": "1.0",
                "total_time": routing_metadata.get('total_time', 0.0),
                "converged": routing_metadata.get('converged', False),
            }
        }

        # Write to file
        ors_path = Path(ors_path)
        json_str = json.dumps(ors_data, indent=2)

        if compress:
            with gzip.open(ors_path, 'wt', encoding='utf-8') as f:
                f.write(json_str)
        else:
            with open(ors_path, 'w', encoding='utf-8') as f:
                f.write(json_str)

        track_count = sum(len(net.get('traces', [])) for net in ors_data['nets'].values())
        via_count = sum(len(net.get('vias', [])) for net in ors_data['nets'].values())

        logger.info(f"[EXPORT-ORS] Successfully exported solution: "
                   f"{track_count} tracks, {via_count} vias, {len(iteration_metrics)} iterations")
        return True

    except Exception as e:
        logger.error(f"[EXPORT-ORS] Failed to export solution: {e}", exc_info=True)
        return False


def import_solution_from_ors(ors_path: str) -> Optional[Dict[str, Any]]:
    """
    Import routing solution from .ORS file (OrthoRoute Solution format).

    Args:
        ors_path: Path to .ORS file

    Returns:
        Dictionary containing routing solution, or None if import failed

    Solution structure:
        {
            "nets": {
                "net_name": {
                    "traces": [(layer, x1, y1, x2, y2, width), ...],
                    "vias": [(x, y, layer_from, layer_to, diameter, drill), ...]
                }
            },
            "metrics": {
                "iterations": [...],  # Per-iteration convergence data
                "final": {...}        # Final routing quality metrics
            },
            "metadata": {...}
        }
    """
    try:
        ors_path = Path(ors_path)

        # Try to read (auto-detect gzip)
        try:
            with gzip.open(ors_path, 'rt', encoding='utf-8') as f:
                ors_data = json.load(f)
        except (gzip.BadGzipFile, OSError):
            with open(ors_path, 'r', encoding='utf-8') as f:
                ors_data = json.load(f)

        # Validate format
        if ors_data.get('format') != 'OrthoRoute Solution':
            logger.error(f"Invalid ORS file format: {ors_data.get('format')}")
            return None

        # Check version compatibility
        version = ors_data.get('version', '0.0')
        if version != ORS_FORMAT_VERSION:
            logger.warning(f"ORS format version mismatch: {version} != {ORS_FORMAT_VERSION}")

        logger.info(f"Imported solution from {ors_path}")
        return ors_data

    except Exception as e:
        logger.error(f"Failed to import solution from ORS: {e}", exc_info=True)
        return None


def _serialize_geometry_by_net(geometry_payload) -> Dict[str, Any]:
    """Group tracks and vias by net ID."""
    from collections import defaultdict

    nets = defaultdict(lambda: {"traces": [], "vias": []})

    # Group tracks by net
    for track in geometry_payload.tracks:
        net_id = track.net_id if hasattr(track, 'net_id') else 'unknown'
        trace_data = [
            track.layer if hasattr(track, 'layer') else 0,
            float(track.start.x) if hasattr(track, 'start') else 0.0,
            float(track.start.y) if hasattr(track, 'start') else 0.0,
            float(track.end.x) if hasattr(track, 'end') else 0.0,
            float(track.end.y) if hasattr(track, 'end') else 0.0,
            float(track.width) if hasattr(track, 'width') else 0.2,
        ]
        nets[net_id]["traces"].append(trace_data)

    # Group vias by net
    for via in geometry_payload.vias:
        net_id = via.net_id if hasattr(via, 'net_id') else 'unknown'
        via_data = [
            float(via.position.x) if hasattr(via, 'position') else 0.0,
            float(via.position.y) if hasattr(via, 'position') else 0.0,
            via.layers[0] if hasattr(via, 'layers') and len(via.layers) > 0 else 0,
            via.layers[1] if hasattr(via, 'layers') and len(via.layers) > 1 else 0,
            float(via.size) if hasattr(via, 'size') else 0.4,
            float(via.drill) if hasattr(via, 'drill') else 0.2,
        ]
        nets[net_id]["vias"].append(via_data)

    return dict(nets)


def _serialize_pads(pads: List[Any]) -> List[Dict[str, Any]]:
    """Convert pad objects to serializable format."""
    serialized = []
    for pad in pads:
        try:
            # Handle different pad representations
            if hasattr(pad, '__dict__'):
                pad_dict = {
                    "position": {
                        "x": getattr(pad, 'x', 0.0),
                        "y": getattr(pad, 'y', 0.0),
                    },
                    "net": getattr(pad, 'net', None),
                    "drill_size": getattr(pad, 'drill_size', 0.0),
                    "layer_mask": getattr(pad, 'layer_mask', 0),
                    "shape": getattr(pad, 'shape', 'circle'),
                    "size": {
                        "width": getattr(pad, 'width', 0.0),
                        "height": getattr(pad, 'height', 0.0),
                    }
                }
            elif isinstance(pad, dict):
                pad_dict = pad
            else:
                continue

            serialized.append(pad_dict)
        except Exception as e:
            logger.warning(f"Failed to serialize pad: {e}")
            continue

    return serialized


def _serialize_nets(nets: Dict[str, Any]) -> Dict[str, Any]:
    """Convert nets dictionary to serializable format."""
    serialized = {}
    for net_name, net_data in nets.items():
        try:
            if isinstance(net_data, dict):
                serialized[net_name] = net_data
            elif hasattr(net_data, '__dict__'):
                serialized[net_name] = {
                    "terminals": getattr(net_data, 'terminals', []),
                    "priority": getattr(net_data, 'priority', 1),
                }
            else:
                serialized[net_name] = {"terminals": []}
        except Exception as e:
            logger.warning(f"Failed to serialize net {net_name}: {e}")
            continue

    return serialized


def _serialize_components(components: List[Any]) -> List[Dict[str, Any]]:
    """Convert component objects to serializable format."""
    serialized = []
    for comp in components:
        try:
            if hasattr(comp, '__dict__'):
                comp_dict = {
                    "reference": getattr(comp, 'reference', 'Unknown'),
                    "position": {
                        "x": getattr(comp, 'x', 0.0),
                        "y": getattr(comp, 'y', 0.0),
                    },
                    "rotation": getattr(comp, 'rotation', 0.0),
                    "layer": getattr(comp, 'layer', 'F.Cu'),
                }
            elif isinstance(comp, dict):
                comp_dict = comp
            else:
                continue

            serialized.append(comp_dict)
        except Exception as e:
            logger.warning(f"Failed to serialize component: {e}")
            continue

    return serialized


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
        ors_data: Parsed ORS data

    Returns:
        Multi-line summary string for display in GUI
    """
    try:
        summary_lines = []

        # Metadata
        metadata = ors_data.get('metadata', {})
        summary_lines.append(f"Solution Timestamp: {metadata.get('timestamp', 'Unknown')}")
        summary_lines.append(f"OrthoRoute Version: {metadata.get('orthoroute_version', 'Unknown')}")
        summary_lines.append("")

        # Final metrics
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

        # Per-net summary
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
