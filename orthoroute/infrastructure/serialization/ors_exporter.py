"""
ORS (OrthoRoute Solution) Format Exporter/Importer

Serializes routing results to .ORS format for cloud routing workflows.
The .ORS format is a JSON-based format for routing geometry and metrics.

Format version: 1.0
Coordinates: PCB coordinates in millimeters (mm)
"""

import json
import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


def export_solution_to_ors(
    geometry,
    iteration_metrics: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    filepath: str,
    compress: bool = True
) -> None:
    """
    Export routing solution to .ORS (OrthoRoute Solution) format.

    Args:
        geometry: Geometry payload from UnifiedPathFinder.get_geometry_payload()
                  Should have .tracks and .vias attributes containing routing geometry
        iteration_metrics: List of per-iteration metrics dictionaries containing:
            - iteration: Iteration number
            - overuse_count: Number of overused resources
            - nets_routed: Number of successfully routed nets
            - overflow_cost: Total overflow cost
            - wirelength: Total wirelength in mm
            - via_count: Total number of vias
            - iteration_time: Time taken for iteration in seconds
        metadata: Dictionary containing routing metadata:
            - total_iterations: Total number of iterations completed
            - converged: Whether routing converged (boolean)
            - total_time: Total routing time in seconds
            - final_wirelength: Final total wirelength in mm
            - final_via_count: Final total via count
            - final_overflow: Final overflow count
            - orthoroute_version: Version string (optional)
            - board_name: Board name (optional)
            - notes: Additional notes (optional)
        filepath: Path where the .ORS file will be saved
        compress: If True, use gzip compression (default: True)

    Raises:
        ValueError: If geometry is invalid or missing required fields
        IOError: If file cannot be written

    Example:
        >>> router = UnifiedPathFinder(...)
        >>> router.route_all_nets()
        >>> geometry = router.get_geometry_payload()
        >>> metrics = router.get_iteration_metrics()
        >>> metadata = {
        ...     "total_iterations": 50,
        ...     "converged": True,
        ...     "total_time": 12.5,
        ...     "final_wirelength": 1250.5,
        ...     "final_via_count": 45,
        ...     "final_overflow": 0
        ... }
        >>> export_solution_to_ors(geometry, metrics, metadata, "solution.ors")
    """
    # Validate geometry
    if geometry is None:
        raise ValueError("Geometry cannot be None")

    if not hasattr(geometry, 'tracks') or not hasattr(geometry, 'vias'):
        raise ValueError("Geometry must have 'tracks' and 'vias' attributes")

    # Build ORS structure
    ors_data = {
        "format_version": "1.0",
        "metadata": _build_solution_metadata(metadata),
        "geometry": _build_geometry_data(geometry),
        "iteration_metrics": _build_iteration_metrics(iteration_metrics),
        "statistics": _build_statistics(geometry, iteration_metrics, metadata),
    }

    # Write to file with pretty formatting
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if compress:
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            json.dump(ors_data, f, indent=2, ensure_ascii=False)
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ors_data, f, indent=2, ensure_ascii=False)


def _build_solution_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Build solution metadata section."""
    return {
        "export_timestamp": datetime.utcnow().isoformat() + "Z",
        "orthoroute_version": metadata.get("orthoroute_version", "0.1.0"),
        "board_name": metadata.get("board_name", "unknown"),
        "total_iterations": metadata.get("total_iterations", 0),
        "converged": metadata.get("converged", False),
        "total_time_seconds": metadata.get("total_time", 0.0),
        "notes": metadata.get("notes", ""),
    }


def _build_geometry_data(geometry) -> Dict[str, Any]:
    """
    Build geometry section from routing results.

    Extracts tracks and vias from the geometry payload and organizes them
    by net for efficient querying and visualization.

    Returns:
        Dictionary containing:
        - by_net: Geometry organized by net_id
        - all_tracks: Flat list of all tracks
        - all_vias: Flat list of all vias
        - layer_usage: Statistics about layer usage
    """
    # Organize geometry by net
    geometry_by_net = {}
    layer_usage = {}

    # Process tracks
    import logging
    logger = logging.getLogger(__name__)

    for idx, track in enumerate(geometry.tracks):
        # DEBUG: Log first track to see actual format
        if idx == 0:
            logger.info(f"[ORS-EXPORT] First track from geometry: {track}")
            logger.info(f"[ORS-EXPORT] Track keys: {track.keys() if isinstance(track, dict) else 'not a dict'}")

        # Extract net_id - handle both 'net' and 'net_id' keys
        net_id = str(track.get('net_id') or track.get('net', 'unknown'))

        if net_id not in geometry_by_net:
            geometry_by_net[net_id] = {
                "net_id": net_id,
                "tracks": [],
                "vias": [],
            }

        # Extract track data - handle multiple coordinate formats
        if 'start' in track and 'end' in track:
            # Format 1: tuple coordinates
            start_x, start_y = track['start']
            end_x, end_y = track['end']
            if idx == 0:
                logger.info(f"[ORS-EXPORT] Format: tuples - start=({start_x}, {start_y}), end=({end_x}, {end_y})")
        elif 'x1' in track and 'y1' in track:
            # Format 2: x1, y1, x2, y2 keys (actual format from pathfinder!)
            start_x = track.get('x1', 0.0)
            start_y = track.get('y1', 0.0)
            end_x = track.get('x2', 0.0)
            end_y = track.get('y2', 0.0)
            if idx == 0:
                logger.info(f"[ORS-EXPORT] Format: x1/y1/x2/y2 - start=({start_x}, {start_y}), end=({end_x}, {end_y})")
        else:
            # Format 3: start_x, start_y, end_x, end_y keys
            start_x = track.get('start_x', 0.0)
            start_y = track.get('start_y', 0.0)
            end_x = track.get('end_x', 0.0)
            end_y = track.get('end_y', 0.0)
            if idx == 0:
                logger.info(f"[ORS-EXPORT] Format: start_x/start_y - start=({start_x}, {start_y}), end=({end_x}, {end_y})")

        layer = track.get('layer', 0)
        width = track.get('width', 0.15)

        track_dict = {
            "layer": layer,
            "start": {
                "x": float(start_x),
                "y": float(start_y),
            },
            "end": {
                "x": float(end_x),
                "y": float(end_y),
            },
            "width": float(width),
        }

        geometry_by_net[net_id]["tracks"].append(track_dict)

        # Track layer usage
        layer_usage[layer] = layer_usage.get(layer, 0) + 1

    # Process vias
    for via in geometry.vias:
        # Extract net_id - handle both 'net' and 'net_id' keys
        net_id = str(via.get('net_id') or via.get('net', 'unknown'))

        if net_id not in geometry_by_net:
            geometry_by_net[net_id] = {
                "net_id": net_id,
                "tracks": [],
                "vias": [],
            }

        # Extract via data - handle multiple position formats
        if 'position' in via:
            position = via['position']
            if isinstance(position, (list, tuple)):
                pos_x, pos_y = position
            else:
                pos_x = position.get('x', 0.0)
                pos_y = position.get('y', 0.0)
        elif 'x' in via and 'y' in via:
            # Direct x, y keys on via
            pos_x = via.get('x', 0.0)
            pos_y = via.get('y', 0.0)
        else:
            pos_x, pos_y = 0.0, 0.0

        via_dict = {
            "position": {
                "x": float(pos_x),
                "y": float(pos_y),
            },
            "from_layer": via.get('from_layer', 0),
            "to_layer": via.get('to_layer', 1),
            "diameter": float(via.get('size', 0.4)),
            "drill": float(via.get('drill', 0.2)),
        }

        geometry_by_net[net_id]["vias"].append(via_dict)

    # Build flat lists for backward compatibility
    all_tracks = []
    all_vias = []

    for net_data in geometry_by_net.values():
        for track in net_data["tracks"]:
            track_with_net = track.copy()
            track_with_net["net_id"] = net_data["net_id"]
            all_tracks.append(track_with_net)

        for via in net_data["vias"]:
            via_with_net = via.copy()
            via_with_net["net_id"] = net_data["net_id"]
            all_vias.append(via_with_net)

    return {
        "by_net": geometry_by_net,
        "all_tracks": all_tracks,
        "all_vias": all_vias,
        "layer_usage": layer_usage,
    }


def _build_iteration_metrics(iteration_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build iteration metrics array.

    Normalizes per-iteration metrics for consistent storage.
    """
    normalized_metrics = []

    for metric in iteration_metrics:
        normalized_metric = {
            "iteration": metric.get("iteration", 0),
            "overuse_count": metric.get("overuse_count", metric.get("overuse", 0)),
            "nets_routed": metric.get("nets_routed", metric.get("routed", 0)),
            "overflow_cost": metric.get("overflow_cost", metric.get("overflow", 0.0)),
            "wirelength": metric.get("wirelength", 0.0),
            "via_count": metric.get("via_count", metric.get("vias", 0)),
            "iteration_time_seconds": metric.get("iteration_time", metric.get("time", 0.0)),
        }
        normalized_metrics.append(normalized_metric)

    return normalized_metrics


def _build_statistics(
    geometry,
    iteration_metrics: List[Dict[str, Any]],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build statistics summary.

    Aggregates final routing statistics for quick reference.
    """
    total_tracks = len(geometry.tracks) if hasattr(geometry, 'tracks') else 0
    total_vias = len(geometry.vias) if hasattr(geometry, 'vias') else 0

    # Calculate total wirelength from tracks
    total_wirelength = 0.0
    if hasattr(geometry, 'tracks'):
        for track in geometry.tracks:
            if 'start' in track and 'end' in track:
                start_x, start_y = track['start']
                end_x, end_y = track['end']
            else:
                start_x = track.get('start_x', 0.0)
                start_y = track.get('start_y', 0.0)
                end_x = track.get('end_x', 0.0)
                end_y = track.get('end_y', 0.0)

            dx = end_x - start_x
            dy = end_y - start_y
            length = (dx * dx + dy * dy) ** 0.5
            total_wirelength += length

    # Get nets routed (from metadata or count unique net_ids in geometry)
    nets_routed = metadata.get("nets_routed", 0)
    if nets_routed == 0 and hasattr(geometry, 'tracks'):
        unique_nets = set()
        for track in geometry.tracks:
            net_id = track.get('net_id')
            if net_id:
                unique_nets.add(str(net_id))
        nets_routed = len(unique_nets)

    # Get final metrics from last iteration
    final_overuse = 0
    final_overflow = 0.0
    if iteration_metrics:
        last_metric = iteration_metrics[-1]
        final_overuse = last_metric.get("overuse_count", last_metric.get("overuse", 0))
        final_overflow = last_metric.get("overflow_cost", last_metric.get("overflow", 0.0))

    return {
        "total_tracks": total_tracks,
        "total_vias": total_vias,
        "total_wirelength_mm": total_wirelength,
        "nets_routed": nets_routed,
        "final_overuse_count": final_overuse,
        "final_overflow_cost": final_overflow,
        "converged": metadata.get("converged", final_overuse == 0),
        "iterations_completed": len(iteration_metrics),
    }


def import_solution_from_ors(filepath: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Import routing solution from .ORS file.

    Args:
        filepath: Path to .ORS file

    Returns:
        Tuple of (geometry_data, metadata):
        - geometry_data: Dictionary containing geometry organized by net
        - metadata: Dictionary containing routing metadata and statistics

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid or unsupported version

    Example:
        >>> geometry, metadata = import_solution_from_ors("solution.ors")
        >>> print(f"Solution has {metadata['statistics']['total_tracks']} tracks")
        >>> print(f"Converged: {metadata['metadata']['converged']}")
        >>> for net_id, net_geom in geometry['by_net'].items():
        ...     print(f"Net {net_id}: {len(net_geom['tracks'])} tracks, {len(net_geom['vias'])} vias")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"ORS file not found: {filepath}")

    # Auto-detect gzip compression
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            ors_data = json.load(f)
    except (gzip.BadGzipFile, OSError):
        # Not compressed, read as plain JSON
        with open(filepath, 'r', encoding='utf-8') as f:
            ors_data = json.load(f)

    # Validate format version
    format_version = ors_data.get("format_version")
    if not format_version:
        raise ValueError("Missing format_version in ORS file")

    if format_version != "1.0":
        raise ValueError(f"Unsupported ORS format version: {format_version}")

    # Validate required sections
    required_sections = ["metadata", "geometry", "iteration_metrics", "statistics"]
    missing_sections = [s for s in required_sections if s not in ors_data]
    if missing_sections:
        raise ValueError(f"Missing required sections: {', '.join(missing_sections)}")

    # Return geometry and combined metadata
    geometry_data = ors_data["geometry"]
    metadata = {
        "metadata": ors_data["metadata"],
        "statistics": ors_data["statistics"],
        "iteration_metrics": ors_data["iteration_metrics"],
    }

    return geometry_data, metadata


def convert_ors_to_geometry_payload(geometry_data: Dict[str, Any]):
    """
    Convert ORS geometry data to a format compatible with geometry payload.

    Args:
        geometry_data: Geometry data from import_solution_from_ors()

    Returns:
        Object with .tracks and .vias attributes containing routing geometry

    Example:
        >>> geometry_data, metadata = import_solution_from_ors("solution.ors")
        >>> geometry_payload = convert_ors_to_geometry_payload(geometry_data)
        >>> print(f"Loaded {len(geometry_payload.tracks)} tracks")
    """
    class GeometryPayload:
        """Simple container for geometry data."""
        def __init__(self, tracks: List[Dict], vias: List[Dict]):
            self.tracks = tracks
            self.vias = vias

    # Use the flat lists from geometry data
    tracks = geometry_data.get("all_tracks", [])
    vias = geometry_data.get("all_vias", [])

    # Convert back to internal format (with tuples for positions)
    converted_tracks = []
    for track in tracks:
        converted_track = {
            "net_id": track.get("net_id"),
            "layer": track.get("layer"),
            "start": (track["start"]["x"], track["start"]["y"]),
            "end": (track["end"]["x"], track["end"]["y"]),
            "width": track.get("width"),
        }
        converted_tracks.append(converted_track)

    converted_vias = []
    for via in vias:
        converted_via = {
            "net_id": via.get("net_id"),
            "position": (via["position"]["x"], via["position"]["y"]),
            "from_layer": via.get("from_layer"),
            "to_layer": via.get("to_layer"),
            "size": via.get("diameter"),  # Note: ORS uses 'diameter', internal uses 'size'
            "drill": via.get("drill"),
        }
        converted_vias.append(converted_via)

    return GeometryPayload(converted_tracks, converted_vias)
