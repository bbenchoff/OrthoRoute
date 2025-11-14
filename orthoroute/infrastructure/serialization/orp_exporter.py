"""
ORP (OrthoRoute PCB) Format Exporter

Serializes Board objects to .ORP format for cloud routing workflows.
The .ORP format is a JSON-based, KiCad-independent representation of PCB data.

Format version: 1.0
Coordinates: PCB coordinates in millimeters (mm)
"""

import json
import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import uuid4


def _make_pad_id(component_ref: str, pad_name: str, fallback_idx: int) -> str:
    """
    Create consistent pad ID for both export and import.

    Args:
        component_ref: Component reference (e.g., "J1")
        pad_name: Pad name/number (e.g., "1", "A1")
        fallback_idx: Index to use if component_ref is empty

    Returns:
        Consistent pad ID string
    """
    if component_ref and pad_name:
        return f"{component_ref}@{pad_name}"
    elif component_ref:
        return f"{component_ref}@{fallback_idx}"
    else:
        return f"PAD{fallback_idx:05d}"


def export_board_to_orp(board, filepath: str, compress: bool = True) -> None:
    """
    Export a Board object or board_data dictionary to .ORP (OrthoRoute PCB) format.

    Args:
        board: Board object (orthoroute.domain.models.board.Board) OR board_data dictionary
        filepath: Path where the .ORP file will be saved
        compress: If True, use gzip compression (default: True)

    Raises:
        ValueError: If board validation fails
        IOError: If file cannot be written

    Example:
        >>> from orthoroute.domain.models.board import Board
        >>> board = load_board_from_kicad("design.kicad_pcb")
        >>> export_board_to_orp(board, "design.orp")
    """
    # Check if board is a dictionary (from GUI) or Board object
    is_dict = isinstance(board, dict)

    if not is_dict:
        # Validate board integrity for Board objects
        issues = board.validate_integrity()
        if issues:
            raise ValueError(f"Board validation failed: {', '.join(issues)}")

    # Build ORP structure
    orp_data = {
        "format_version": "1.0",
        "metadata": _build_metadata(board, filepath, is_dict),
        "board": _build_board_data(board, is_dict),
        "pads": _build_pads_data(board, is_dict),
        "nets": _build_nets_data(board, is_dict),
        "layers": _build_layers_data(board, is_dict),
        "drc_rules": _build_drc_rules(board, is_dict),
        "grid_parameters": _build_grid_parameters(board, is_dict),
    }

    # Clean up temporary data from board dict
    if is_dict and '_pad_id_map' in board:
        del board['_pad_id_map']

    # Write to file with pretty formatting
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if compress:
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            json.dump(orp_data, f, indent=2, ensure_ascii=False)
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(orp_data, f, indent=2, ensure_ascii=False)


def _build_metadata(board, filepath: str, is_dict: bool = False) -> Dict[str, Any]:
    """Build metadata section."""
    if is_dict:
        return {
            "filename": Path(filepath).name,
            "board_name": board.get('filename', 'unknown.kicad_pcb'),
            "board_id": board.get('filename', 'unknown'),
            "export_timestamp": datetime.utcnow().isoformat() + "Z",
            "orthoroute_version": "0.1.0",
        }
    else:
        return {
            "filename": Path(filepath).name,
            "board_name": board.name,
            "board_id": board.id,
            "export_timestamp": datetime.utcnow().isoformat() + "Z",
            "orthoroute_version": "0.1.0",
        }


def _build_board_data(board, is_dict: bool = False) -> Dict[str, Any]:
    """Build board geometry and properties."""
    if is_dict:
        return {
            "bounds": {
                "x_min": board.get('x_min', 0.0),
                "y_min": board.get('y_min', 0.0),
                "x_max": board.get('x_max', board.get('width', 0.0)),
                "y_max": board.get('y_max', board.get('height', 0.0)),
                "width": board.get('width', 0.0),
                "height": board.get('height', 0.0),
            },
            "layer_count": board.get('layers', 0),
            "thickness": board.get('thickness', 1.6),
        }
    else:
        bounds = board.get_bounds()
        return {
            "bounds": {
                "x_min": bounds.min_x,
                "y_min": bounds.min_y,
                "x_max": bounds.max_x,
                "y_max": bounds.max_y,
                "width": bounds.width,
                "height": bounds.height,
            },
            "layer_count": board.layer_count,
            "thickness": board.thickness,
        }


def _build_pads_data(board, is_dict: bool = False) -> List[Dict[str, Any]]:
    """
    Build pads list with all required routing information.

    Returns list of pad dictionaries containing:
    - id: Unique pad identifier
    - component_id: Parent component ID
    - component_ref: Component reference designator (e.g., "U1", "R2")
    - net_id: Net ID (null for unconnected pads)
    - net_name: Net name for human readability
    - position: {x, y} in mm
    - size: {width, height} in mm
    - drill_size: Drill diameter in mm (null for SMD pads)
    - layer: Layer name or "THRU" for through-hole pads
    - shape: Pad shape (circle, rect, oval, etc.)
    - angle: Rotation angle in degrees
    """
    pads_data = []

    if is_dict:
        # Handle board_data dictionary from GUI
        # GUI pads have structure: {'x', 'y', 'component', 'name', 'net_name', 'net_code', 'width', 'height', 'drill', 'layers', 'type'}
        pads_list = board.get('pads', [])
        nets_dict = board.get('nets', {})

        # Build pad ID mapping for consistent reference (store in board temporarily)
        pad_id_map = {}  # Maps original pad dict index to generated pad_id

        for idx, pad in enumerate(pads_list):
            # Generate unique pad ID from component and pad name
            component_ref = pad.get('component', '')
            pad_name = pad.get('name', str(idx))
            pad_id = _make_pad_id(component_ref, pad_name, idx)

            # Store mapping
            pad_id_map[idx] = pad_id

            # Get net info
            net_name = pad.get('net_name', '')
            net_code = pad.get('net_code', 0)

            # Determine layer (for multi-layer pads, use first layer)
            layers = pad.get('layers', ['F.Cu'])
            layer = layers[0] if layers else 'F.Cu'

            pad_dict = {
                "id": pad_id,
                "component_id": component_ref,  # Use component as component_id
                "component_ref": component_ref,
                "net_id": net_name,  # Use net_name as net_id
                "net_name": net_name,
                "position": {
                    "x": pad.get('x', 0.0),
                    "y": pad.get('y', 0.0),
                },
                "size": {
                    "width": pad.get('width', 0.0),
                    "height": pad.get('height', 0.0),
                },
                "drill_size": pad.get('drill'),
                "layer": layer,
                "shape": "circle",  # GUI doesn't provide shape, default to circle
                "angle": 0.0,  # GUI doesn't provide angle
            }

            pads_data.append(pad_dict)

        # Store pad_id_map for use in nets building
        board['_pad_id_map'] = pad_id_map
    else:
        # Handle Board object
        for component in board.components:
            for pad in component.pads:
                pad_dict = {
                    "id": pad.id,
                    "component_id": pad.component_id,
                    "component_ref": component.reference,
                    "net_id": pad.net_id,
                    "net_name": None,  # Will be populated below
                    "position": {
                        "x": pad.position.x,
                        "y": pad.position.y,
                    },
                    "size": {
                        "width": pad.size[0],
                        "height": pad.size[1],
                    },
                    "drill_size": pad.drill_size,
                    "layer": pad.layer,
                    "shape": pad.shape,
                    "angle": pad.angle,
                }

                # Add net name if pad is connected
                if pad.net_id:
                    net = board.get_net(pad.net_id)
                    if net:
                        pad_dict["net_name"] = net.name

                pads_data.append(pad_dict)

    return pads_data


def _build_nets_data(board, is_dict: bool = False) -> List[Dict[str, Any]]:
    """
    Build nets list with terminal positions.

    Returns list of net dictionaries containing:
    - id: Unique net identifier
    - name: Net name (e.g., "GND", "/data_bus/D0")
    - netclass: Netclass name for DRC rules
    - pad_count: Number of pads in net
    - is_routable: Whether net has 2+ pads and needs routing
    - terminals: List of terminal positions for routing
    - bounds: Bounding box of all terminals
    """
    nets_data = []

    if is_dict:
        # Handle board_data dictionary from GUI
        # GUI nets have structure: {net_name: {'name': net_name, 'code': net_code, 'pads': [pad_dicts]}}
        nets_dict = board.get('nets', {})
        pads_list = board.get('pads', [])
        pad_id_map = board.get('_pad_id_map', {})  # Get the mapping created in _build_pads_data

        for net_name, net in nets_dict.items():
            # Find all pads belonging to this net (GUI pads use 'net_name' field)
            # Store indices along with pads
            net_pad_indices = [(idx, pad) for idx, pad in enumerate(pads_list) if pad.get('net_name') == net_name]

            terminals = []
            x_coords = []
            y_coords = []

            for pad_idx, pad in net_pad_indices:
                # GUI pads have 'x', 'y' directly, not 'position.x'
                x = pad.get('x', 0.0)
                y = pad.get('y', 0.0)

                # Use the pad_id from the mapping created in _build_pads_data
                pad_id = pad_id_map.get(pad_idx, f"pad_{pad_idx}")

                # Determine layer
                layers = pad.get('layers', ['F.Cu'])
                layer = layers[0] if layers else 'F.Cu'

                terminals.append({
                    "pad_id": pad_id,
                    "position": {
                        "x": x,
                        "y": y,
                    },
                    "layer": layer,
                })

                x_coords.append(x)
                y_coords.append(y)

            # Calculate bounds
            if x_coords and y_coords:
                bounds = {
                    "x_min": min(x_coords),
                    "y_min": min(y_coords),
                    "x_max": max(x_coords),
                    "y_max": max(y_coords),
                }
            else:
                bounds = {
                    "x_min": 0.0,
                    "y_min": 0.0,
                    "x_max": 0.0,
                    "y_max": 0.0,
                }

            net_dict = {
                "id": net_name,  # Use net_name as ID
                "name": net.get('name', net_name),
                "netclass": "Default",  # GUI doesn't provide netclass
                "pad_count": len(net_pad_indices),
                "is_routable": len(net_pad_indices) >= 2,
                "terminals": terminals,
                "bounds": bounds,
            }

            nets_data.append(net_dict)
    else:
        # Handle Board object
        for net in board.nets:
            terminals = []
            for pad in net.pads:
                terminals.append({
                    "pad_id": pad.id,
                    "position": {
                        "x": pad.position.x,
                        "y": pad.position.y,
                    },
                    "layer": pad.layer,
                })

            bounds = net.get_bounds()

            net_dict = {
                "id": net.id,
                "name": net.name,
                "netclass": net.netclass,
                "pad_count": len(net.pads),
                "is_routable": net.is_routable,
                "terminals": terminals,
                "bounds": {
                    "x_min": bounds.min_x,
                    "y_min": bounds.min_y,
                    "x_max": bounds.max_x,
                    "y_max": bounds.max_y,
                },
            }

            nets_data.append(net_dict)

    return nets_data


def _build_layers_data(board, is_dict: bool = False) -> List[Dict[str, Any]]:
    """
    Build layers list with stackup information.

    Returns list of layer dictionaries containing:
    - name: Layer name (e.g., "F.Cu", "In1.Cu", "B.Cu")
    - type: Layer type (signal, power, ground, copper)
    - stackup_position: Position in stackup (0 = top)
    - thickness: Copper thickness in mm
    - material: Material type
    - is_routing_layer: Whether layer can be used for routing
    """
    layers_data = []

    if is_dict:
        # Handle board_data dictionary from GUI
        layer_count = board.get('layers', 2)
        layer_names = board.get('layer_names', [])

        # If no layer names provided, generate default names
        if not layer_names:
            if layer_count == 2:
                layer_names = ["F.Cu", "B.Cu"]
            else:
                layer_names = ["F.Cu"] + [f"In{i}.Cu" for i in range(1, layer_count - 1)] + ["B.Cu"]

        for i, layer_name in enumerate(layer_names[:layer_count]):
            layer_dict = {
                "name": layer_name,
                "type": "signal",
                "stackup_position": i,
                "thickness": 0.035,  # mm - standard 1oz copper
                "material": "copper",
                "is_routing_layer": True,
            }
            layers_data.append(layer_dict)
    else:
        # Handle Board object
        for layer in board.layers:
            layer_dict = {
                "name": layer.name,
                "type": layer.type,
                "stackup_position": layer.stackup_position,
                "thickness": layer.thickness,
                "material": layer.material,
                "is_routing_layer": layer.is_routing_layer,
            }
            layers_data.append(layer_dict)

    return layers_data


def _build_drc_rules(board, is_dict: bool = False) -> Dict[str, Any]:
    """
    Build DRC rules from board netclasses.

    Extracts design rules from the Default netclass.
    For boards with custom netclasses, stores per-netclass rules.

    Returns:
        Dictionary containing:
        - default: Default netclass rules
        - netclasses: Dict of netclass-specific rules
        - min_values: Absolute minimum values for manufacturing
    """
    drc_rules = {
        "default": {},
        "netclasses": {},
        "min_values": {
            "track_width": 0.1,    # mm - typical PCB fab minimum
            "clearance": 0.1,       # mm - typical PCB fab minimum
            "via_diameter": 0.2,    # mm - typical PCB fab minimum
            "via_drill": 0.1,       # mm - typical PCB fab minimum
        }
    }

    if is_dict:
        # Handle board_data dictionary from GUI
        # Extract DRC rules directly from board_data keys
        drc_rules["default"] = {
            "track_width": board.get('track_width', 0.25),
            "clearance": board.get('clearance', 0.2),
            "via_diameter": board.get('via_diameter', 0.6),
            "via_drill": board.get('via_drill', 0.3),
            "netclass": "Default",
        }

        # Update min_values if available
        drc_rules["min_values"] = {
            "track_width": board.get('min_track_width', 0.1),
            "clearance": board.get('min_clearance', 0.1),
            "via_diameter": board.get('min_via_diameter', 0.2),
            "via_drill": board.get('min_via_drill', 0.1),
        }
    else:
        # Handle Board object
        # Try to get DRC constraints from board if available
        # The Board model may have constraints attached, or we extract from netclasses
        if hasattr(board, 'constraints'):
            constraints = board.constraints
            drc_rules["min_values"] = {
                "track_width": constraints.min_track_width,
                "clearance": constraints.min_track_spacing,
                "via_diameter": constraints.min_via_diameter,
                "via_drill": constraints.min_via_drill,
            }

        # Extract default netclass rules
        # Note: Board model doesn't have direct netclass access yet,
        # so we use reasonable defaults that match typical KiCad boards
        drc_rules["default"] = {
            "track_width": 0.25,      # mm - common default
            "clearance": 0.2,          # mm - common default
            "via_diameter": 0.6,       # mm - common default
            "via_drill": 0.3,          # mm - common default
            "netclass": "Default",
        }

    # TODO: Extract per-netclass rules when Board model is enhanced with constraints
    # For now, we only export default rules

    return drc_rules


def _build_grid_parameters(board, is_dict: bool = False) -> Dict[str, Any]:
    """
    Build grid/discretization parameters for routing.

    These parameters define how the continuous PCB space is discretized
    for the routing grid. They should match the parameters used during
    routing to ensure coordinate consistency.

    Returns:
        Dictionary containing:
        - grid_pitch: Grid spacing in mm (typically 0.4-0.5mm)
        - expansion_margin: Margin around board bounds in mm
        - coordinate_system: Description of coordinate reference
    """
    if is_dict:
        # Handle board_data dictionary from GUI
        grid_pitch = board.get('grid_resolution', 0.4)

        # Get bounds from board_data
        x_min = board.get('x_min', 0.0)
        y_min = board.get('y_min', 0.0)
        x_max = board.get('x_max', board.get('width', 0.0))
        y_max = board.get('y_max', board.get('height', 0.0))

        return {
            "grid_pitch": grid_pitch,
            "expansion_margin": board.get('expansion_margin', 3.0),
            "coordinate_system": "PCB coordinates in millimeters, origin at board bounds minimum",
            "board_bounds_used": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            },
        }
    else:
        # Handle Board object
        bounds = board.get_bounds()

        return {
            "grid_pitch": 0.4,  # mm - default from PathFinderConfig.GRID_PITCH
            "expansion_margin": 3.0,  # mm - default expansion around board
            "coordinate_system": "PCB coordinates in millimeters, origin at board bounds minimum",
            "board_bounds_used": {
                "x_min": bounds.min_x,
                "y_min": bounds.min_y,
                "x_max": bounds.max_x,
                "y_max": bounds.max_y,
            },
        }


def import_board_from_orp(filepath: str) -> Dict[str, Any]:
    """
    Import board data from .ORP file.

    Args:
        filepath: Path to .ORP file

    Returns:
        Dictionary containing parsed ORP data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid or unsupported version

    Example:
        >>> orp_data = import_board_from_orp("design.orp")
        >>> print(f"Board has {len(orp_data['nets'])} nets")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"ORP file not found: {filepath}")

    # Auto-detect gzip compression
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            orp_data = json.load(f)
    except (gzip.BadGzipFile, OSError):
        # Not compressed, read as plain JSON
        with open(filepath, 'r', encoding='utf-8') as f:
            orp_data = json.load(f)

    # Validate format version
    format_version = orp_data.get("format_version")
    if not format_version:
        raise ValueError("Missing format_version in ORP file")

    if format_version != "1.0":
        raise ValueError(f"Unsupported ORP format version: {format_version}")

    # Validate required sections
    required_sections = ["metadata", "board", "pads", "nets", "layers", "drc_rules", "grid_parameters"]
    missing_sections = [s for s in required_sections if s not in orp_data]
    if missing_sections:
        raise ValueError(f"Missing required sections: {', '.join(missing_sections)}")

    return orp_data


def convert_orp_to_board_data(orp_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert ORP dictionary to board_data format (same as GUI uses).

    This is the preferred conversion for headless routing since UnifiedPathFinder
    works better with the board_data dictionary format.

    Args:
        orp_data: Dictionary containing ORP data (from import_board_from_orp())

    Returns:
        board_data dict with keys like:
        - 'filename': str
        - 'x_min', 'y_min', 'x_max', 'y_max': floats (bounds)
        - 'width', 'height': floats
        - 'layers': int (layer count)
        - 'layer_names': list of str (['F.Cu', 'In1.Cu', ..., 'B.Cu'])
        - 'pads': list of pad dicts with keys: 'id', 'component', 'name', 'net_name',
                  'x', 'y', 'width', 'height', 'drill', 'layers', 'type'
        - 'nets': dict {net_name: {'name': str, 'code': int, 'pads': list}}
        - 'components': list of component dicts
        - 'clearance', 'track_width', 'via_diameter', 'via_drill': floats (DRC rules)
        - 'grid_resolution': float
        - 'airwires': list (can be empty)
        - 'bounds': tuple (x_min, y_min, x_max, y_max)

    Raises:
        ValueError: If required data is missing or invalid

    Example:
        >>> orp_data = import_board_from_orp("design.orp")
        >>> board_data = convert_orp_to_board_data(orp_data)
        >>> print(f"Board has {len(board_data['nets'])} nets")
    """
    # Extract metadata
    metadata = orp_data.get("metadata", {})
    board_name = metadata.get("board_name", "Unknown Board")

    # Extract board properties
    board_section = orp_data.get("board", {})
    bounds_data = board_section.get("bounds", {})
    layer_count = board_section.get("layer_count", 2)

    # Extract bounds
    x_min = bounds_data.get("x_min", 0.0)
    y_min = bounds_data.get("y_min", 0.0)
    x_max = bounds_data.get("x_max", 0.0)
    y_max = bounds_data.get("y_max", 0.0)
    width = bounds_data.get("width", x_max - x_min)
    height = bounds_data.get("height", y_max - y_min)

    # Extract layer names
    layers_section = orp_data.get("layers", [])
    layer_names = [layer.get("name", f"Layer{i}") for i, layer in enumerate(layers_section)]

    # If no layer names, generate defaults
    if not layer_names:
        if layer_count == 2:
            layer_names = ["F.Cu", "B.Cu"]
        else:
            layer_names = ["F.Cu"] + [f"In{i}.Cu" for i in range(1, layer_count - 1)] + ["B.Cu"]

    # Convert pads from ORP format to board_data format
    pads_section = orp_data.get("pads", [])
    pads = []

    for pad_data in pads_section:
        # ORP pads have nested position, need to flatten for board_data
        position_data = pad_data.get("position", {})
        size_data = pad_data.get("size", {})

        # Determine layers for pad
        layer = pad_data.get("layer", "F.Cu")
        drill_size = pad_data.get("drill_size")

        if drill_size and drill_size > 0:
            # Through-hole pad
            pad_layers = layer_names  # All layers
            pad_type = "thru_hole"
        else:
            # SMD pad - determine which layer
            if layer == "B.Cu":
                pad_layers = ["B.Cu"]
            else:
                pad_layers = ["F.Cu"]
            pad_type = "smd"

        pad = {
            'id': pad_data.get("id", ""),
            'component': pad_data.get("component_ref", ""),
            'name': pad_data.get("id", "").split("@")[-1] if "@" in pad_data.get("id", "") else "",
            'net_name': pad_data.get("net_name", ""),
            'net_code': 0,  # ORP doesn't store net codes
            'x': position_data.get("x", 0.0),
            'y': position_data.get("y", 0.0),
            'width': size_data.get("width", 0.0),
            'height': size_data.get("height", 0.0),
            'drill': drill_size,
            'layers': pad_layers,
            'type': pad_type,
        }
        pads.append(pad)

    # Convert nets from ORP format to board_data format
    nets_section = orp_data.get("nets", [])
    nets = {}

    for net_data in nets_section:
        net_name = net_data.get("name", "")
        net_id = net_data.get("id", net_name)

        # Find all pads belonging to this net
        net_pads = [p for p in pads if p['net_name'] == net_name]

        nets[net_name] = {
            'name': net_name,
            'code': 0,  # ORP doesn't store net codes
            'pads': net_pads,
        }

    # Extract DRC rules
    drc_rules = orp_data.get("drc_rules", {})
    default_rules = drc_rules.get("default", {})

    clearance = default_rules.get("clearance", 0.2)
    track_width = default_rules.get("track_width", 0.25)
    via_diameter = default_rules.get("via_diameter", 0.6)
    via_drill = default_rules.get("via_drill", 0.3)

    # Extract grid parameters
    grid_params = orp_data.get("grid_parameters", {})
    grid_resolution = grid_params.get("grid_pitch", 0.4)

    # Build components list (derive from pads)
    components_dict = {}
    for pad in pads:
        component_ref = pad['component']
        if component_ref and component_ref not in components_dict:
            components_dict[component_ref] = {
                'reference': component_ref,
                'value': '',
                'footprint': '',
                'x': pad['x'],
                'y': pad['y'],
            }

    components = list(components_dict.values())

    # Build board_data dictionary
    board_data = {
        'filename': board_name,
        'x_min': x_min,
        'y_min': y_min,
        'x_max': x_max,
        'y_max': y_max,
        'width': width,
        'height': height,
        'layers': layer_count,
        'layer_names': layer_names,
        'pads': pads,
        'nets': nets,
        'components': components,
        'clearance': clearance,
        'track_width': track_width,
        'via_diameter': via_diameter,
        'via_drill': via_drill,
        'grid_resolution': grid_resolution,
        'airwires': [],  # Empty for now
        'bounds': (x_min, y_min, x_max, y_max),
        'tracks': [],  # No existing tracks in ORP
        'zones': [],  # No zones in ORP
        'drc_rules': drc_rules,
    }

    return board_data


def convert_orp_to_board(orp_data: Dict[str, Any]):
    """
    Convert ORP dictionary data into a proper Board domain model object.

    This function transforms the dictionary data returned by import_board_from_orp()
    into a fully populated Board object with all domain entities properly constructed.

    Args:
        orp_data: Dictionary containing ORP data (from import_board_from_orp())

    Returns:
        Board: Fully populated Board domain model object

    Raises:
        ValueError: If required data is missing or invalid

    Example:
        >>> orp_data = import_board_from_orp("design.orp")
        >>> board = convert_orp_to_board(orp_data)
        >>> print(f"Board '{board.name}' has {len(board.nets)} nets")
    """
    # Import domain models here to avoid circular imports
    from orthoroute.domain.models.board import (
        Board, Component, Net, Pad, Layer, Coordinate, Bounds
    )

    # Extract metadata
    metadata = orp_data.get("metadata", {})
    board_name = metadata.get("board_name", "Unknown Board")
    board_id = metadata.get("board_id", str(uuid4()))

    # Extract board properties
    board_data = orp_data.get("board", {})
    layer_count = board_data.get("layer_count", 2)
    thickness = board_data.get("thickness", 1.6)

    # Build layers first
    layers = []
    layers_data = orp_data.get("layers", [])
    for layer_data in layers_data:
        layer = Layer(
            name=layer_data.get("name", "F.Cu"),
            type=layer_data.get("type", "signal"),
            stackup_position=layer_data.get("stackup_position", 0),
            thickness=layer_data.get("thickness", 0.035),
            material=layer_data.get("material", "copper")
        )
        layers.append(layer)

    # If no layers provided, create default layers
    if not layers:
        if layer_count == 2:
            layers = [
                Layer(name="F.Cu", type="signal", stackup_position=0),
                Layer(name="B.Cu", type="signal", stackup_position=1),
            ]
        else:
            layers = [Layer(name="F.Cu", type="signal", stackup_position=0)]
            for i in range(1, layer_count - 1):
                layers.append(Layer(name=f"In{i}.Cu", type="signal", stackup_position=i))
            layers.append(Layer(name="B.Cu", type="signal", stackup_position=layer_count - 1))

    # Build pads with their positions and sizes
    pads_data = orp_data.get("pads", [])
    pads_by_id = {}
    pads_by_component = {}  # Group pads by component_id

    for pad_data in pads_data:
        pad_id = pad_data.get("id", str(uuid4()))
        component_id = pad_data.get("component_id", "")
        net_id = pad_data.get("net_id")

        # Extract position
        position_data = pad_data.get("position", {})
        position = Coordinate(
            x=position_data.get("x", 0.0),
            y=position_data.get("y", 0.0)
        )

        # Extract size
        size_data = pad_data.get("size", {})
        size = (
            size_data.get("width", 0.0),
            size_data.get("height", 0.0)
        )

        # Create pad
        pad = Pad(
            id=pad_id,
            component_id=component_id,
            net_id=net_id,
            position=position,
            size=size,
            drill_size=pad_data.get("drill_size"),
            layer=pad_data.get("layer", "F.Cu"),
            shape=pad_data.get("shape", "circle"),
            angle=pad_data.get("angle", 0.0)
        )

        pads_by_id[pad_id] = pad

        # Group by component
        if component_id not in pads_by_component:
            pads_by_component[component_id] = []
        pads_by_component[component_id].append(pad)

    # Build components from grouped pads
    components = []
    components_by_id = {}

    for component_id, component_pads in pads_by_component.items():
        # Handle orphaned pads (empty component_id) by creating default component
        if not component_id:
            component_id = "ORPHANED_PADS"
            component_ref = "ORPHANED"
            # Update all orphaned pads to reference this component
            for pad in component_pads:
                pad.component_id = component_id
        else:
            # Get component reference from first pad (stored in pad data)
            component_ref = "U?"  # Default
            if component_pads:
                # Try to find component_ref in pad data
                for pad_data in pads_data:
                    if pad_data.get("component_id") == component_id:
                        component_ref = pad_data.get("component_ref", component_id)
                        break

        # Calculate component position as centroid of pads
        if component_pads:
            avg_x = sum(p.position.x for p in component_pads) / len(component_pads)
            avg_y = sum(p.position.y for p in component_pads) / len(component_pads)
            component_position = Coordinate(x=avg_x, y=avg_y)
        else:
            component_position = Coordinate(x=0.0, y=0.0)

        # Determine component layer (use layer of first pad)
        component_layer = component_pads[0].layer if component_pads else "F.Cu"

        component = Component(
            id=component_id,
            reference=component_ref,
            value="",  # Not stored in ORP format
            footprint="",  # Not stored in ORP format
            position=component_position,
            angle=0.0,  # Not stored in ORP format
            layer=component_layer,
            pads=component_pads
        )

        components.append(component)
        components_by_id[component_id] = component

    # Build nets with their pads
    nets = []
    nets_data = orp_data.get("nets", [])

    for net_data in nets_data:
        net_id = net_data.get("id", str(uuid4()))
        net_name = net_data.get("name", "")
        netclass = net_data.get("netclass", "Default")

        # Find all pads belonging to this net
        net_pads = []
        terminals = net_data.get("terminals", [])
        for terminal in terminals:
            pad_id = terminal.get("pad_id")
            if pad_id and pad_id in pads_by_id:
                pad = pads_by_id[pad_id]
                # Update pad's net_id to match this net
                pad.net_id = net_id
                net_pads.append(pad)

        # Create net
        net = Net(
            id=net_id,
            name=net_name,
            netclass=netclass,
            pads=net_pads
        )

        nets.append(net)

    # Create the Board object
    board = Board(
        id=board_id,
        name=board_name,
        components=components,
        nets=nets,
        layers=layers,
        thickness=thickness,
        layer_count=layer_count
    )

    # Set bounds from ORP data (used by UnifiedPathFinder for lattice creation)
    # _kicad_bounds should be a tuple (min_x, min_y, max_x, max_y)
    bounds_data = board_data.get("bounds", {})
    if bounds_data:
        board._kicad_bounds = (
            bounds_data.get("x_min", 0.0),
            bounds_data.get("y_min", 0.0),
            bounds_data.get("x_max", 0.0),
            bounds_data.get("y_max", 0.0)
        )

    # Set layer_names attribute (expected by UnifiedPathFinder)
    # The pathfinder expects board.layers to be a list of layer name strings
    board.layer_names = [layer.name for layer in layers] if layers else []

    # Validate board integrity
    issues = board.validate_integrity()
    if issues:
        # Log warnings but don't fail - some issues might be acceptable
        import logging
        logger = logging.getLogger(__name__)
        for issue in issues:
            logger.warning(f"Board integrity issue: {issue}")

    return board
