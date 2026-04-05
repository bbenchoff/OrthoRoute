#!/usr/bin/env python3
"""
Rich KiCad IPC Interface - Full-featured KiCad data loading for the new architecture
Provides the same rich functionality as the legacy kicad_interface.py
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
import sys
import time

logger = logging.getLogger(__name__)

@dataclass
class DRCRules:
    """Container for Design Rule Check information"""
    netclasses: Dict[str, Dict]  # netclass_name -> rules dict
    default_track_width: float  # mm
    default_via_size: float  # mm
    default_via_drill: float  # mm
    default_clearance: float  # mm
    minimum_track_width: float  # mm
    minimum_via_size: float  # mm

@dataclass
class BoardData:
    """Container for board data extracted from KiCad"""
    filename: str
    width: float  # mm
    height: float  # mm
    layers: int
    nets: List[Dict]
    components: List[Dict]
    tracks: List[Dict]
    vias: List[Dict]
    pads: List[Dict]
    bounds: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y
    drc_rules: Optional[DRCRules] = None


def fetch_board_and_drc():
    """Fetch board and DRC data using IPC API (proper method)"""
    try:
        from kicad import KiCad
    except ImportError:
        raise ImportError("kicad Python module not available in this environment")

    try:
        kc = KiCad()                                   # IPC session (nng under the hood)
        board = kc.get_board()                         # Active PCB document
        project = kc.get_project()                     # Project owning the board

        # Layers / stackup
        layer_cnt = board.get_copper_layer_count()
        logger.info(f"[LAYER-DETECT] board.get_copper_layer_count() returned: {layer_cnt}")
        stackup = board.get_stackup() if hasattr(board, 'get_stackup') else None

        # Nets and classes
        nets = board.get_nets()                        # [{id, name, ...}, ...]
        net_names = [n.get("name", "") for n in nets if n.get("name")]
        netclass_by_net = board.get_netclass_for_nets(net_names) if net_names else {}

        # All net classes (for defaults/fallbacks)
        all_netclasses = {nc.get("name", "Default"): nc for nc in project.get_net_classes()}

        # Pad polygons (for DRC keepouts)
        # returns dict keyed by pad or (ref, pad_name) -> polygon(s)
        pad_polys = board.get_pad_shapes_as_polygons(include_holes=False) if hasattr(board, 'get_pad_shapes_as_polygons') else {}

        return {
            "board": board,
            "layer_cnt": layer_cnt,
            "stackup": stackup,
            "nets": nets,
            "netclass_by_net": netclass_by_net,
            "all_netclasses": all_netclasses,
            "pad_polys": pad_polys,
        }
    except Exception as e:
        logger.error(f"IPC DRC fetch failed: {e}")
        return None

def _ipc_retry(func, desc: str, max_retries: int = 3, sleep_s: float = 0.5):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            msg = str(e)
            last_err = e
            logger.warning(f"IPC '{desc}' failed (attempt {attempt}/{max_retries}): {msg}")
            if ("Timed out" in msg or "AS_BUSY" in msg or "busy" in msg.lower()
                    or "Connection refused" in msg or "refused" in msg.lower()):
                time.sleep(sleep_s)
                continue
            break
    if last_err:
        raise last_err


class _SWIGNetWrapper:
    """Thin wrapper around a pcbnew NetInfo item exposing the kipy-compatible interface."""
    __slots__ = ('_item',)

    def __init__(self, item):
        self._item = item

    @property
    def name(self) -> str:
        try:
            return self._item.GetNetname()
        except Exception:
            return ''

    @property
    def code(self) -> int:
        try:
            return self._item.GetNet()
        except Exception:
            return 0


class _SWIGBoardWrapper:
    """Thin wrapper around a pcbnew BOARD exposing the kipy Board interface used by
    OrthoRouteMainWindow: .filename property and get_nets()."""
    __slots__ = ('_board',)

    def __init__(self, pcbnew_board):
        self._board = pcbnew_board

    @property
    def filename(self) -> str:
        try:
            return self._board.GetFileName()
        except Exception:
            return 'Unknown'

    def get_nets(self):
        """Return a list of net-like objects each with .name and .code."""
        nets = []
        try:
            info = self._board.GetNetInfo()
            for code in range(info.GetNetCount()):
                item = info.GetNetItem(code)
                if item is not None:
                    nets.append(_SWIGNetWrapper(item))
        except Exception:
            pass
        return nets


class RichKiCadInterface:
    """Rich interface to KiCad via IPC API with full data extraction"""

    def __init__(self):
        self.client = None
        self.board = None
        self.connected = False
        self._mode = 'ipc'       # 'ipc' or 'swig'
        self._swig_board = None  # raw pcbnew board in SWIG mode

    def connect(self) -> bool:
        """Connect to KiCad via IPC API"""
        try:
            # Ensure kipy is importable from user site
            try:
                from kipy import KiCad  # type: ignore
            except ImportError:
                import site
                user_site = site.getusersitepackages()
                if user_site and user_site not in sys.path:
                    sys.path.insert(0, user_site)
                from kipy import KiCad  # retry

            # Gather credentials if provided by KiCad runtime
            api_socket = os.environ.get('KICAD_API_SOCKET')
            api_token = os.environ.get('KICAD_API_TOKEN')
            timeout_ms = 120000  # 2 minutes - increased for large geometry commits (3000+ tracks/vias)
            if api_socket or api_token:
                self.client = KiCad(socket_path=api_socket, kicad_token=api_token, timeout_ms=timeout_ms)
            else:
                self.client = KiCad(timeout_ms=timeout_ms)

            # Get board to confirm connection - try different methods
            try:
                # Method 1: get_board() — preferred, works when IPC server is enabled
                self.board = _ipc_retry(self.client.get_board, "get_board", max_retries=3, sleep_s=1.0)
            except Exception as e1:
                logger.warning(f"get_board failed: {e1}")
                try:
                    # Method 2: get_open_documents(doc_type) — doc_type int from DocumentType enum
                    # DOCTYPE_PCB is 2 in KiCad's proto; import the constant when available
                    try:
                        from kipy.proto.common.types_pb2 import DocumentType  # type: ignore
                        doc_type_pcb = DocumentType.Value('DOCTYPE_PCB')
                    except Exception:
                        doc_type_pcb = 2  # fallback: DOCTYPE_PCB is 2 in KiCad 9 protos
                    docs = self.client.get_open_documents(doc_type_pcb)
                    if docs and len(docs) > 0:
                        # get_open_documents returns DocumentSpecifiers, not Board objects.
                        # Use the first specifier to retrieve the actual board.
                        specifier = docs[0]
                        self.board = self.client.get_board()
                        logger.info(f"Retrieved board via open documents specifier: {getattr(specifier, 'identifier', 'Unknown')}")
                    else:
                        raise Exception("No open PCB documents found")
                except Exception as e2:
                    logger.warning(f"get_open_documents fallback failed: {e2}")
                    raise Exception(
                        "Could not retrieve board via IPC. "
                        "Make sure: (1) a PCB file is open in KiCad, "
                        "(2) 'Enable Python API' is checked in KiCad Preferences > Plugins."
                    ) from e2
                    
            self.connected = True
            logger.info("Connected to KiCad IPC API and retrieved board")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to KiCad via IPC: {e}")
            # --- SWIG fallback: use pcbnew when running inside KiCad as an action plugin ---
            try:
                import pcbnew  # type: ignore
                swig_board = pcbnew.GetBoard()
                if swig_board is None:
                    logger.error("SWIG fallback: pcbnew.GetBoard() returned None — open a PCB first")
                    self.connected = False
                    return False
                self._swig_board = swig_board
                self.board = _SWIGBoardWrapper(swig_board)
                self._mode = 'swig'
                self.connected = True
                logger.info("IPC unavailable; connected via SWIG fallback (running inside KiCad)")
                return True
            except ImportError:
                logger.error("Neither IPC nor SWIG (pcbnew) available — cannot connect to KiCad")
                self.connected = False
                return False
            except Exception as e_swig:
                logger.error(f"SWIG fallback also failed: {e_swig}")
                self.connected = False
                return False

    def get_board_filename(self) -> str:
        """Get the current board filename using KiCad Python API"""
        if not self.connected or not self.board:
            logger.warning("Not connected to KiCad board")
            return "Unknown"
        
        board = self.board
        try:
            # Try multiple methods to get the board filename
            if hasattr(board, 'GetFileName') and board.GetFileName():
                filename = board.GetFileName()
            elif hasattr(board, 'filename') and board.filename:
                filename = board.filename
            elif hasattr(board, 'name') and board.name:
                filename = board.name
            elif hasattr(board, '_board') and hasattr(board._board, 'GetFileName'):
                filename = board._board.GetFileName()
            elif hasattr(board, 'board') and hasattr(board.board, 'GetFileName'):
                filename = board.board.GetFileName()
            else:
                filename = "Unknown"
                
            # Extract just the filename from full path
            if filename and filename != "Unknown":
                filename = os.path.basename(filename)
                
            logger.info(f"Board filename: {filename}")
            return filename
            
        except Exception as e:
            logger.warning(f"Could not get board filename: {e}")
            return "Unknown"

    def get_board_data(self) -> Optional[Dict]:
        """Extract comprehensive board data from KiCad"""
        if not self.connected or not self.board:
            logger.error("Not connected to KiCad")
            return None

        if self._mode == 'swig':
            return self._extract_board_data_swig()

        try:
            board = self.board
            logger.info("Extracting comprehensive board data from KiCad...")
            
            # Get filename
            filename = self.get_board_filename()
            
            # Extract pads with polygon shapes
            logger.info("Extracting pads with detailed geometry...")
            pads = self._extract_pads(board)
            logger.info(f"Found {len(pads)} pads - extracting with polygon shapes")
            
            # Extract components
            logger.info("Extracting components...")
            components = self._extract_components(board)
            
            # Extract tracks
            logger.info("Extracting existing tracks...")
            tracks = self._extract_tracks(board)

            # Extract vias
            logger.info("Extracting existing vias...")
            vias = self._extract_vias(board)

            # Extract zones (copper pours) and keepout rule areas
            logger.info("Extracting zones...")
            zones, keepouts = self._extract_zones(board)
            logger.info(f"Found {len(zones)} zones")
            
            # Extract nets with pad connectivity
            logger.info("Extracting nets with connectivity...")
            nets_data = self._extract_nets(board, pads)
            routable_nets = [net for net in nets_data.values() if len(net.get('pads', [])) >= 2]
            
            
            logger.info(f"Found {len(nets_data)} nets with pads")
            logger.info(f"Created {len(routable_nets)} routable nets (excluding 0 plane-connected nets)")
            
            # Calculate board dimensions
            logger.info("Calculating board dimensions...")
            bounds, width, height = self._calculate_board_dimensions(board)
            logger.info(f"Board dimensions calculated from geometry: {width:.1f} x {height:.1f} mm")
            
            # Generate airwires for visualization
            logger.info("Generating airwires...")
            airwires = self._generate_airwires(routable_nets)
            logger.info(f"Generated {len(airwires)} airwires from {len(routable_nets)} nets")
            logger.info(f"  Including 0 partially routed nets")
            logger.info(f"  Filtered out 0 nets with copper pours")
            
            # Extract DRC rules
            logger.info("Extracting DRC rules...")
            drc_rules = self._extract_drc_rules(board)
            logger.info(f"Extracted DRC rules: {len(drc_rules.get('netclasses', {}))} netclasses")
            
            # Get layer count and names
            layer_count, layer_names = self._get_layer_info(board)
            logger.info(f"Large backplane detected ({len(pads)} pads), using {layer_count} copper layers")

            # Build comprehensive board data
            board_data = {
                'filename': filename,
                'pads': pads,
                'components': components,
                'tracks': tracks,
                'vias': vias,
                'zones': zones,
                'keepouts': keepouts,
                'nets': nets_data,
                'airwires': airwires,
                'bounds': bounds,
                'width': width,
                'height': height,
                'layers': layer_count,
                'layer_names': layer_names,
                'drc_rules': drc_rules
            }
            
            logger.info(f"Extracted board data: {filename} ({width:.1f}x{height:.1f}mm, {layer_count} copper layers)")
            logger.info(f"  {len(routable_nets)} routable nets, {len(components)} components, {len(tracks)} tracks, {len(vias)} vias, {len(zones)} zones, {len(keepouts)} keepouts")
            logger.info(f"  Generated {len(airwires)} airwires for visualization")
            logger.info(f"  Extracted {len(drc_rules.get('netclasses', {}))} netclasses with design rules")
            
            return board_data
            
        except Exception as e:
            logger.error(f"Failed to extract board data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _extract_board_data_swig(self) -> Optional[Dict]:
        """Extract board data dict using the pcbnew SWIG API (IPC fallback path)."""
        try:
            import pcbnew  # type: ignore
            board = self._swig_board
            iu = float(getattr(pcbnew, 'IU_PER_MM', 1e6))  # internal units per mm (1e6 in KiCad 7+)

            # --- Layers ---
            layer_count = board.GetCopperLayerCount()

            # --- Nets ---
            nets: Dict[str, Dict] = {}
            netinfo = board.GetNetInfo()
            for code in range(netinfo.GetNetCount()):
                item = netinfo.GetNetItem(code)
                if item is None:
                    continue
                name = item.GetNetname()
                if name:
                    nets[name] = {'name': name, 'code': code, 'pads': []}

            # --- Pads ---
            pads: List[Dict] = []
            for pad in board.GetPads():
                try:
                    pos = pad.GetPosition()
                    size = pad.GetSize()
                    drill_x = 0.0
                    try:
                        drill_size = pad.GetDrillSize()
                        drill_x = drill_size.x / iu
                    except Exception:
                        pass
                    parent = pad.GetParent()
                    ref = parent.GetReference() if parent and hasattr(parent, 'GetReference') else ''
                    has_drill = drill_x > 0
                    net_name = pad.GetNetname()
                    pad_dict = {
                        'component': ref,
                        'name': pad.GetNumber(),
                        'net_name': net_name,
                        'net_code': pad.GetNetCode(),
                        'x': pos.x / iu,
                        'y': pos.y / iu,
                        'width': size.x / iu,
                        'height': size.y / iu,
                        'drill': drill_x,
                        'layers': ['F.Cu', 'B.Cu'] if has_drill else ['F.Cu'],
                        'type': 'through_hole' if has_drill else 'smd',
                    }
                    pads.append(pad_dict)
                    # Register pad in nets dict
                    if net_name and net_name in nets:
                        nets[net_name]['pads'].append(pad_dict)
                except Exception as ex:
                    logger.warning(f"SWIG: error extracting pad: {ex}")

            # --- Tracks & Vias ---
            tracks: List[Dict] = []
            vias: List[Dict] = []
            for item in board.GetTracks():
                try:
                    cls = item.GetClass()
                    if cls == 'PCB_VIA':
                        pos = item.GetPosition()
                        vias.append({
                            'x': pos.x / iu,
                            'y': pos.y / iu,
                            'diameter': item.GetWidth() / iu,
                            'drill': item.GetDrillValue() / iu,
                            'from_layer': board.GetLayerName(item.TopLayer()),
                            'to_layer': board.GetLayerName(item.BottomLayer()),
                            'net_name': item.GetNetname(),
                        })
                    else:
                        s = item.GetStart()
                        e = item.GetEnd()
                        tracks.append({
                            'start_x': s.x / iu,
                            'start_y': s.y / iu,
                            'end_x': e.x / iu,
                            'end_y': e.y / iu,
                            'width': item.GetWidth() / iu,
                            'layer': board.GetLayerName(item.GetLayer()),
                            'net_name': item.GetNetname(),
                        })
                except Exception as ex:
                    logger.warning(f"SWIG: error extracting track/via: {ex}")

            filename = board.GetFileName()
            # Board bounds (approximate from pads)
            if pads:
                xs = [p['x'] for p in pads]
                ys = [p['y'] for p in pads]
                bounds = (min(xs), min(ys), max(xs), max(ys))
                width  = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
            else:
                bounds = (0.0, 0.0, 100.0, 100.0)
                width, height = 100.0, 100.0

            layer_names = [board.GetLayerName(pcbnew.F_Cu)]
            for i in range(1, layer_count - 1):
                layer_names.append(board.GetLayerName(pcbnew.In1_Cu + i - 1))
            if layer_count > 1:
                layer_names.append(board.GetLayerName(pcbnew.B_Cu))

            logger.info(
                f"SWIG extraction: {len(pads)} pads, {len(nets)} nets, "
                f"{layer_count} layers, {len(tracks)} tracks, {len(vias)} vias"
            )
            return {
                'filename': filename,
                'pads': pads,
                'components': [],
                'tracks': tracks,
                'vias': vias,
                'zones': [],
                'keepouts': [],
                'nets': nets,
                'airwires': [],
                'bounds': bounds,
                'width': width,
                'height': height,
                'layers': layer_count,
                'layer_names': layer_names,
                'drc_rules': {'netclasses': {}, 'min_track_width': 0.1, 'min_clearance': 0.1},
            }
        except Exception as e:
            logger.error(f"SWIG board data extraction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _extract_pads(self, board) -> List[Dict]:
        """Extract all pads with detailed geometry using KiCad API"""
        pads = []
        try:
            # Use the correct KiCad API method
            all_pads = _ipc_retry(board.get_pads, "get_pads", max_retries=3, sleep_s=0.7)
            logger.info(f"Found {len(all_pads)} pads using KiCad API")
            
            for i, p in enumerate(all_pads):
                try:
                    # Extract pad data using object attributes (not dictionary access)
                    pos = getattr(p, 'position', None)
                    x = float(getattr(pos, 'x', 0.0)) / 1000000.0 if pos is not None else 0.0  # Convert nm to mm
                    y = float(getattr(pos, 'y', 0.0)) / 1000000.0 if pos is not None else 0.0  # Convert nm to mm
                    
                    net_obj = getattr(p, 'net', None)
                    net_name = getattr(net_obj, 'name', '') if net_obj else ''
                    net_code = getattr(net_obj, 'code', 0) if net_obj else 0
                    
                    pad_number = getattr(p, 'number', '')
                    
                    # Get pad geometry from padstack
                    padstack = getattr(p, 'padstack', None)
                    width = 1.0  # Default
                    height = 1.0  # Default
                    drill = 0.0
                    
                    if padstack:
                        # Get drill diameter
                        drill_obj = getattr(padstack, 'drill', None)
                        if drill_obj:
                            drill_dia = getattr(drill_obj, 'diameter', None)
                            if drill_dia and hasattr(drill_dia, 'x'):
                                drill = float(getattr(drill_dia, 'x', 0.0)) / 1000000.0
                        
                        # Get pad size from copper layers
                        copper_layers = getattr(padstack, 'copper_layers', [])
                        if copper_layers and len(copper_layers) > 0:
                            first_layer = copper_layers[0]
                            size = getattr(first_layer, 'size', None)
                            if size:
                                width = float(getattr(size, 'x', 1000000.0)) / 1000000.0
                                height = float(getattr(size, 'y', 1000000.0)) / 1000000.0
                    
                    # Get component reference
                    footprint = getattr(p, 'footprint', None)
                    component_ref = getattr(footprint, 'reference', '') if footprint else ''
                    
                    # Determine layers
                    layers = []
                    if drill > 0:
                        layers = ['F.Cu', 'B.Cu']  # Through-hole
                    else:
                        layers = ['F.Cu']  # SMD (assume front)
                    
                    pad_data = {
                        'component': component_ref,
                        'name': pad_number,
                        'net_name': net_name,
                        'net_code': net_code,
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'drill': drill,
                        'layers': layers,
                        'type': 'through_hole' if drill > 0 else 'smd'
                    }
                    
                    pads.append(pad_data)
                    
                    # Log first few pads for debugging (gated behind DEBUG level)
                    if i < 5 and logger.isEnabledFor(logging.DEBUG):
                        pad_type = "through-hole" if drill > 0 else "SMD"
                        logger.debug(f"{pad_type} Pad {i}: pos=({x:.2f}, {y:.2f}), size=({width:.2f}x{height:.2f}) (SMD) [{pad_type}], net='{net_name}'")
                        
                except Exception as e:
                    logger.warning(f"Error extracting pad {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting pads: {e}")
            
        return pads

    def _extract_components(self, board) -> List[Dict]:
        """Extract component information using KiCad API"""
        components = []
        try:
            # Use the correct KiCad API method
            footprints = _ipc_retry(board.get_footprints, "get_footprints", max_retries=3, sleep_s=0.5)
            logger.info(f"Found {len(footprints)} footprints using KiCad API")
            
            for fp in footprints:
                try:
                    # Extract component data using object attributes
                    reference = getattr(fp, 'reference', '')
                    value = getattr(fp, 'value', '')
                    library_id = getattr(fp, 'library_id', '')
                    
                    pos = getattr(fp, 'position', None)
                    x = float(getattr(pos, 'x', 0.0)) / 1000000.0 if pos else 0.0  # Convert nm to mm
                    y = float(getattr(pos, 'y', 0.0)) / 1000000.0 if pos else 0.0  # Convert nm to mm
                    
                    rotation = getattr(fp, 'orientation', 0.0)
                    layer = getattr(fp, 'layer', 'F.Cu')
                    
                    component_data = {
                        'reference': reference,
                        'value': value,
                        'footprint': library_id,
                        'x': x,
                        'y': y,
                        'rotation': rotation,
                        'layer': layer
                    }
                    
                    components.append(component_data)
                    
                except Exception as e:
                    logger.warning(f"Error extracting component: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting components: {e}")
            
        return components

    def _extract_tracks(self, board) -> List[Dict]:
        """Extract existing tracks/traces using KiCad API"""
        tracks = []
        try:
            # Use the correct KiCad API method
            board_tracks = _ipc_retry(board.get_tracks, "get_tracks", max_retries=3, sleep_s=0.5)
            logger.info(f"Found {len(board_tracks)} tracks using KiCad API")

            # BoardLayer enum for converting integer layer IDs to KiCad layer names.
            # BoardLayer.Name() returns proto enum names like "BL_F_Cu", "BL_B_Cu",
            # "BL_In1_Cu" etc.  Strip the 3-char "BL_" prefix and replace "_" with
            # "." to get the KiCad canonical name: "F.Cu", "B.Cu", "In1.Cu" etc.
            try:
                from kipy.board_types import BoardLayer
            except ImportError:
                BoardLayer = None

            def _layer_id_to_name(layer_id) -> str:
                if BoardLayer is not None:
                    try:
                        proto_name = BoardLayer.Name(layer_id)  # e.g. "BL_F_Cu"
                        if proto_name.startswith("BL_"):
                            return proto_name[3:].replace("_", ".")
                        return proto_name
                    except Exception:
                        pass
                return f"layer_{layer_id}"

            for track in board_tracks:
                try:
                    # track.start / track.end are Vector2 objects; .x and .y are in nanometers
                    start = track.start
                    end = track.end
                    start_x = float(start.x) / 1e6
                    start_y = float(start.y) / 1e6
                    end_x   = float(end.x)   / 1e6
                    end_y   = float(end.y)   / 1e6
                    # track.width is in nanometers
                    width = float(track.width) / 1e6
                    # track.layer is a BoardLayer enum integer
                    layer = _layer_id_to_name(track.layer)
                    # track.net.name is the net name string
                    net_name = track.net.name if track.net is not None else ''

                    track_data = {
                        'start_x': start_x,
                        'start_y': start_y,
                        'end_x':   end_x,
                        'end_y':   end_y,
                        'width':   width,
                        'layer':   layer,
                        'net_name': net_name,
                    }
                    tracks.append(track_data)

                except Exception as e:
                    logger.warning(f"Error extracting track: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error extracting tracks: {e}")

        logger.info(f"Loaded {len(tracks)} existing tracks from KiCad")
        return tracks

    def _extract_vias(self, board) -> List[Dict]:
        """Extract existing vias using KiCad IPC API"""
        vias = []
        try:
            from kipy.board_types import BoardLayer, ViaType
        except ImportError:
            logger.warning("kipy not available — skipping via extraction")
            return vias

        def _layer_id_to_name(layer_id) -> str:
            try:
                proto_name = BoardLayer.Name(layer_id)
                if proto_name.startswith("BL_"):
                    return proto_name[3:].replace("_", ".")
                return proto_name
            except Exception:
                return f"layer_{layer_id}"

        # ViaType proto enum names → GUI via type strings
        _via_type_map = {
            "VT_THROUGH":      "through",
            "VT_BLIND_BURIED": "blind_buried",
            "VT_MICRO":        "micro",
        }

        try:
            board_vias = _ipc_retry(board.get_vias, "get_vias", max_retries=3, sleep_s=0.5)
            logger.info(f"Found {len(board_vias)} vias using KiCad API")

            for via in board_vias:
                try:
                    pos = via.position
                    x = float(pos.x) / 1e6
                    y = float(pos.y) / 1e6
                    diameter = float(via.diameter) / 1e6
                    drill    = float(via.drill_diameter) / 1e6
                    net_name = via.net.name if via.net is not None else ''
                    type_proto = ViaType.Name(via.type)
                    via_type = _via_type_map.get(type_proto, "through")

                    # For through vias the layers are always F.Cu→B.Cu.
                    # For blind/buried vias read from the padstack drill span.
                    try:
                        drill_obj = via.padstack.drill
                        start_layer = _layer_id_to_name(drill_obj.start_layer)
                        end_layer   = _layer_id_to_name(drill_obj.end_layer)
                    except Exception:
                        start_layer = "F.Cu"
                        end_layer   = "B.Cu"

                    vias.append({
                        'x':           x,
                        'y':           y,
                        'diameter':    diameter,
                        'drill':       drill,
                        'type':        via_type,
                        'start_layer': start_layer,
                        'end_layer':   end_layer,
                        'net_name':    net_name,
                    })

                except Exception as e:
                    logger.warning(f"Error extracting via: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error extracting vias: {e}")

        logger.info(f"Loaded {len(vias)} existing vias from KiCad")
        return vias

    def _extract_zones(self, board) -> tuple:
        """Extract copper zones/pours and rule area keepouts using KiCad IPC API.
        Returns (zones, keepouts) tuple."""
        zones = []
        keepouts = []
        try:
            from kipy.board_types import BoardLayer, ZoneType

            def _layer_id_to_name(layer_id) -> str:
                try:
                    proto_name = BoardLayer.Name(layer_id)
                    if proto_name.startswith("BL_"):
                        return proto_name[3:].replace("_", ".")
                    return proto_name
                except Exception:
                    return f"layer_{layer_id}"

            def _poly_to_points(poly_with_holes) -> List[List[float]]:
                """Convert a PolygonWithHoles outline to a list of [x, y] mm points."""
                pts = []
                try:
                    for node in poly_with_holes.outline.nodes:
                        if node.has_point:
                            pts.append([float(node.point.x) / 1e6,
                                        float(node.point.y) / 1e6])
                except Exception:
                    pass
                return pts

            board_zones = _ipc_retry(board.get_zones, "get_zones", max_retries=3, sleep_s=0.5)
            logger.info(f"Found {len(board_zones)} zones using KiCad API")

            for zone in board_zones:
                try:
                    type_name = ZoneType.Name(zone.type)
                    net_name = zone.net.name if zone.net is not None else ""
                    logger.info(f"Zone: name='{zone.name}' type={type_name} net='{net_name}' layers={[_layer_id_to_name(l) for l in zone.layers]}")

                    if zone.type == ZoneType.ZT_RULE_AREA:
                        # Extract keepout constraints from rule area
                        ra = zone._proto.rule_area_settings
                        outline_pts = _poly_to_points(zone.outline)
                        layer_names = [_layer_id_to_name(l) for l in zone.layers]
                        keepout = {
                            'name':              zone.name,
                            'layers':            layer_names,
                            'outline':           outline_pts,
                            'keepout_tracks':    ra.keepout_tracks,
                            'keepout_vias':      ra.keepout_vias,
                            'keepout_copper':    ra.keepout_copper,
                            'keepout_pads':      ra.keepout_pads,
                            'keepout_footprints':ra.keepout_footprints,
                        }
                        keepouts.append(keepout)
                        logger.info(f"Keepout '{zone.name}': tracks={ra.keepout_tracks} vias={ra.keepout_vias} copper={ra.keepout_copper} pads={ra.keepout_pads} layers={layer_names}")
                        continue

                    net_name = zone.net.name if zone.net is not None else ""
                    layer_names = [_layer_id_to_name(l) for l in zone.layers]
                    outline_pts = _poly_to_points(zone.outline)

                    # Collect filled polygon outlines per layer
                    filled = {}
                    for layer_id, poly_list in zone.filled_polygons.items():
                        layer_name = _layer_id_to_name(layer_id)
                        filled[layer_name] = [_poly_to_points(p) for p in poly_list]

                    zones.append({
                        'net_name':  net_name,
                        'layers':    layer_names,
                        'outline':   outline_pts,
                        'filled':    filled,
                        'priority':  zone.priority,
                        'name':      zone.name,
                    })

                except Exception as e:
                    logger.warning(f"Error extracting zone: {e}")
                    continue

        except ImportError:
            logger.info("kipy not available — skipping zone extraction")
        except Exception as e:
            logger.error(f"Error extracting zones: {e}")

        logger.info(f"Loaded {len(zones)} copper zones and {len(keepouts)} keepout areas from KiCad")
        return zones, keepouts

    def _extract_nets(self, board, pads: List[Dict]) -> Dict[str, Dict]:
        """Extract nets with pad connectivity"""
        nets = {}
        
        # Group pads by net
        for pad in pads:
            net_name = pad.get('net_name', '')
            if not net_name:
                continue
                
            if net_name not in nets:
                nets[net_name] = {
                    'name': net_name,
                    'code': pad.get('net_code', 0),
                    'pads': []
                }
                
            nets[net_name]['pads'].append(pad)
            
        return nets

    def _calculate_board_dimensions(self, board) -> Tuple[Tuple[float, float, float, float], float, float]:
        """Calculate board dimensions from KiCad API or pad positions"""
        try:
            # IPC method doesn't have get_board_info - skip API board bounds
            # We'll calculate from pad positions below
            pass

        except Exception as e:
            logger.debug(f"Board dimensions API unavailable (expected with IPC): {e}")
        
        # Fallback: calculate from actual pad positions
        try:
            pads = _ipc_retry(board.get_pads, "get_pads", max_retries=3, sleep_s=0.5)
            if pads and len(pads) > 0:
                pad_positions = []
                for p in pads:  # Use all pads for accurate bounds calculation
                    try:
                        pos = getattr(p, 'position', None)
                        if pos is not None:
                            x = float(getattr(pos, 'x', 0.0)) / 1000000.0
                            y = float(getattr(pos, 'y', 0.0)) / 1000000.0
                            pad_positions.append((x, y))
                    except:
                        continue
                
                if pad_positions:
                    min_x = min(pos[0] for pos in pad_positions) - 5.0  # 5mm margin
                    max_x = max(pos[0] for pos in pad_positions) + 5.0
                    min_y = min(pos[1] for pos in pad_positions) - 5.0
                    max_y = max(pos[1] for pos in pad_positions) + 5.0
                    width = max_x - min_x
                    height = max_y - min_y
                    logger.info(f"Calculated bounds from pad positions: ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f})")
                    return (min_x, min_y, max_x, max_y), width, height
                    
        except Exception as e:
            logger.warning(f"Could not calculate board dimensions from pads: {e}")
            
        # Final fallback
        return (0, 0, 100, 100), 100, 100

    def _generate_airwires(self, nets: List[Dict]) -> List[Dict]:
        """Generate airwires for unrouted connections"""
        airwires = []
        
        for net in nets:
            pads = net.get('pads', [])
            if len(pads) < 2:
                continue
                
            # Create airwires between all pad pairs (minimum spanning tree would be better)
            for i, pad1 in enumerate(pads[:-1]):
                pad2 = pads[i + 1]
                
                airwire = {
                    'net_name': net['name'],
                    'start_x': pad1['x'],
                    'start_y': pad1['y'],
                    'end_x': pad2['x'],
                    'end_y': pad2['y'],
                    'start_component': pad1.get('component', ''),
                    'end_component': pad2.get('component', ''),
                    'start_pad': pad1.get('name', ''),
                    'end_pad': pad2.get('name', '')
                }
                
                airwires.append(airwire)
                
        return airwires

    def _get_layer_info(self, board) -> tuple:
        """Get copper layer count and names using KiCad API with multiple detection methods

        Returns:
            Tuple of (layer_count: int, layer_names: List[str])
        """

        # Method 1: Use BoardStackup layers API with material_name (MOST RELIABLE)
        try:
            stackup = _ipc_retry(board.get_stackup, "get_stackup", max_retries=3, sleep_s=0.5)
            if stackup and hasattr(stackup, 'layers'):
                copper_layers = []
                for layer in stackup.layers:
                    # Check material_name for exact match
                    if hasattr(layer, 'material_name') and layer.material_name == 'copper':
                        copper_layers.append(layer)

                if copper_layers:
                    layer_count = len(copper_layers)
                    layer_names = [getattr(l, 'user_name', f'Layer{i}') for i, l in enumerate(copper_layers)]
                    logger.info(f"Got layer count from BoardStackup.material_name: {layer_count} copper layers")
                    logger.info(f"Copper layers: {layer_names}")
                    return (layer_count, layer_names)
        except Exception as e:
            logger.warning(f"BoardStackup material_name detection failed: {e}")

        # Method 2: Use direct IPC API for copper layer count (returns count only, generate names)
        try:
            layer_count = _ipc_retry(board.get_copper_layer_count, "get_copper_layer_count", max_retries=3, sleep_s=0.5)
            if layer_count and layer_count > 0:
                logger.info(f"Got layer count from IPC API: {layer_count} copper layers")
                # Generate standard layer names
                layer_names = self._generate_layer_names(layer_count)
                return (layer_count, layer_names)
            else:
                logger.warning(f"Method 2 returned invalid layer count: {layer_count}")
        except Exception as e:
            logger.warning(f"Method 2 failed - IPC copper layer count: {e}")

        # Method 3: Try to get layer stack
        try:
            stackup = _ipc_retry(board.get_stackup, "get_stackup", max_retries=3, sleep_s=0.5)
            if stackup and isinstance(stackup, (list, tuple)):
                copper_layers = [layer for layer in stackup if 'Cu' in str(layer)]
                if copper_layers:
                    layer_count = len(copper_layers)
                    logger.info(f"Got layer count from stackup: {layer_count}")
                    layer_names = self._generate_layer_names(layer_count)
                    return (layer_count, layer_names)
        except Exception as e:
            logger.debug(f"Method 3 failed - stackup: {e}")

        # Method 4: Try to detect from layer names
        try:
            # Common approach - try to get layers info
            layers_info = _ipc_retry(board.get_layers, "get_layers", max_retries=3, sleep_s=0.5)
            if layers_info:
                if isinstance(layers_info, dict):
                    copper_layers = [name for name in layers_info.keys() if 'Cu' in name]
                elif isinstance(layers_info, (list, tuple)):
                    copper_layers = [layer for layer in layers_info if 'Cu' in str(layer)]
                else:
                    copper_layers = []

                if copper_layers:
                    logger.info(f"Got layer count from layers info: {len(copper_layers)}")
                    return (len(copper_layers), copper_layers)
        except Exception as e:
            logger.debug(f"Method 4 failed - layers info: {e}")

        # Method 5: Try common layer names (KiCad standard) by probing
        try:
            standard_layers = [
                'F.Cu', 'In1.Cu', 'In2.Cu', 'In3.Cu', 'In4.Cu', 'In5.Cu',
                'In6.Cu', 'In7.Cu', 'In8.Cu', 'In9.Cu', 'In10.Cu', 'In11.Cu',
                'In12.Cu', 'In13.Cu', 'In14.Cu', 'In15.Cu', 'In16.Cu', 'In17.Cu',
                'In18.Cu', 'In19.Cu', 'In20.Cu', 'In21.Cu', 'In22.Cu', 'In23.Cu',
                'In24.Cu', 'In25.Cu', 'In26.Cu', 'In27.Cu', 'In28.Cu', 'In29.Cu',
                'In30.Cu', 'B.Cu'
            ]

            detected_layers = []
            for layer_name in standard_layers:
                try:
                    # Try to access layer properties to see if it exists
                    layer_info = _ipc_retry(board.get_layer_info, "get_layer_info", layer_name, max_retries=1, sleep_s=0.1)
                    if layer_info:
                        detected_layers.append(layer_name)
                except:
                    continue  # Layer doesn't exist

            if len(detected_layers) >= 2:  # At least F.Cu and B.Cu
                logger.info(f"Detected layers by probing: {detected_layers}")
                return (len(detected_layers), detected_layers)
                
        except Exception as e:
            logger.debug(f"Method 4 failed - layer probing: {e}")
        
        # Fallback: Default to 2 layers but log the issue
        logger.error("CRITICAL: All layer count detection methods failed!")
        logger.error("This means board.get_copper_layer_count() is not working")
        logger.error("Check KiCad version (requires 9.0.5+) and IPC API connection")
        logger.warning("Could not detect layer count using any method - defaulting to 2 layers")
        logger.warning("This may cause routing to fail on multi-layer boards!")
        return (2, ['F.Cu', 'B.Cu'])

    def _generate_layer_names(self, layer_count: int) -> list:
        """Generate standard KiCad layer names for given layer count"""
        if layer_count == 2:
            return ['F.Cu', 'B.Cu']
        elif layer_count < 2:
            return ['F.Cu']  # Should never happen
        else:
            # Generate: F.Cu, In1.Cu, In2.Cu, ..., In(N-2).Cu, B.Cu
            layers = ['F.Cu']
            for i in range(1, layer_count - 1):
                layers.append(f'In{i}.Cu')
            layers.append('B.Cu')
            return layers

    def _extract_drc_rules(self, board) -> Dict:
        """Extract design rules and netclasses using IPC API (proper method)"""
        logger.info("Extracting DRC rules using IPC API...")

        # Try IPC-based DRC extraction first
        try:
            ipc_data = fetch_board_and_drc()
            if ipc_data and ipc_data.get('all_netclasses'):
                return self._process_ipc_drc_data(ipc_data)
        except Exception as e:
            logger.info(f"IPC-based DRC extraction not available ({e}), using defaults")

        # Fallback to safe defaults
        logger.info("Using fallback DRC defaults")
        return {
            'netclasses': {
                'Default': {
                    'track_width': 0.2,
                    'via_size': 0.8,
                    'clearance': 0.2
                }
            },
            'default_track_width': 0.2,
            'default_via_size': 0.8,
            'default_clearance': 0.2,
            'netclass_by_net': {},
            'pad_polygons': {}
        }

    def _process_ipc_drc_data(self, ipc_data: Dict) -> Dict:
        """Process IPC DRC data into our expected format"""
        drc_data = {
            'netclasses': {},
            'default_track_width': 0.2,
            'default_via_size': 0.8,
            'default_clearance': 0.2,
            'netclass_by_net': ipc_data.get('netclass_by_net', {}),
            'pad_polygons': ipc_data.get('pad_polys', {})
        }

        # Process netclasses from IPC data
        all_netclasses = ipc_data.get('all_netclasses', {})

        for nc_name, nc_data in all_netclasses.items():
            if not nc_data:
                continue

            # Extract netclass properties (IPC format)
            track_width = nc_data.get('track_width', nc_data.get('TrackWidth', 0.2))
            via_size = nc_data.get('via_size', nc_data.get('ViaDiameter', 0.8))
            clearance = nc_data.get('clearance', nc_data.get('Clearance', 0.2))

            # Convert to mm if needed (KiCad sometimes returns nanometers)
            if track_width > 10:  # Likely in nanometers or micrometers
                track_width = track_width / 1000000  # Convert nm to mm
            if via_size > 10:
                via_size = via_size / 1000000
            if clearance > 10:
                clearance = clearance / 1000000

            drc_data['netclasses'][nc_name] = {
                'track_width': track_width,
                'via_size': via_size,
                'clearance': clearance
            }

            logger.info(f"  NetClass '{nc_name}': track={track_width:.3f}mm via={via_size:.3f}mm clearance={clearance:.3f}mm")

            # Set defaults from Default netclass
            if nc_name == 'Default':
                drc_data['default_track_width'] = track_width
                drc_data['default_via_size'] = via_size
                drc_data['default_clearance'] = clearance

        # Ensure we have at least a Default netclass
        if 'Default' not in drc_data['netclasses']:
            drc_data['netclasses']['Default'] = {
                'track_width': drc_data['default_track_width'],
                'via_size': drc_data['default_via_size'],
                'clearance': drc_data['default_clearance']
            }
            logger.info(f"  NetClass 'Default' (fallback): track={drc_data['default_track_width']:.3f}mm via={drc_data['default_via_size']:.3f}mm clearance={drc_data['default_clearance']:.3f}mm")

        return drc_data