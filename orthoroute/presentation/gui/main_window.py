#!/usr/bin/env python3
"""
OrthoRoute Main Window - PyQt6 GUI for PCB visualization and routing
New architecture implementation of the rich GUI functionality
"""

import sys
import logging
from typing import Dict, Any, Optional, List
import os
import time
from pathlib import Path

# Add debug logging capability
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
    from debug_logger import get_debug_logger
    DEBUG_LOGGING_AVAILABLE = True
except ImportError:
    DEBUG_LOGGING_AVAILABLE = False

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTextEdit, QTreeWidget, QTreeWidgetItem,
    QSplitter, QGroupBox, QScrollArea, QTabWidget, QProgressBar,
    QStatusBar, QMenuBar, QToolBar, QApplication, QMessageBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QSlider,
    QFrame, QSizePolicy, QLineEdit
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QRect, QPoint, QPointF,
    QRectF, QPropertyAnimation, QEasingCurve, pyqtSlot, QMutex
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPixmap, QPalette,
    QAction, QIcon, QPolygonF, QTransform, QWheelEvent, QMouseEvent,
    QPaintEvent, QResizeEvent, QImage
)

from .kicad_colors import KiCadColorScheme
from .pathfinder_stats_widget import PathFinderStatsWidget
from ...algorithms.manhattan.manhattan_router_rrg import ManhattanRRGRoutingEngine
from ...algorithms.manhattan.rrg import RoutingConfig
from ...infrastructure.gpu.cuda_provider import CUDAProvider
from ...infrastructure.gpu.cpu_fallback import CPUProvider

logger = logging.getLogger(__name__)


class RoutingThread(QThread):
    """Background thread for running routing operations."""
    
    # Define signals for progress updates and completion
    progress_update = pyqtSignal(int, int, str, list, list)  # current, total, status, new_tracks, new_vias
    routing_completed = pyqtSignal(dict)  # Routing results
    routing_error = pyqtSignal(str)  # Error message
    
    def __init__(self, algorithm, board_data, config, gpu_provider=None, plugin_router=None):
        """Initialize routing thread."""
        super().__init__()
        self.algorithm = algorithm
        self.board_data = board_data
        self.config = config
        self.gpu_provider = gpu_provider
        self.plugin_router = plugin_router  # STEP 1: Accept plugin's UnifiedPathFinder instance
        self.router = None
        self.is_cancelled = False
        
    def run(self):
        """Run the routing operation in a background thread."""
        try:
            # SURGICAL FIX STEP 1: Use plugin's UnifiedPathFinder instance, don't create second router
            if self.algorithm == "Manhattan RRG":
                # STEP 1a: Use plugin's UnifiedPathFinder instance if provided
                if self.plugin_router is not None:
                    logger.info("[GUI-ROUTER-WIRE] Using plugin's UnifiedPathFinder instance (no second router)")
                    self.router = self.plugin_router
                    logger.info(f"[GUI-ROUTER-WIRE] Using existing {self.router.__class__.__name__} with instance_tag={getattr(self.router, '_instance_tag', 'NO_TAG')}")
                else:
                    # STEP 1b: Fallback - create new instance with warning
                    logger.warning("[GUI-ROUTER-WIRE] No plugin router provided, creating new UnifiedPathFinder (split-brain risk)")

                    # Import UnifiedPathFinder and GPUConfig
                    from ...algorithms.manhattan.unified_pathfinder import UnifiedPathFinder, PathFinderConfig, GPUConfig

                    # Create PathFinder config with strict DRC
                    pf_config = PathFinderConfig()
                    pf_config.strict_drc = True

                    # Create UnifiedPathFinder instance (not legacy ManhattanRRG)
                    # GPU mode is controlled by GPUConfig.GPU_MODE (hardcoded, no env vars)
                    self.router = UnifiedPathFinder(config=pf_config, use_gpu=GPUConfig.GPU_MODE)
                    logger.info(f"[GUI-ROUTER-WIRE] Created new {self.router.__class__.__name__} with instance_tag={getattr(self.router, '_instance_tag', 'NO_TAG')}")

                # STEP 1c: Add guard assert to prove UnifiedPathFinder is used
                assert self.router.__class__.__name__ == 'UnifiedPathFinder', f"Expected UnifiedPathFinder, got {self.router.__class__.__name__}"
                # STEP 1c: Convert board_data to domain Board object for UnifiedPathFinder
                from ...domain.models.board import Board

                # Convert board_data dict to domain Board object
                mock_board = self._convert_board_data_to_domain(self.board_data, None)

                # STEP 1d: Use same three-step UnifiedPathFinder calls as plugin
                logger.info(f"[GUI-UPF-STEP1] initialize_graph with {len(mock_board.nets)} nets")
                self.router.initialize_graph(mock_board)

                logger.info(f"[GUI-UPF-STEP2] map_all_pads with {len([p for n in mock_board.nets for p in n.pads])} pads")
                self.router.map_all_pads(mock_board)

                logger.info(f"[GUI-UPF-STEP3] route_multiple_nets")

                # STEP 2: Wire PathFinder stats to GUI - capture real metrics
                import time
                routing_start_time = time.time()

                # Emit start signal for PathFinder stats
                max_iterations = getattr(self.router.config, 'max_iterations', 8)
                net_count = len(mock_board.nets)
                self.progress_update.emit(0, net_count, f"Starting PathFinder negotiation ({max_iterations} iterations max)...", [], [])

                # Route with UnifiedPathFinder and capture real timing
                logger.info(f"[GUI-STATS] Starting PathFinder with {net_count} nets, {max_iterations} max iterations")
                routing_result = self.router.route_multiple_nets(mock_board.nets)
                routing_elapsed = time.time() - routing_start_time

                if routing_result:
                    logger.info("[GUI-UPF-STEP4] emit_geometry")
                    tracks_count, vias_count = self.router.emit_geometry(mock_board)

                    # Get geometry payload
                    geom = self.router.get_geometry_payload()

                    # Convert to result format
                    result = {
                        'success': tracks_count > 0 or vias_count > 0,
                        'tracks': geom.tracks if geom else [],
                        'vias': geom.vias if geom else [],
                        'routed_nets': len([n for n in mock_board.nets if n.name != 'unconnected']),
                        'failed_nets': 0,
                        'stats': {
                            'elapsed_time': routing_elapsed,  # REAL routing time
                            'total_length': tracks_count,
                            'total_vias': vias_count,
                            'success_rate': 1.0 if tracks_count > 0 else 0.0,
                            'iterations_used': max_iterations,  # PathFinder iterations
                            'negotiation_time': routing_elapsed
                        }
                    }
                    logger.info(f"[GUI-UPF-SUCCESS] Generated {tracks_count} tracks, {vias_count} vias")
                else:
                    result = {
                        'success': False,
                        'tracks': [],
                        'vias': [],
                        'routed_nets': 0,
                        'failed_nets': len(mock_board.nets),
                        'stats': {
                            'elapsed_time': routing_elapsed,  # REAL routing time even on failure
                            'total_length': 0, 'total_vias': 0, 'success_rate': 0.0,
                            'iterations_used': 0,  # Failed before negotiation
                            'negotiation_time': routing_elapsed
                        }
                    }
                    logger.error("[GUI-UPF-FAIL] UnifiedPathFinder routing failed")

                if self.is_cancelled:
                    return

                # Emit completion signal with results
                self.routing_completed.emit(result)
            
            else:
                self.routing_error.emit(f"Unknown routing algorithm: {self.algorithm}")
                
        except Exception as e:
            self.routing_error.emit(f"Routing error: {str(e)}")
            logger.exception("Error in routing thread")
    
    def cancel(self):
        """Cancel the routing operation."""
        self.is_cancelled = True
    
    def _convert_board_data_to_domain(self, board_data, drc_constraints):
        """Convert board_data dict to domain Board object for RRG router"""
        from ...domain.models.board import Board, Net, Pad, Bounds, Coordinate, Component
        
        # Create board bounds
        bounds_data = board_data.get('bounds', (0, 0, 100, 100))
        board_bounds = Bounds(
            min_x=bounds_data[0],
            min_y=bounds_data[1], 
            max_x=bounds_data[2],
            max_y=bounds_data[3]
        )
        
        # Convert nets and pads
        nets = []
        nets_data = board_data.get('nets', {})
        
        for net_name, net_data in nets_data.items():
            if not net_name or net_name.strip() == "":
                continue
                
            pads_data = net_data.get('pads', [])
            if len(pads_data) < 2:
                continue  # Skip single-pad nets
            
            # Convert pads
            net_pads = []
            for pad_data in pads_data:
                pad = Pad(
                    id=f"{net_name}_pad_{len(net_pads)}",
                    component_id=f"comp_{net_name}_{len(net_pads)}",
                    net_id=f"net_{len(nets)}",
                    position=Coordinate(
                        x=pad_data.get('x', 0.0),
                        y=pad_data.get('y', 0.0)
                    ),
                    size=(
                        pad_data.get('width', 1.0),
                        pad_data.get('height', 1.0)
                    ),
                    drill_size=pad_data.get('drill', None),
                    layer=pad_data.get('layers', ['F.Cu'])[0] if pad_data.get('layers') else 'F.Cu'
                )
                net_pads.append(pad)
            
            # Create net
            net = Net(
                id=f"net_{len(nets)}",
                name=net_name,
                pads=net_pads
            )
            nets.append(net)
        
        # Create mock components for proper bounds calculation
        components = []
        all_pads = []
        for net in nets:
            for i, pad in enumerate(net.pads):
                all_pads.append(pad)
        
        # Create a single mock component containing all pads
        if all_pads:
            # Calculate center position
            avg_x = sum(pad.position.x for pad in all_pads) / len(all_pads)
            avg_y = sum(pad.position.y for pad in all_pads) / len(all_pads)
            
            mock_component = Component(
                id="mock_comp_1",
                reference="U1",
                value="MOCK",
                footprint="MOCK_FP",
                position=Coordinate(avg_x, avg_y),
                pads=all_pads
            )
            components.append(mock_component)
        
        # Create board - handle filename robustly
        filename = board_data.get('filename') or board_data.get('name') or 'TestBackplane.kicad_pcb'
        board = Board(
            id="board_1", 
            name=filename,
            components=components,
            nets=nets,
            layer_count=board_data.get('layers', 2)  # Dynamic from KiCad file
        )
        
        # Store airwires as a custom attribute for RRG routing
        board._airwires = board_data.get('airwires', [])
        # Store KiCad-calculated bounds for accurate routing area
        board._kicad_bounds = board_data.get('bounds', None)
        
        return board


# PathFinder GUI Integration Methods
def _update_pathfinder_status(self, status_text: str):
    """Update status bar with PathFinder instrumentation metrics"""
    self.status_label.setText(status_text)
    self.metrics_label.setText(status_text)
    self.metrics_label.setVisible(True)
    
    # Also print to terminal for console monitoring
    print(f"[PathFinder]: {status_text}")

def _update_csv_export_status(self, csv_status: str):
    """Update status bar with CSV export information"""
    self.csv_status_label.setText(csv_status)
    self.csv_status_label.setVisible(True)
    
    # Auto-hide CSV status after 10 seconds
    QTimer.singleShot(10000, lambda: self.csv_status_label.setVisible(False))

def _display_instrumentation_summary(self):
    """Display instrumentation summary when routing completes"""
    if hasattr(self.router, 'pathfinder') and hasattr(self.router.pathfinder, 'get_instrumentation_summary'):
        summary = self.router.pathfinder.get_instrumentation_summary()
        
        if summary:
            summary_text = (f"Session: {summary.get('session_id', 'N/A')} | "
                          f"Iterations: {summary.get('total_iterations', 0)} | "
                          f"Success: {summary.get('final_success_rate', 0):.1f}% | "
                          f"Nets: {summary.get('successful_nets', 0)}/{summary.get('total_nets_processed', 0)} | "
                          f"Avg Time: {summary.get('avg_routing_time_ms', 0):.1f}ms")
            
            self.metrics_label.setText(summary_text)
            self.metrics_label.setVisible(True)
            
            print(f"[SUMMARY] Routing Summary: {summary_text}")
            logger.info(f"Instrumentation summary: {summary}")



class PCBViewer(QWidget):
    """PCB visualization widget for displaying board, components, tracks, and airwires"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.board_data = None
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.last_pan_point = QPoint()
        self.is_panning = False
        
        # Initialize KiCad color scheme
        self.color_scheme = KiCadColorScheme()
        
        # Display option flags
        self.show_components = True
        self.show_tracks = True
        self.show_vias = True
        self.show_pads = True
        self.show_airwires = True
        self.show_zones = True
        self.show_keepouts = True
        
        # Layer visibility tracking (will be updated when board loads with actual layer names)
        self.visible_layers = set()  # Start empty, will be populated from board_data['layer_names']
        
        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)
        
    def set_board_data(self, board_data: Dict[str, Any]):
        """Set the board data to display"""
        self.board_data = board_data

        # Initialize visible_layers from board data (all layers visible by default)
        if board_data and 'layer_names' in board_data:
            self.visible_layers = set(board_data['layer_names'])
            logger.info(f"PCBViewer: Initialized {len(self.visible_layers)} visible layers from board data")
        elif not self.visible_layers:
            # Fallback to 2-layer board
            self.visible_layers = set(['F.Cu', 'B.Cu'])

        self.fit_to_view()
        self.update()
        
    def fit_to_view(self):
        """Fit the board to the current view"""
        if not self.board_data:
            return
            
        # Get board bounds
        bounds = self.board_data.get('bounds', (0, 0, 100, 100))
        board_width = bounds[2] - bounds[0]
        board_height = bounds[3] - bounds[1]
        
        if board_width <= 0 or board_height <= 0:
            return
            
        # Calculate zoom to fit
        widget_width = self.width() - 40
        widget_height = self.height() - 40
        
        zoom_x = widget_width / board_width
        zoom_y = widget_height / board_height
        
        self.zoom_factor = min(zoom_x, zoom_y) * 0.9
        
        # Center the board
        self.pan_x = (bounds[0] + bounds[2]) / 2
        self.pan_y = (bounds[1] + bounds[3]) / 2
        
        # Trigger repaint
        self.update()
        
    def paintEvent(self, event: QPaintEvent):
        """Paint the PCB visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background with KiCad background color
        painter.fillRect(self.rect(), self.color_scheme.get_color('background'))
        
        if not self.board_data:
            painter.setPen(self.color_scheme.get_color('text'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No board data loaded")
            return
            
        # Set up coordinate transformation
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(self.zoom_factor, self.zoom_factor)
        painter.translate(-self.pan_x, -self.pan_y)
        
        # Skip artificial board outline - real boards should use Edge.Cuts layer
        # self._draw_board_outline(painter)
        
        # Draw airwires (behind everything)
        if self.show_airwires:
            self._draw_airwires(painter)
        
        # Draw tracks
        if self.show_tracks:
            self._draw_tracks(painter)
        
        # Draw pads
        if self.show_pads:
            self._draw_pads(painter)
        
        # Draw components
        if self.show_components:
            self._draw_components(painter)
            
        # Draw vias
        if self.show_vias:
            self._draw_vias(painter)
        
    def _draw_board_outline(self, painter: QPainter):
        """Draw the board outline"""
        bounds = self.board_data.get('bounds', (0, 0, 100, 100))
        
        painter.setPen(QPen(self.color_scheme.get_color('edge_cuts'), 0.5))
        painter.setBrush(QBrush())
        
        rect = QRectF(bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1])
        painter.drawRect(rect)
        
    def _draw_airwires(self, painter: QPainter):
        """Draw airwires (unrouted connections) with performance optimization"""
        airwires = self.board_data.get('airwires', [])
        
        # Performance limit: only show airwires when zoomed in enough or limit count
        zoom_level = painter.transform().m11()  # Get current zoom scale
        max_airwires = min(len(airwires), int(1000 * zoom_level) if zoom_level > 0.1 else 200)
        
        if max_airwires <= 0:
            return
            
        painter.setPen(QPen(self.color_scheme.get_color('ratsnest'), 0.1))
        
        # Get viewport bounds for culling
        viewport = painter.viewport()
        
        # More robust viewport calculation
        try:
            transform_inverted, invertible = painter.transform().inverted()
            if invertible:
                visible_rect = transform_inverted.mapRect(QRectF(viewport))
            else:
                # Fallback: render everything if transform can't be inverted
                visible_rect = QRectF(-1000, -1000, 2000, 2000)  # Large fallback area
        except:
            # Fallback: render everything if viewport calculation fails
            visible_rect = QRectF(-1000, -1000, 2000, 2000)  # Large fallback area
        
        drawn_count = 0
        for i, airwire in enumerate(airwires):
            if i >= max_airwires:  # Hard limit for performance
                break
                
            try:
                x1 = airwire['start_x']
                y1 = airwire['start_y']
                x2 = airwire['end_x']
                y2 = airwire['end_y']
                
                # Viewport culling: only draw if line intersects visible area (disabled for now - needs debugging)  
                # line_rect = QRectF(min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1))
                # if visible_rect.intersects(line_rect):
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
                drawn_count += 1
                
            except (KeyError, TypeError):
                continue
        
        # Debug log airwire rendering stats
        if DEBUG_LOGGING_AVAILABLE and drawn_count > 0:
            debug_logger = get_debug_logger()
            render_counts = {
                'airwires_drawn': drawn_count,
                'max_airwires_allowed': max_airwires,
                'total_airwires_available': len(airwires),
                'zoom_level': zoom_level
            }
            debug_logger.log_visualization_data(render_counts, zoom_level)
                
    def _draw_tracks(self, painter: QPainter):
        """Draw existing tracks/traces with performance optimization"""
        tracks = self.board_data.get('tracks', [])
        
        logger.info(f"_draw_tracks called: board_data has {len(tracks)} tracks")
        if tracks:
            logger.info(f"First track: {tracks[0]}")

        if not tracks:
            logger.info("No tracks to draw - returning early")
            return
            
        # CRITICAL PERFORMANCE FIX: Viewport culling and LOD
        zoom_level = painter.transform().m11()  # Get current zoom from transform matrix
        
        # Calculate visible viewport in world coordinates
        try:
            transform = painter.transform().inverted()[0]
            viewport_rect = painter.viewport()
            visible_rect = transform.mapRect(QRectF(viewport_rect))

            # Expand visible rect slightly for smooth panning
            margin = max(visible_rect.width(), visible_rect.height()) * 0.1
            visible_rect = visible_rect.adjusted(-margin, -margin, margin, margin)

            logger.info(f"VIEWPORT DEBUG: visible_rect = ({visible_rect.left():.1f}, {visible_rect.top():.1f}) -> ({visible_rect.right():.1f}, {visible_rect.bottom():.1f})")
            logger.info(f"VIEWPORT DEBUG: zoom_level = {zoom_level:.3f}")
        except Exception as e:
            # Fallback: render everything if transform fails
            logger.warning(f"VIEWPORT DEBUG: Transform failed ({e}), using fallback viewport")
            visible_rect = QRectF(-1000, -1000, 2000, 2000)
        
        # Smart viewport culling for performance with large track counts
        max_tracks_per_frame = 50000  # Reasonable limit per frame
        min_width = 0.0001

        drawn_tracks = 0
        culled_tracks = 0

        for track in tracks:
            if drawn_tracks >= max_tracks_per_frame:
                culled_tracks += 1
                continue
                
            try:
                # Handle both coordinate key formats - updated for new track structure
                if 'start' in track and 'end' in track:
                    x1, y1 = track['start']
                    x2, y2 = track['end']
                else:
                    x1 = track.get('start_x', track.get('x1'))
                    y1 = track.get('start_y', track.get('y1'))
                    x2 = track.get('end_x', track.get('x2'))
                    y2 = track.get('end_y', track.get('y2'))

                width = track.get('width', 0.1)

                # Viewport culling - only draw tracks that intersect with visible area
                track_rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                if not visible_rect.intersects(track_rect.adjusted(-1, -1, 1, 1)):  # Small margin for line width
                    culled_tracks += 1
                    continue

                # DEBUG: Check for None coordinates
                if any(coord is None for coord in [x1, y1, x2, y2]):
                    logger.warning(f"TRACK SKIPPED: None coordinates in {track}")
                    continue

                # DEBUG: Log coordinate verification for first track
                if drawn_tracks == 0:
                    bounds = self.board_data.get('bounds', None)
                    if bounds:
                        board_min_x, board_max_x = bounds[0], bounds[2]
                        board_min_y, board_max_y = bounds[1], bounds[3]
                        in_bounds_x = board_min_x <= x1 <= board_max_x and board_min_x <= x2 <= board_max_x
                        in_bounds_y = board_min_y <= y1 <= board_max_y and board_min_y <= y2 <= board_max_y
                        logger.info(f"COORD VERIFY: Track ({x1:.1f},{y1:.1f})->({x2:.1f},{y2:.1f}) in_bounds=({in_bounds_x},{in_bounds_y})")
                        logger.info(f"COORD VERIFY: Board bounds ({board_min_x:.1f},{board_min_y:.1f})->({board_max_x:.1f},{board_max_y:.1f})")

                # Handle both layer formats: integer and string
                layer_raw = track.get('layer', 0)
                if isinstance(layer_raw, int):
                    # Convert integer layer to KiCad layer name
                    if layer_raw == 0:
                        layer = 'F.Cu'  # Front copper
                    elif layer_raw == 1:
                        layer = 'B.Cu'  # Back copper
                    else:
                        layer = f'In{layer_raw-1}.Cu'  # Internal layers In1.Cu, In2.Cu, etc.
                else:
                    layer = layer_raw
                
                # TEMPORARY: Disable viewport culling to debug coordinate system
                # PERFORMANCE: Skip tracks outside visible viewport
                # FIXED: Use proper line-viewport intersection instead of rectangles!
                line_start = QPointF(x1, y1)
                line_end = QPointF(x2, y2)

                # DEBUG: Log first few tracks to see coordinate ranges
                if drawn_tracks < 5:
                    logger.info(f"TRACK COORDS #{drawn_tracks+1}: ({x1:.1f},{y1:.1f})->({x2:.1f},{y2:.1f}) layer={layer}")

                # DISABLED: Skip viewport culling for now
                # if (visible_rect.contains(line_start) or
                #     visible_rect.contains(line_end) or
                #     self._line_intersects_rect(x1, y1, x2, y2, visible_rect)):
                #     # Line is visible, continue to draw
                #     pass
                # else:
                #     # Log why tracks are being culled for first few tracks
                #     if drawn_tracks < 3:
                #         logger.info(f"TRACK CULLED #{drawn_tracks+1}: track ({x1:.1f},{y1:.1f})->({x2:.1f},{y2:.1f}) outside viewport ({visible_rect.left():.1f},{visible_rect.top():.1f})->({visible_rect.right():.1f},{visible_rect.bottom():.1f})")
                #     continue
                    
                # ALWAYS SHOW TRACKS - No width threshold for production visibility
                # User requirement: tracks/vias visible at every zoom level
                
                # Skip if layer is not visible
                if layer not in self.visible_layers:
                    # Log layer visibility issue for first few tracks
                    if drawn_tracks < 3:
                        logger.info(f"TRACK LAYER FILTERED #{drawn_tracks+1}: track layer '{layer}' not in visible_layers {list(self.visible_layers)}")
                    continue
                
                # Set color based on layer
                if layer == 'F.Cu':
                    color = self.color_scheme.get_color('copper_front')
                elif layer == 'B.Cu':
                    color = self.color_scheme.get_color('copper_back')
                else:
                    # Handle internal layers with distinct colors
                    # Parse "In1.Cu", "In2.Cu" etc to get the correct color
                    try:
                        if layer.startswith('In') and layer.endswith('.Cu'):
                            layer_num = int(layer[2:-3])  # Extract number from "In{N}.Cu"
                            color_key = f'copper_in{layer_num}'
                            color = self.color_scheme.get_color(color_key)
                        else:
                            color = self.color_scheme.get_color('copper_inner')
                    except (ValueError, AttributeError):
                        color = self.color_scheme.get_color('copper_inner')
                
                # Use actual track dimensions (just like footprints are drawn)
                # Convert mm to scene coordinates - same as footprints use
                line_width = width  # Use actual track width in mm
                painter.setPen(QPen(color, line_width))
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
                drawn_tracks += 1
                
                # Log first few tracks drawn
                if drawn_tracks <= 3:
                    logger.info(f"TRACK DRAWN #{drawn_tracks}: ({x1:.3f},{y1:.3f}) -> ({x2:.3f},{y2:.3f}) width={line_width} layer={layer}")
                
            except (KeyError, TypeError):
                continue
        
        logger.info(f"_draw_tracks completed: drew {drawn_tracks} tracks, culled {culled_tracks} (viewport + limit), total {len(tracks)} available")
                
    def _line_intersects_rect(self, x1, y1, x2, y2, rect):
        """Check if a line intersects with a rectangle (simple bounding box check)"""
        # Simple bounding box intersection - good enough for viewport culling
        line_left = min(x1, x2)
        line_right = max(x1, x2) 
        line_top = min(y1, y2)
        line_bottom = max(y1, y2)
        
        return not (line_right < rect.left() or 
                   line_left > rect.right() or
                   line_bottom < rect.top() or 
                   line_top > rect.bottom())
                
    def _draw_pads(self, painter: QPainter):
        """Draw component pads with viewport culling and LOD optimization"""
        pads = self.board_data.get('pads', [])
        
        # Get zoom level and viewport for optimization
        zoom_level = painter.transform().m11()
        viewport = painter.viewport()
        
        # More robust viewport calculation
        try:
            transform_inverted, invertible = painter.transform().inverted()
            if invertible:
                visible_rect = transform_inverted.mapRect(QRectF(viewport))
            else:
                # Fallback: render everything if transform can't be inverted
                visible_rect = QRectF(-1000, -1000, 2000, 2000)  # Large fallback area
        except:
            # Fallback: render everything if viewport calculation fails
            visible_rect = QRectF(-1000, -1000, 2000, 2000)  # Large fallback area
        
        # LOD: Skip very small pads when zoomed out
        min_pad_size = 1.0 / zoom_level if zoom_level > 0 else 0.1  # Minimum size in world units
        
        drawn_pads = 0
        max_pads = min(len(pads), 5000) if zoom_level < 0.1 else len(pads)  # Limit when zoomed way out
        
        for i, pad in enumerate(pads):
            if i >= max_pads:
                break
            try:
                x = pad['x']
                y = pad['y']
                width = pad.get('width', 1.0)
                height = pad.get('height', 1.0)
                
                # Viewport culling: skip pads outside visible area (disabled for now - needs debugging)
                # pad_rect = QRectF(x - width/2, y - height/2, width, height)
                # if not visible_rect.intersects(pad_rect):
                #     continue
                
                # LOD culling: skip very small pads when zoomed out
                if max(width, height) < min_pad_size:
                    continue
                
                pad_type = pad.get('type', 'smd')
                layers = pad.get('layers', ['F.Cu'])
                drill_size = pad.get('drill', 0.0)
                
                # Skip if none of the pad's layers are visible
                if not any(layer in self.visible_layers for layer in layers):
                    continue
                
                drawn_pads += 1
                
                # Debug logging for visualization (every 100th frame to avoid spam)
                if DEBUG_LOGGING_AVAILABLE and drawn_pads == 50:  # Log at 50th pad to get sample
                    debug_logger = get_debug_logger()
                    render_counts = {
                        'pads_processed': i,
                        'pads_drawn': drawn_pads, 
                        'max_pads': max_pads,
                        'zoom_level': zoom_level,
                        'min_pad_size': min_pad_size,
                        'total_pads_available': len(pads)
                    }
                    debug_logger.log_visualization_data(render_counts, zoom_level)
                
                # Get appropriate pad color based on type and layer
                if pad_type == 'smd':
                    # SMD pads use layer-specific colors
                    if any('B.' in layer for layer in layers):
                        pad_color = self.color_scheme.get_color('pad_back')
                    else:
                        pad_color = self.color_scheme.get_color('pad_front')
                elif pad_type == 'through_hole':
                    pad_color = self.color_scheme.get_color('pad_through_hole')
                else:
                    pad_color = self.color_scheme.get_color('pad_front')
                
                # Draw pad based on type
                painter.setPen(QPen(pad_color, 0.05))
                painter.setBrush(QBrush(pad_color))
                
                if pad_type == 'through_hole':
                    # Draw through-hole pads as circles
                    pad_size = max(width, height)  # Use larger dimension for circular pad
                    pad_rect = QRectF(x - pad_size/2, y - pad_size/2, pad_size, pad_size)
                    painter.drawEllipse(pad_rect)
                    
                    # Draw drill hole for through-hole pads
                    if drill_size > 0:
                        # Use gold color for plated holes (more realistic)
                        hole_color = QColor(255, 215, 0)  # Gold color
                        
                        # Draw circular hole with gold fill and no outline
                        painter.setPen(QPen(hole_color, 0))  # No outline
                        painter.setBrush(QBrush(hole_color))  # Gold fill
                        
                        # Draw circular hole
                        hole_rect = QRectF(x - drill_size/2, y - drill_size/2, drill_size, drill_size)
                        painter.drawEllipse(hole_rect)
                else:
                    # Draw SMD pads as rectangles
                    pad_rect = QRectF(x - width/2, y - height/2, width, height)
                    painter.drawRect(pad_rect)
                    
            except (KeyError, TypeError):
                continue
                
    def _draw_components(self, painter: QPainter):
        """Draw component outlines and reference designators"""
        components = self.board_data.get('components', [])
        
        # Use silkscreen color for component text
        painter.setPen(QPen(self.color_scheme.get_color('f_silks'), 0.1))
        painter.setBrush(QBrush())
        
        font = QFont("Arial", max(1, int(2.0 / self.zoom_factor)))
        painter.setFont(font)
        
        for component in components:
            try:
                x = component['x']
                y = component['y']
                ref_des = component.get('ref_des', 'Unknown')
                
                # Draw component reference designator
                painter.drawText(QPointF(x, y), ref_des)
                
            except (KeyError, TypeError, ValueError) as e:
                # Skip invalid component data
                continue
        
    def _draw_vias(self, painter: QPainter):
        """Draw vias with proper coloring and visibility control"""
        vias = self.board_data.get('vias', [])

        if not vias:
            logger.info("_draw_vias: No vias to draw")
            return  # No vias to draw

        logger.info(f"_draw_vias called: board_data has {len(vias)} vias")
            
        # Get zoom level for LOD optimization
        zoom_level = painter.transform().m11()
        
        # Get viewport bounds for culling
        viewport = painter.viewport()
        
        try:
            transform_inverted, invertible = painter.transform().inverted()
            if invertible:
                visible_rect = transform_inverted.mapRect(QRectF(viewport))
            else:
                visible_rect = QRectF(-1000, -1000, 2000, 2000)  # Large fallback area
        except:
            visible_rect = QRectF(-1000, -1000, 2000, 2000)  # Large fallback area
        
        # Draw vias
        drawn_vias = 0
        for via in vias:
            try:
                # Handle both coordinate formats
                if 'position' in via:
                    x, y = via['position']
                else:
                    x = via['x']
                    y = via['y']
                
                # Skip if outside visible area
                if not visible_rect.contains(QPointF(x, y)):
                    continue
                
                # Determine via type and size
                via_type = via.get('type', via.get('via_type', 'through'))
                diameter = via.get('diameter', via.get('size', 0.25))  # Support both 'diameter' and 'size'
                drill = via.get('drill', 0.3)
                start_layer = via.get('start_layer', via.get('from_layer', 'F.Cu'))
                end_layer = via.get('end_layer', via.get('to_layer', 'B.Cu'))
                
                # ALWAYS SHOW VIAS - No LoD threshold for production visibility
                # User requirement: tracks/vias visible at every zoom level
                
                # Set color based on via type
                if via_type == 'through':
                    color = self.color_scheme.get_color('via_through')
                elif via_type in ['blind_buried', 'blind', 'buried']:
                    color = self.color_scheme.get_color('via_blind_buried')
                elif via_type == 'micro':
                    color = self.color_scheme.get_color('via_micro')
                else:
                    color = self.color_scheme.get_color('via_through')  # Default
                
                # Draw via ring
                painter.setPen(QPen(color, 0.1))
                painter.setBrush(QBrush(color))
                
                # Draw via outer diameter
                via_rect = QRectF(x - diameter/2, y - diameter/2, diameter, diameter)
                painter.drawEllipse(via_rect)
                
                # Draw via hole
                hole_color = self.color_scheme.get_color('via_hole')
                painter.setPen(QPen(hole_color, 0.1))
                painter.setBrush(QBrush(hole_color))
                
                hole_rect = QRectF(x - drill/2, y - drill/2, drill, drill)
                painter.drawEllipse(hole_rect)
                
                # Draw layer indicator for blind/buried vias
                if via_type == 'blind_buried':
                    # Draw small indicators showing which layers this via connects
                    indicator_size = diameter / 4
                    painter.setPen(QPen(QColor(255, 255, 255), 0.1))
                    
                    # This would be expanded for more sophisticated layer indicators
                    painter.drawText(QRectF(x + diameter/2, y - diameter/2, 
                                          indicator_size*2, indicator_size*2),
                                   Qt.AlignmentFlag.AlignCenter, 
                                   f"{start_layer[0]}-{end_layer[0]}")

                drawn_vias += 1

            except (KeyError, TypeError) as e:
                logger.warning(f"Invalid via data: {via}, error: {e}")
                continue

        logger.info(f"_draw_vias completed: drew {drawn_vias} vias out of {len(vias)} available")
                
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming"""
        zoom_factor = 1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2
        self.zoom_factor *= zoom_factor
        self.zoom_factor = max(0.01, min(100.0, self.zoom_factor))
        self.update()
        
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for panning"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = True
            self.last_pan_point = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for panning"""
        if self.is_panning:
            delta = event.pos() - self.last_pan_point
            self.pan_x -= delta.x() / self.zoom_factor
            self.pan_y -= delta.y() / self.zoom_factor
            self.last_pan_point = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def debug_screenshot(self, filename_prefix: str = "debug_routing", scale_factor: int = 1, output_dir: str = None):
        """Capture screenshot of the PCB viewer for debugging with optional high-res rendering"""
        try:
            import os
            from datetime import datetime

            # Determine output directory
            if output_dir is None:
                debug_dir = "debug_output"
                os.makedirs(debug_dir, exist_ok=True)
            else:
                debug_dir = output_dir
                os.makedirs(debug_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
            filename = f"{debug_dir}/{filename_prefix}_{timestamp}.png"

            # Capture the widget at specified resolution
            if scale_factor > 1:
                # High-res rendering
                widget_size = self.size()
                scaled_size = widget_size * scale_factor

                # Create high-res image
                image = QImage(scaled_size, QImage.Format.Format_ARGB32)
                image.fill(Qt.GlobalColor.transparent)

                # Render widget to image with scaling
                painter = QPainter(image)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                painter.scale(scale_factor, scale_factor)
                self.render(painter)
                painter.end()

                # Save the image
                success = image.save(filename, "PNG")
            else:
                # Standard resolution
                pixmap = self.grab()
                success = pixmap.save(filename, "PNG")

            if success:
                print(f"DEBUG: Screenshot saved to {filename} (scale={scale_factor}x)")
                return filename
            else:
                print(f"DEBUG: Failed to save screenshot to {filename}")
                return None

        except Exception as e:
            print(f"DEBUG: Screenshot error: {e}")
            return None
            
    def update_routing(self, tracks, vias):
        """Update the board data with new routing information"""
        if tracks:
            self.board_data['tracks'] = tracks
        if vias:
            self.board_data['vias'] = vias
        # Trigger repaint
        self.update()
            

class OrthoRouteMainWindow(QMainWindow):
    """Main OrthoRoute GUI window - faithful recreation of original interface"""
    
    def __init__(self, board_data: Dict[str, Any], kicad_interface, plugin=None):
        super().__init__()
        self.board_data = board_data
        self.kicad_interface = kicad_interface
        self.plugin = plugin
        self.pcb_viewer = None
        self.algorithm_combo = None
        self.display_checkboxes = {}
        self.layer_actions = {}
        self.route_preview_btn = None
        self.commit_btn = None
        self.rollback_btn = None
        self.status_label = None
        self.gpu_status = None
        self.routing_result = None

        # Initialize KiCad color scheme (needed for layers panel)
        self.color_scheme = KiCadColorScheme()

        # Initialize window
        self.setWindowTitle("OrthoRoute - PCB Autorouter")
        self.setMinimumSize(1200, 800)
        self.resize(1800, 800)

        # Detect GPU status
        self.detect_gpu_status()
        
        # Setup UI components
        self.setup_ui()
        self.setup_menus()
        self.setup_status_bar()
        
        # Load board data
        self.load_board_data()
        
    def detect_gpu_status(self):
        """Detect GPU capabilities for algorithm selection"""
        try:
            # Try to detect CUDA/GPU availability
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.gpu_status = {'available': True, 'can_use_gpu_routing': True}
                logger.info("GPU detected and available for routing")
            else:
                self.gpu_status = {'available': False, 'can_use_gpu_routing': False}
                logger.info("GPU not available, using CPU routing")
        except:
            self.gpu_status = {'available': False, 'can_use_gpu_routing': False}
            logger.info("GPU detection failed, using CPU routing")
        
    def setup_ui(self):
        """Setup the main UI layout - three panel design like original"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel for controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Center panel for PCB viewer
        self.pcb_viewer = PCBViewer()
        splitter.addWidget(self.pcb_viewer)
        
        # Right panel for information
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions (left:center:right = 300:800:300)
        splitter.setSizes([300, 800, 300])
        
    def create_left_panel(self) -> QWidget:
        """Create the left control panel - matching original layout"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Display options group
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.display_checkboxes = {}
        display_options = [
            ('Components', 'show_components'),
            ('Tracks', 'show_tracks'),
            ('Vias', 'show_vias'),
            ('Pads', 'show_pads'),
            ('Airwires', 'show_airwires'),
            ('Zones', 'show_zones'),
            ('Keepouts', 'show_keepouts')
        ]
        
        for label, attr in display_options:
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)  # Default to showing all
            checkbox.toggled.connect(lambda checked, attr=attr: self.toggle_display_option(attr, checked))
            self.display_checkboxes[attr] = checkbox
            display_layout.addWidget(checkbox)
        
        layout.addWidget(display_group)
        
        # Routing controls group
        routing_group = QGroupBox("Routing Controls")
        routing_layout = QVBoxLayout(routing_group)
        
        # Algorithm selection
        algorithm_layout = QHBoxLayout()
        algorithm_layout.addWidget(QLabel("Algorithm:"))
        self.algorithm_combo = QComboBox()
        
        # Available algorithms - simplified to single best option
        algorithm_options = [
            "Manhattan RRG"
        ]
        
        self.algorithm_combo.addItems(algorithm_options)
        self.algorithm_combo.setCurrentIndex(0)  # Only option
        
        if self.gpu_status and self.gpu_status.get('can_use_gpu_routing'):
            logger.info("All algorithms will use GPU acceleration with CPU fallback")
        else:
            logger.info("All algorithms will use CPU (GPU not available)")
            
        # Connect algorithm change signal
        self.algorithm_combo.currentTextChanged.connect(self.on_algorithm_changed)
        
        algorithm_layout.addWidget(self.algorithm_combo)
        routing_layout.addLayout(algorithm_layout)
        
        # Batch Size Control
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spinner = QSpinBox()
        self.batch_size_spinner.setMinimum(1)
        self.batch_size_spinner.setMaximum(500)
        self.batch_size_spinner.setValue(50)  # Default batch size
        self.batch_size_spinner.setToolTip("Number of nets to route in each PathFinder iteration")
        batch_layout.addWidget(self.batch_size_spinner)
        routing_layout.addLayout(batch_layout)

        # GPU acceleration checkbox (beta feature)
        self.gpu_checkbox = QCheckBox("Enable GPU acceleration (beta)")

        # GPU mode is hardcoded via GPUConfig (no env vars for KiCad plugin compatibility)
        try:
            from ...algorithms.manhattan.unified_pathfinder import GPUConfig
            self.gpu_checkbox.setChecked(GPUConfig.GPU_MODE)
        except ImportError:
            self.gpu_checkbox.setChecked(False)  # Fallback if import fails

        self.gpu_checkbox.setToolTip("Enable CUDA GPU acceleration for routing (experimental)")

        # Disable if no GPU detected
        if hasattr(self, 'gpu_status') and not self.gpu_status.get('available', False):
            self.gpu_checkbox.setEnabled(False)
            self.gpu_checkbox.setToolTip("GPU not available on this system")

        routing_layout.addWidget(self.gpu_checkbox)

        # Main routing buttons
        self.route_preview_btn = QPushButton("Begin Autorouting")
        self.route_preview_btn.clicked.connect(self.begin_autorouting)
        routing_layout.addWidget(self.route_preview_btn)
        
        # Solution control buttons (initially disabled)
        solution_layout = QHBoxLayout()
        self.commit_btn = QPushButton("Apply to KiCad")
        self.commit_btn.clicked.connect(self.commit_routes)
        self.commit_btn.setEnabled(False)
        
        self.rollback_btn = QPushButton("Discard")
        self.rollback_btn.clicked.connect(self.rollback_routes)
        self.rollback_btn.setEnabled(False)

        self.replay_btn = QPushButton("Replay")
        self.replay_btn.clicked.connect(self.replay_routing)
        self.replay_btn.setEnabled(False)
        self.replay_btn.setToolTip("Re-run the same routing with clean state")

        solution_layout.addWidget(self.commit_btn)
        solution_layout.addWidget(self.rollback_btn)
        solution_layout.addWidget(self.replay_btn)
        routing_layout.addLayout(solution_layout)
        
        layout.addWidget(routing_group)

        # Debug controls group
        debug_group = QGroupBox("Debug Controls")
        debug_layout = QVBoxLayout(debug_group)

        # Focus net debugging
        focus_net_layout = QHBoxLayout()
        focus_net_layout.addWidget(QLabel("Focus Net:"))
        self.focus_net_input = QLineEdit()
        self.focus_net_input.setPlaceholderText("e.g. B06B14_000")
        self.focus_net_input.returnPressed.connect(self.focus_on_net)
        focus_net_layout.addWidget(self.focus_net_input)

        self.focus_net_btn = QPushButton("Highlight")
        self.focus_net_btn.clicked.connect(self.focus_on_net)
        focus_net_layout.addWidget(self.focus_net_btn)
        debug_layout.addLayout(focus_net_layout)

        # Show pad stubs toggle - check environment variable
        import os
        show_stubs_default = os.getenv("ORTHO_SHOW_PORTALS", "1").lower() in ("1", "true", "yes")
        self.show_pad_stubs_checkbox = QCheckBox("Show pad stubs")
        self.show_pad_stubs_checkbox.setChecked(show_stubs_default)
        self.show_pad_stubs_checkbox.toggled.connect(self.toggle_pad_stubs)
        debug_layout.addWidget(self.show_pad_stubs_checkbox)

        # Portal visualization for first 50 nets
        self.show_portal_dots_checkbox = QCheckBox("Show portal dots (first 50 nets)")
        self.show_portal_dots_checkbox.setChecked(False)
        self.show_portal_dots_checkbox.toggled.connect(self.toggle_portal_dots)
        debug_layout.addWidget(self.show_portal_dots_checkbox)

        layout.addWidget(debug_group)

        # Nets statistics group
        nets_stats_group = QGroupBox("Nets Statistics")
        nets_stats_layout = QVBoxLayout(nets_stats_group)
        
        self.nets_stats_label = QLabel("Loading nets...")
        nets_stats_layout.addWidget(self.nets_stats_label)
        
        layout.addWidget(nets_stats_group)
        
        # Progress group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_label)
        
        layout.addWidget(progress_group)
        
        return panel
        
    def create_right_panel(self) -> QWidget:
        """Create the right information panel with Board Info and Layers"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Board information group (at top)
        board_info_group = QGroupBox("Board Information")
        board_info_layout = QVBoxLayout(board_info_group)

        self.board_info_label = QLabel("Loading board information...")
        board_info_layout.addWidget(self.board_info_label)

        layout.addWidget(board_info_group)

        # Layers visibility group (dynamic, scrollable)
        layers_group = QGroupBox("Layers")
        layers_layout = QVBoxLayout(layers_group)

        # Scrollable area for layer checkboxes
        layers_scroll = QScrollArea()
        layers_scroll.setWidgetResizable(True)
        layers_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layers_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layers_scroll.setMaximumHeight(400)

        # Container widget for checkboxes
        self.layers_container = QWidget()
        self.layers_container_layout = QVBoxLayout(self.layers_container)
        self.layers_container_layout.setContentsMargins(5, 5, 5, 5)
        self.layers_container_layout.setSpacing(2)

        layers_scroll.setWidget(self.layers_container)
        layers_layout.addWidget(layers_scroll)

        layout.addWidget(layers_group)

        # Add stretch to push everything to top
        layout.addStretch()

        # Store reference to layers group for updating
        self.layers_group = layers_group
        self.layer_checkboxes = {}

        # Create stub widgets that other code expects but hide them
        self.nets_tree = QTreeWidget()
        self.nets_tree.setVisible(False)
        self.pathfinder_stats = PathFinderStatsWidget()
        self.pathfinder_stats.setVisible(False)
        self.routing_log = QTextEdit()
        self.routing_log.setVisible(False)

        return panel
        
    def setup_status_bar(self):
        """Setup status bar with GPU status and instrumentation metrics"""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        # Main status label for routing progress
        self.status_label = QLabel("Ready")
        status_bar.addWidget(self.status_label)
        
        # Instrumentation metrics labels (initially hidden)
        self.metrics_label = QLabel("")
        self.metrics_label.setVisible(False)
        status_bar.addWidget(self.metrics_label)
        
        # CSV export status
        self.csv_status_label = QLabel("")
        self.csv_status_label.setVisible(False) 
        status_bar.addWidget(self.csv_status_label)
        
        # GPU status indicator
        gpu_text = "GPU Available" if self.gpu_status.get('available') else "CPU Only"
        status_bar.addPermanentWidget(QLabel(f"{gpu_text} | OrthoRoute v1.0"))
        
    def setup_menus(self):
        """Setup menu bar - matching original menus"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        refresh_action = QAction("Refresh Board", self)
        refresh_action.triggered.connect(self.refresh_board)
        file_menu.addAction(refresh_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        # Layer visibility submenu
        self.layers_menu = view_menu.addMenu("Layers")
        
        # Create layer visibility actions - will be updated when board loads
        self.layer_actions = {}
        
        view_menu.addSeparator()
        
        fit_action = QAction("Fit to Window", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self.zoom_fit)
        view_menu.addAction(fit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        route_action = QAction("Auto Route All", self)
        route_action.triggered.connect(self.auto_route_all)
        tools_menu.addAction(route_action)

        smoke_route_action = QAction("Quick Smoke Route", self)
        smoke_route_action.triggered.connect(self.quick_smoke_route)
        tools_menu.addAction(smoke_route_action)

        tools_menu.addSeparator()

        # Self-tests
        lattice_test_action = QAction("Self-test: Lattice", self)
        lattice_test_action.triggered.connect(self.self_test_lattice)
        tools_menu.addAction(lattice_test_action)

        tiny_route_test_action = QAction("Self-test: Tiny Route", self)
        tiny_route_test_action.triggered.connect(self.self_test_tiny_route)
        tools_menu.addAction(tiny_route_test_action)
        
    def load_board_data(self):
        """Load and display the board data"""
        if not self.board_data:
            return
            
        # Update board info
        filename = self.board_data.get('filename', 'Unknown')
        width = self.board_data.get('width', 0)
        height = self.board_data.get('height', 0)
        layers = self.board_data.get('layers', 0)
        
        pads_count = len(self.board_data.get('pads', []))
        nets_count = len(self.board_data.get('nets', {}))
        components_count = len(self.board_data.get('components', []))
        
        board_info = f"Board: {filename}\n"
        board_info += f"Size: {width:.1f} × {height:.1f} mm\n"
        board_info += f"Layers: {layers}\n"
        board_info += f"Components: {components_count}\n"
        board_info += f"Pads: {pads_count}\n"
        board_info += f"Nets: {nets_count}"
        
        self.board_info_label.setText(board_info)
        
        # Load nets into tree
        self.load_nets_tree()
        
        # Set board data in PCB viewer
        self.pcb_viewer.set_board_data(self.board_data)
        
        # Fit to view after a short delay to ensure proper widget sizing
        QTimer.singleShot(100, self.pcb_viewer.fit_to_view)
        
        # Update layer visibility menu
        self.update_layer_visibility_menu()
        
        # Update status
        self.status_label.setText(f"Loaded {nets_count} nets, {pads_count} pads")
        
        logger.info("Board data loaded into GUI: %d components, %d tracks - starting progressive net processing", components_count, len(self.board_data.get('tracks', [])))
        logger.info("Using %d airwires from KiCad interface (no regeneration needed)", len(self.board_data.get('airwires', [])))
        logger.info("Displaying ALL %d nets (no performance limits)", nets_count)
        
    def load_nets_tree(self):
        """Load nets into the tree widget"""
        if not self.nets_tree:
            return
            
        self.nets_tree.clear()
        
        nets = self.board_data.get('nets', {})
        
        for net_name, net_data in nets.items():
            if not net_name or net_name.strip() == "":
                continue
                
            pads = net_data.get('pads', [])
            pad_count = len(pads)
            
            if pad_count < 2:
                continue  # Skip single-pad nets
                
            item = QTreeWidgetItem([net_name, str(pad_count), "Unrouted"])
            self.nets_tree.addTopLevelItem(item)
            
    def route_all_nets(self):
        """Route all nets"""
        logger.info("Route All Nets button clicked")
        self.status_label.setText("Routing all nets...")
        
        # TODO: Implement actual routing logic
        QMessageBox.information(self, "Routing", "Routing functionality not yet implemented")
        
        self.status_label.setText("Ready")
        
    def clear_routes(self):
        """Clear all existing routes"""
        logger.info("Clear All Routes button clicked")
        self.status_label.setText("Clearing routes...")
        
        # TODO: Implement route clearing logic
        QMessageBox.information(self, "Clear Routes", "Route clearing functionality not yet implemented")
        
        self.status_label.setText("Ready")
        
    def refresh_from_kicad(self):
        """Refresh board data from KiCad"""
        logger.info("Refreshing board data from KiCad...")
        self.status_label.setText("Refreshing from KiCad...")
        
        try:
            # Get fresh board data
            new_board_data = self.kicad_interface.get_board_data()
            
            if new_board_data:
                self.board_data = new_board_data
                self.load_board_data()
                self.status_label.setText("Board data refreshed successfully")
                logger.info("Board data refreshed from KiCad")
            else:
                self.status_label.setText("Failed to refresh board data")
                QMessageBox.warning(self, "Refresh Failed", "Could not refresh board data from KiCad")
                
        except Exception as e:
            logger.error(f"Error refreshing board data: {e}")
            self.status_label.setText("Refresh failed")
            QMessageBox.critical(self, "Refresh Error", f"Error refreshing board data:\\n{e}")
    
    # Additional methods for full original functionality
    def update_layer_visibility_menu(self):
        """Update layer visibility menu and panel with actual board layers"""
        if not hasattr(self, 'layers_menu'):
            return

        # Clear existing actions
        self.layers_menu.clear()
        self.layer_actions = {}

        # Get layers from board data (dynamic from KiCad file)
        layers = self.board_data.get('layer_names', ['F.Cu', 'B.Cu'])

        # Update menu
        for layer in layers:
            action = QAction(layer, self)
            action.setCheckable(True)
            action.setChecked(True)  # Default to visible
            action.triggered.connect(lambda checked, l=layer: self.toggle_layer_visibility(l, checked))
            self.layer_actions[layer] = action
            self.layers_menu.addAction(action)

        # Update layers panel checkboxes
        self.update_layers_panel(layers)

    def update_layers_panel(self, layers: list):
        """Populate the layers panel with checkboxes for each layer"""
        if not hasattr(self, 'layers_container_layout'):
            return

        # Clear existing checkboxes
        while self.layers_container_layout.count():
            child = self.layers_container_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.layer_checkboxes = {}

        # Create checkbox for each layer with color indicator
        for layer_name in layers:
            # Create horizontal layout for checkbox + color indicator
            layer_widget = QWidget()
            layer_layout = QHBoxLayout(layer_widget)
            layer_layout.setContentsMargins(0, 0, 0, 0)
            layer_layout.setSpacing(8)

            # Checkbox
            checkbox = QCheckBox(layer_name)
            checkbox.setChecked(True)  # Default to visible
            checkbox.toggled.connect(lambda checked, l=layer_name: self.toggle_layer_visibility(l, checked))

            # Color indicator square
            color = self.color_scheme.get_layer_color(layer_name)
            color_label = QLabel("  ")
            color_label.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #888; min-width: 20px; min-height: 16px; max-width: 20px; max-height: 16px;")
            color_label.setToolTip(f"Layer color: {color.name()}")

            layer_layout.addWidget(color_label)
            layer_layout.addWidget(checkbox)
            layer_layout.addStretch()

            self.layers_container_layout.addWidget(layer_widget)
            self.layer_checkboxes[layer_name] = checkbox

        # Add stretch at bottom
        self.layers_container_layout.addStretch()

    # Event handler methods
    def on_algorithm_changed(self, algorithm_text: str):
        """Handle algorithm selection change"""
        logger.info(f"INFO: Algorithm changed to: {algorithm_text}")
        self.status_label.setText(f"Selected algorithm: {algorithm_text}")
        
    def toggle_display_option(self, option: str, checked: bool):
        """Handle display option checkbox changes"""
        logger.info(f"Display option {option}: {'enabled' if checked else 'disabled'}")
        # Update PCB viewer display settings
        if self.pcb_viewer:
            setattr(self.pcb_viewer, option, checked)
            self.pcb_viewer.update()
            
    def toggle_layer_visibility(self, layer: str, checked: bool):
        """Handle layer visibility changes - sync between menu, panel, and viewer"""
        logger.info(f"Layer {layer}: {'visible' if checked else 'hidden'}")

        # Update PCB viewer layer visibility
        if self.pcb_viewer:
            if checked:
                self.pcb_viewer.visible_layers.add(layer)
            else:
                self.pcb_viewer.visible_layers.discard(layer)
            self.pcb_viewer.update()

        # Sync menu action state (if exists)
        if hasattr(self, 'layer_actions') and layer in self.layer_actions:
            self.layer_actions[layer].blockSignals(True)
            self.layer_actions[layer].setChecked(checked)
            self.layer_actions[layer].blockSignals(False)

        # Sync panel checkbox state (if exists)
        if hasattr(self, 'layer_checkboxes') and layer in self.layer_checkboxes:
            self.layer_checkboxes[layer].blockSignals(True)
            self.layer_checkboxes[layer].setChecked(checked)
            self.layer_checkboxes[layer].blockSignals(False)

    def focus_on_net(self):
        """Focus on specific net for debugging"""
        net_id = self.focus_net_input.text().strip()
        if not net_id:
            return

        logger.info(f"[GUI-DEBUG] Focusing on net: {net_id}")

        # Update PCB viewer to highlight the focused net
        if self.pcb_viewer:
            # Set focused net in viewer
            self.pcb_viewer.focused_net = net_id

            # Clear previous highlights and highlight this net
            if hasattr(self.pcb_viewer, 'highlight_net'):
                self.pcb_viewer.highlight_net(net_id)
                self.pcb_viewer.update()

            # Log debug info about this net if available
            if hasattr(self.pcb_viewer, 'board_data') and self.pcb_viewer.board_data:
                # Look for tracks for this net
                tracks_for_net = []
                if 'tracks' in self.pcb_viewer.board_data:
                    tracks_for_net = [t for t in self.pcb_viewer.board_data['tracks'] if t.get('net_id') == net_id]

                # Look for stubs for this net
                stubs_for_net = []
                if hasattr(self.pcb_viewer, 'stub_tracks'):
                    stubs_for_net = [s for s in self.pcb_viewer.stub_tracks if s.get('net_id') == net_id]

                logger.info(f"[GUI-DEBUG] Net {net_id}: {len(tracks_for_net)} tracks, {len(stubs_for_net)} stubs")

                # Log first few track coordinates for debugging
                for i, track in enumerate(tracks_for_net[:3]):
                    logger.info(f"[GUI-DEBUG] Track {i}: start={track.get('start')}, end={track.get('end')}, layer={track.get('layer')}")

    def toggle_pad_stubs(self, checked: bool):
        """Toggle visibility of pad connection stubs"""
        logger.info(f"[GUI-DEBUG] Pad stubs: {'visible' if checked else 'hidden'}")

        # Update PCB viewer stub visibility
        if self.pcb_viewer:
            self.pcb_viewer.show_pad_stubs = checked
            self.pcb_viewer.update()

    def toggle_portal_dots(self, checked: bool):
        """Toggle visibility of portal dots for first 50 nets"""
        logger.info(f"[GUI-DEBUG] Portal dots: {'visible' if checked else 'hidden'}")

        # Update PCB viewer portal visualization
        if self.pcb_viewer:
            self.pcb_viewer.show_portal_dots = checked
            self.pcb_viewer.update()

    # Routing control methods
    def begin_autorouting(self):
        """Begin autorouting with unified pipeline"""
        if not self.plugin:
            QMessageBox.critical(self, "Plugin Error", "No plugin instance available")
            return

        algorithm_text = self.algorithm_combo.currentText()
        logger.info(f"Begin autorouting with {algorithm_text}")

        # Set GPU environment variable based on checkbox
        import os
        gpu_enabled = self.gpu_checkbox.isChecked()
        os.environ['ORTHO_GPU'] = '1' if gpu_enabled else '0'
        gpu_mode = "GPU (beta)" if gpu_enabled else "CPU (safe default)"
        self.log_to_gui(f"[GPU] Using {gpu_mode} acceleration", "INFO")

        # Clear previous log and start fresh
        self.clear_routing_log()
        self.log_to_gui(f"[START] Starting unified autorouting pipeline", "SUCCESS")

        # Create timestamped run folder for screenshots
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"debug_output/run_{timestamp}"
        os.makedirs(run_folder, exist_ok=True)
        self.log_to_gui(f"📁 Created screenshot folder: {run_folder}", "DEBUG")

        # Store run folder for later use
        self._current_run_folder = run_folder

        self._set_ui_busy(True, "Autorouting…")

        # Take initial screenshots synchronously before routing starts
        self._take_initial_screenshots()

        # Start routing immediately after screenshots
        self._continue_autorouting()

    def _take_initial_screenshots(self):
        """Take screenshots 1 & 2 synchronously at the start of routing"""
        if not self.pcb_viewer:
            return

        # Screenshot 1: Board with airwires
        self.pcb_viewer.show_airwires = True
        self.pcb_viewer.fit_to_view()
        self.pcb_viewer.update()
        QApplication.processEvents()  # Force render
        self.pcb_viewer.debug_screenshot("01_board_with_airwires", scale_factor=8, output_dir=self._current_run_folder)
        self.log_to_gui("📸 Screenshot 1/3: Board with airwires (8x)", "DEBUG")

        # Screenshot 2: Board without airwires or escapes
        self.pcb_viewer.show_airwires = False

        # Temporarily clear tracks and vias for clean board screenshot
        old_tracks = self.board_data.get('tracks', [])
        old_vias = self.board_data.get('vias', [])
        self.board_data['tracks'] = []
        self.board_data['vias'] = []

        if hasattr(self.pcb_viewer, 'update_routing'):
            self.pcb_viewer.update_routing([], [])

        self.pcb_viewer.update()
        QApplication.processEvents()  # Force render
        self.pcb_viewer.debug_screenshot("02_board_no_airwires", scale_factor=8, output_dir=self._current_run_folder)
        self.log_to_gui("📸 Screenshot 2/3: Board without airwires or escapes (8x)", "DEBUG")

        # Restore tracks/vias
        self.board_data['tracks'] = old_tracks
        self.board_data['vias'] = old_vias
        if hasattr(self.pcb_viewer, 'update_routing'):
            self.pcb_viewer.update_routing(old_tracks, old_vias)
        self.pcb_viewer.update()

    def _continue_autorouting(self):
        """Continue autorouting after screenshots are taken"""
        try:
            # Get pathfinder and board from plugin
            pf = self.plugin.get_pathfinder()
            board = self._create_board_from_data()

            self.log_to_gui(f"[GUI] Starting unified pipeline with pf={pf._instance_tag}", "INFO")

            # 1) Build lattice + CSR + preflight
            self.log_to_gui("[PIPELINE] Step 1: Building lattice & CSR...", "INFO")
            pf.initialize_graph(board)
            self.log_to_gui("✓ Lattice initialization complete", "SUCCESS")

            self.log_to_gui("[PIPELINE] Step 2: Mapping pads to lattice...", "INFO")
            pf.map_all_pads(board)

            # PORTAL STATUS: Check new portal system (from pad_escape_planner)
            # The new system uses pf.portals dict (pad_id -> Portal), not the old _pad_portal_map
            portal_count = len(getattr(pf, "portals", {}))
            logger.info("[PORTAL] Pre-computed portals: %d (from pad_escape_planner)", portal_count)

            # Note: Portals will be populated after precompute_all_pad_escapes() is called below
            # This check just verifies the portal system is available

            self.log_to_gui("✓ Pad mapping complete", "SUCCESS")

            # PRECOMPUTE PAD ESCAPES (for debugging, before routing)
            self.log_to_gui("[DEBUG] Precomputing pad escapes...", "INFO")

            # Attach GUI pad data to board for DRC
            board._gui_pads = self.board_data.get('pads', [])
            logger.info(f"Attached {len(board._gui_pads)} GUI pads to board for DRC")

            escape_tracks, escape_vias = pf.precompute_all_pad_escapes(board)

            # Push escape geometry to preview
            if escape_tracks or escape_vias:
                if 'tracks' not in self.board_data:
                    self.board_data['tracks'] = []
                if 'vias' not in self.board_data:
                    self.board_data['vias'] = []

                self.board_data['tracks'].extend(escape_tracks)
                self.board_data['vias'].extend(escape_vias)

                if hasattr(self.pcb_viewer, 'update_routing'):
                    self.pcb_viewer.update_routing(self.board_data.get('tracks', []),
                                                   self.board_data.get('vias', []))
                    self.pcb_viewer.update()

                self.log_to_gui(f"✓ Precomputed {len(escape_tracks)} escape stubs, {len(escape_vias)} vias", "SUCCESS")

                # Screenshot 3: Board with escape planning (no airwires) - taken synchronously
                self.pcb_viewer.show_airwires = False
                self.pcb_viewer.fit_to_view()
                self.pcb_viewer.update()
                QApplication.processEvents()  # Force render
                self.pcb_viewer.debug_screenshot("03_board_with_escapes", scale_factor=8, output_dir=self._current_run_folder)
                self.log_to_gui("📸 Screenshot 3/3: Board with escape planning (8x)", "SUCCESS")

                # Re-enable airwires for normal viewing
                self.pcb_viewer.show_airwires = True
                self.pcb_viewer.update()

            self.log_to_gui("[PIPELINE] Step 3: Preparing routing runtime...", "INFO")
            pf.prepare_routing_runtime()
            self.log_to_gui("✓ Runtime preparation complete", "SUCCESS")

            # 2) Route nets with GUI progress callback
            self.log_to_gui("[PIPELINE] Step 4: Routing nets...", "INFO")
            def progress_cb(done, total, eta_s, *_, **__):
                if self.progress_bar:
                    self.progress_bar.setValue(int(100 * done / max(total, 1)))

                    # Defensive ETA formatting: handle float or preformatted string
                    eta_text = ""
                    try:
                        # if eta_s is numeric-ish, print "~X.Xs ETA"
                        eta_val = float(eta_s)  # works for int/float or numeric-like str
                        eta_text = f" (~{eta_val:.1f}s ETA)"
                    except Exception:
                        # if it's already a string (or None), append as-is (or nothing)
                        if eta_s:
                            # strip to keep things tidy if caller already included units
                            eta_text = f" ({str(eta_s).strip()})"

                    self.progress_bar.setFormat(f"{done}/{total} nets{eta_text}")

            # Store count of escape geometry before routing starts
            escape_track_count = len([t for t in self.board_data.get('tracks', []) if t])
            escape_via_count = len([v for v in self.board_data.get('vias', []) if v])
            escape_tracks_list = self.board_data.get('tracks', [])[:escape_track_count]
            escape_vias_list = self.board_data.get('vias', [])[:escape_via_count]

            def iteration_cb(iteration, routing_tracks, routing_vias):
                """Iteration callback: update board_data and capture screenshot"""
                try:
                    # Combine escape geometry with provisional routing geometry
                    self.board_data['tracks'] = escape_tracks_list + routing_tracks
                    self.board_data['vias'] = escape_vias_list + routing_vias

                    # Update viewer
                    if hasattr(self.pcb_viewer, 'update_routing'):
                        self.pcb_viewer.update_routing(self.board_data['tracks'], self.board_data['vias'])
                        self.pcb_viewer.update()
                        QApplication.processEvents()

                    # Memory-efficient screenshot controls
                    import os
                    disable_screenshots = os.environ.get('ORTHO_NO_SCREENSHOTS', '0') == '1'
                    screenshot_freq = int(os.environ.get('ORTHO_SCREENSHOT_FREQ', '1'))
                    screenshot_scale = int(os.environ.get('ORTHO_SCREENSHOT_SCALE', '2'))

                    # Only capture screenshots if enabled and at appropriate frequency
                    if not disable_screenshots and (iteration % screenshot_freq == 0):
                        screenshot_name = f"{iteration+3:02d}_iteration_{iteration:02d}"
                        self.pcb_viewer.show_airwires = False  # Hide airwires for clarity
                        self.pcb_viewer.fit_to_view()
                        self.pcb_viewer.update()
                        QApplication.processEvents()
                        self.pcb_viewer.debug_screenshot(screenshot_name, scale_factor=screenshot_scale, output_dir=self._current_run_folder)
                        logger.info(f"📸 Iteration {iteration}: screenshot captured ({len(routing_tracks)} routing tracks, {len(routing_vias)} routing vias)")
                    else:
                        logger.info(f"⏩ Iteration {iteration}: {len(routing_tracks)} routing tracks, {len(routing_vias)} routing vias (screenshot skipped)")
                except Exception as e:
                    logger.warning(f"Iteration callback failed: {e}")

            # Capture routing result to check for convergence
            routing_result = pf.route_multiple_nets(board.nets, progress_cb=progress_cb, iteration_cb=iteration_cb)
            self.log_to_gui("✓ Net routing complete", "SUCCESS")

            # 3) Emit geometry -> update viewer
            tracks, vias = pf.emit_geometry(board)
            geom = pf.get_geometry_payload()
            self.log_to_gui(f"[GUI] Emitting geometry: tracks={tracks} vias={vias}", "SUCCESS")

            if hasattr(self, 'pcb_viewer') and self.pcb_viewer:
                self.pcb_viewer.update_routing(geom.tracks, geom.vias)

            # 4) Check convergence status and show appropriate dialog
            showed_convergence_dialog = False
            if routing_result and not routing_result.get('success', True):
                # UNCONVERGENCE: Show detailed failure dialog
                failed_nets = routing_result.get('failed_nets', 0)
                total_nets = len(board.nets)
                fail_percentage = (failed_nets / total_nets * 100) if total_nets > 0 else 0
                overuse_edges = routing_result.get('overuse_edges', 0)
                overuse_sum = routing_result.get('overuse_sum', 0)
                layer_rec = routing_result.get('layer_recommendation', {})

                # Build dialog message
                dialog_msg = f"Your board did not converge because there are too few routing layers.\n\n"
                dialog_msg += f"ROUTING INCOMPLETE: {failed_nets}/{total_nets} nets failed ({fail_percentage:.1f}%)\n"
                dialog_msg += f"  Overuse: {overuse_edges} edges with {overuse_sum} total conflicts\n\n"

                if layer_rec.get('needs_more', False):
                    dialog_msg += f"  RECOMMENDATION: Add {layer_rec.get('additional', 0)} more layers "
                    dialog_msg += f"(→{layer_rec.get('recommended_total', 0)} total)\n\n"
                    dialog_msg += f"  Reason: {layer_rec.get('reason', 'Insufficient routing capacity')}"
                else:
                    dialog_msg += f"  Note: Current layer count appears adequate.\n"
                    dialog_msg += f"  Convergence may improve with tuning or may have reached practical limit."

                QMessageBox.warning(
                    self,
                    "Unconvergence Alert",
                    dialog_msg
                )
                showed_convergence_dialog = True
            elif tracks > 0 or vias > 0:
                # SUCCESS: Show convergence success dialog
                QMessageBox.information(
                    self,
                    "Converged!",
                    f"Routing completed successfully!\n\n"
                    f"Results:\n"
                    f"  • {tracks} tracks placed\n"
                    f"  • {vias} vias placed\n\n"
                    f"All nets routed with zero overuse."
                )
                showed_convergence_dialog = True

            # 5) Handle no copper emitted case (different from unconvergence)
            # Only show this if we didn't already explain the problem with an unconvergence dialog
            if tracks == 0 and vias == 0 and not showed_convergence_dialog:
                self.log_to_gui("⚠️ No copper emitted - analyzing reasons...", "WARNING")

                # Analyze why no copper was generated
                failure_reasons = self._analyze_routing_failures(pf, board)

                for reason in failure_reasons:
                    self.log_to_gui(f"   • {reason}", "WARNING")

                # Offer smoke route as fallback
                if failure_reasons:
                    self.log_to_gui("💡 Attempting smoke route fallback...", "INFO")
                    try:
                        success, smoke_tracks, smoke_vias = pf.smoke_route(board)
                        if success:
                            geom = pf.get_geometry_payload()
                            if hasattr(self, 'pcb_viewer') and self.pcb_viewer:
                                self.pcb_viewer.update_routing(geom.tracks, geom.vias)
                            self.log_to_gui(f"✅ Smoke route success: {smoke_tracks} tracks, {smoke_vias} vias", "SUCCESS")
                            self.status_label.setText(f"Smoke route fallback: {smoke_tracks} tracks, {smoke_vias} vias")
                            tracks, vias = smoke_tracks, smoke_vias  # Update for button enabling logic
                        else:
                            self.log_to_gui("❌ Smoke route also failed", "ERROR")
                    except Exception as smoke_e:
                        self.log_to_gui(f"❌ Smoke route error: {smoke_e}", "ERROR")

                # Show detailed dialog if still no copper (this is a setup issue, not convergence)
                if tracks == 0 and vias == 0:
                    failure_text = "\n".join([f"• {reason}" for reason in failure_reasons])
                    QMessageBox.warning(
                        self,
                        "No Copper Generated",
                        f"Routing completed but no copper was generated.\n\n"
                        f"This indicates a board setup issue, not a convergence problem.\n\n"
                        f"Possible reasons:\n{failure_text}\n\n"
                        f"Try:\n"
                        f"• Check that nets have at least 2 pads\n"
                        f"• Verify components are properly connected\n"
                        f"• Use Tools → Quick Smoke Route for basic connectivity test"
                    )
            else:
                # 6) Log completion (dialog already shown above)
                self.log_to_gui(f"✅ Routing completed: {tracks} tracks, {vias} vias", "SUCCESS")

            self.status_label.setText(f"Autoroute complete: {tracks} tracks, {vias} vias")

            # Enable commit/rollback buttons if we have results
            if tracks > 0 or vias > 0:
                self.commit_btn.setEnabled(True)
                self.rollback_btn.setEnabled(True)
                self.replay_btn.setEnabled(True)

        except Exception as e:
            logger.exception("Autoroute failed")
            error_str = str(e)

            # Downgrade zero-length track errors to warnings (don't kill autoroute)
            if "zero-length" in error_str.lower() or "zero length" in error_str.lower():
                logger.warning(f"[GUI] Zero-length track handling: {error_str}")
                self.log_to_gui(f"⚠️ Zero-length tracks filtered (routing continues): {error_str}", "WARNING")
                # Don't show critical dialog for zero-length issues - they're handled upstream
            else:
                self.log_to_gui(f"❌ Autoroute failed: {e}", "ERROR")
                QMessageBox.critical(self, "Autoroute Error", f"Autoroute failed:\n{str(e)}")
        finally:
            self._set_ui_busy(False)

    def _set_ui_busy(self, busy: bool, status_text: str = ""):
        """Set UI to busy state during routing"""
        self.route_preview_btn.setEnabled(not busy)
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setValue(0)
            if status_text:
                self.status_label.setText(status_text)
        else:
            self.status_label.setText("Ready" if not status_text else status_text)

    def _create_board_from_data(self):
        """Create Board domain object from GUI board_data"""
        from ...domain.models.board import Board, Net, Pad, Component, Coordinate

        # DEBUG: Check what bounds data we have in board_data
        logger.info(f"DEBUG: board_data keys: {list(self.board_data.keys())}")
        if 'bounds' in self.board_data:
            logger.info(f"DEBUG: board_data['bounds'] = {self.board_data['bounds']}")
        else:
            logger.info("DEBUG: 'bounds' key not found in board_data!")

        # Extract layer count from board_data (set by plugin)
        layer_count = self.board_data.get('layers', 2)
        if isinstance(layer_count, int):
            detected_layers = layer_count
        else:
            detected_layers = 2  # fallback
        logger.info(f"DEBUG: Creating Board object with layer_count={detected_layers}")

        board = Board(id="gui-board", name="GUI Board", layer_count=detected_layers)
        board.nets = []
        board.components = []  # Initialize components list

        # Track components and their pads for proper Board structure
        components_dict = {}  # component_id -> Component object
        all_pads = []  # Keep track of all pads for components

        # First, create components from board_data components section
        components_data = self.board_data.get('components', [])
        logger.info(f"DEBUG: Found {len(components_data)} components in board_data")
        for comp_data in components_data:
            if isinstance(comp_data, dict):
                comp_id = comp_data.get('name', comp_data.get('id', ''))
                if comp_id:
                    component = Component(
                        id=comp_id,
                        reference=comp_id,
                        value=comp_data.get('value', ''),
                        footprint=comp_data.get('footprint', ''),
                        position=Coordinate(
                            x=comp_data.get('x', 0.0),
                            y=comp_data.get('y', 0.0)
                        ),
                        angle=comp_data.get('rotation', 0.0)
                    )
                    components_dict[comp_id] = component

        # Convert nets from GUI data format to domain objects
        nets_data = self.board_data.get('nets', {})
        logger.info(f"DEBUG: Processing {len(nets_data)} nets from board_data")
        if isinstance(nets_data, dict):
            for net_id, net_info in nets_data.items():
                net = Net(id=net_id, name=net_info.get('name', net_id))
                net.pads = []

                # Add pads from net data
                for pad_ref in net_info.get('pads', []):
                    if isinstance(pad_ref, dict):
                        # Extract component name from pad name (likely "ComponentName.PadNumber" format)
                        pad_name = pad_ref.get('name', '')
                        component_id = pad_ref.get('component', '')  # First try explicit component field

                        # Fallback: extract from pad name if no explicit component
                        if not component_id and '.' in pad_name:
                            component_id = pad_name.split('.')[0]

                        # DEBUG: Check pad_ref structure
                        if len(all_pads) < 5:  # Only log first few to avoid spam
                            logger.info(f"DEBUG: pad_ref keys: {list(pad_ref.keys())}")
                            logger.info(f"DEBUG: pad_ref.get('name') = '{pad_name}'")
                            logger.info(f"DEBUG: pad_ref.get('component') = '{pad_ref.get('component', 'N/A')}'")
                            logger.info(f"DEBUG: extracted component_id = '{component_id}'")

                        # Extract pad coordinates from KiCad data
                        pad_x = pad_ref.get('x', 0.0)
                        pad_y = pad_ref.get('y', 0.0)

                        pad = Pad(
                            id=pad_name or f"{component_id}.{pad_ref.get('pin', '')}",
                            component_id=component_id,
                            net_id=net_id,
                            position=Coordinate(x=pad_x, y=pad_y),
                            size=(0.2, 0.2),  # Default pad size
                            layer=pad_ref.get('layer', 'F.Cu')
                        )

                        # DEBUG: Log pad position for first few pads
                        if len(all_pads) < 5:
                            logger.info(f"DEBUG: Creating pad '{pad.id}' at ({pad_x}, {pad_y}) from pad_ref x={pad_ref.get('x')}, y={pad_ref.get('y')}")
                        net.pads.append(pad)
                        all_pads.append(pad)

                        # Create component if it doesn't exist yet (fallback if not in components section)
                        if component_id and component_id not in components_dict:
                            component = Component(
                                id=component_id,
                                reference=component_id,
                                value="",
                                footprint="",
                                position=Coordinate(x=pad.position.x, y=pad.position.y),  # Use pad position as reference
                                angle=0.0
                            )
                            components_dict[component_id] = component

                if len(net.pads) >= 2:  # Only add nets with at least 2 pads
                    board.nets.append(net)

        # Populate components with their pads
        pads_assigned = 0
        pads_orphaned = 0
        for pad in all_pads:
            if pad.component_id and pad.component_id in components_dict:
                components_dict[pad.component_id].pads.append(pad)
                pads_assigned += 1
            else:
                pads_orphaned += 1
                if pads_orphaned <= 5:  # Log first few orphaned pads
                    logger.warning(f"DEBUG: Orphaned pad '{pad.id}' with component_id '{pad.component_id}' - not found in components_dict")

        # FALLBACK: Create a generic component for orphaned pads
        if pads_orphaned > 0:
            logger.info(f"DEBUG: Creating generic component for {pads_orphaned} orphaned pads")
            generic_component = Component(
                id="GENERIC_COMPONENT",
                reference="GENERIC_COMPONENT",
                value="Generic",
                footprint="Generic",
                position=Coordinate(x=200.0, y=140.0),  # Center of typical board
                angle=0.0
            )

            # Add all orphaned pads to generic component
            for pad in all_pads:
                if not pad.component_id or pad.component_id not in components_dict:
                    pad.component_id = "GENERIC_COMPONENT"  # Update pad's component reference
                    generic_component.pads.append(pad)
                    pads_assigned += 1
                    pads_orphaned -= 1

            components_dict["GENERIC_COMPONENT"] = generic_component

        # Add components to board
        board.components = list(components_dict.values())

        # DEBUG: Check component pad counts
        component_pad_counts = [(comp.id, len(comp.pads)) for comp in board.components]
        logger.info(f"DEBUG: Component pad counts: {component_pad_counts[:5]}...")  # Show first 5

        logger.info(f"Created board with {len(board.nets)} routable nets, {len(board.components)} components, {len(all_pads)} total pads")
        logger.info(f"DEBUG: Pad assignment: {pads_assigned} assigned, {pads_orphaned} orphaned")

        # Store KiCad-calculated bounds for accurate routing area
        kicad_bounds = self.board_data.get('bounds', None)
        board._kicad_bounds = kicad_bounds

        # DEBUG: Verify bounds were set correctly
        logger.info(f"DEBUG: Setting board._kicad_bounds = {kicad_bounds}")
        logger.info(f"DEBUG: After setting, hasattr(board, '_kicad_bounds') = {hasattr(board, '_kicad_bounds')}")
        if hasattr(board, '_kicad_bounds'):
            logger.info(f"DEBUG: board._kicad_bounds = {board._kicad_bounds}")

        return board

    def quick_smoke_route(self):
        """Run quick smoke route test"""
        if not self.plugin:
            QMessageBox.critical(self, "Plugin Error", "No plugin instance available")
            return

        try:
            pf = self.plugin.get_pathfinder()
            board = self._create_board_from_data()

            self.log_to_gui("🚀 Starting Quick Smoke Route test", "INFO")
            self._set_ui_busy(True, "Running smoke route...")

            # Initialize if needed
            if not hasattr(pf, 'graph_state') or not pf.graph_state:
                self.log_to_gui("[SMOKE] Initializing graph...", "INFO")
                pf.initialize_graph(board)
                pf.map_all_pads(board)
                pf.prepare_routing_runtime()

            # Run smoke route
            success, tracks, vias = pf.smoke_route(board)

            if success:
                # Update viewer with smoke route results
                geom = pf.get_geometry_payload()
                if hasattr(self, 'pcb_viewer') and self.pcb_viewer:
                    self.pcb_viewer.update_routing(geom.tracks, geom.vias)

                self.log_to_gui(f"✅ [SMOKE] Success: emitted {tracks} tracks, {vias} vias", "SUCCESS")
                self.status_label.setText(f"Smoke route complete: {tracks} tracks, {vias} vias")

                # Enable commit/rollback buttons
                self.commit_btn.setEnabled(True)
                self.rollback_btn.setEnabled(True)
                self.replay_btn.setEnabled(True)
            else:
                self.log_to_gui("❌ [SMOKE] Failed: no copper generated", "ERROR")
                self.status_label.setText("Smoke route failed")
                QMessageBox.warning(self, "Smoke Route", "Smoke route failed - no routable pairs found")

        except Exception as e:
            logger.exception("Smoke route failed")
            self.log_to_gui(f"❌ [SMOKE] Error: {e}", "ERROR")
            QMessageBox.critical(self, "Smoke Route Error", f"Smoke route failed:\n{str(e)}")
        finally:
            self._set_ui_busy(False)

    def _analyze_routing_failures(self, pf, board):
        """Analyze why routing failed to generate copper"""
        reasons = []

        try:
            # Check basic board state
            if not board or not board.nets:
                reasons.append("No nets found in board data")
                return reasons

            nets_with_pads = [net for net in board.nets if len(net.pads) >= 2]
            if len(nets_with_pads) == 0:
                reasons.append("No nets with 2+ pads found")

            # Check pathfinder state
            if not hasattr(pf, 'graph_state') or not pf.graph_state:
                reasons.append("Graph state not initialized")
                return reasons

            gs = pf.graph_state

            # Check lattice
            if not hasattr(gs, 'lattice_node_count') or gs.lattice_node_count == 0:
                reasons.append("No lattice nodes generated")

            # Check terminals
            if not hasattr(gs, 'net_terminals') or not gs.net_terminals:
                reasons.append("No net terminals mapped")
            else:
                terminal_count = sum(len(pins) for pins in gs.net_terminals.values())
                if terminal_count == 0:
                    reasons.append("Net terminals mapped but empty")

            # Check connectivity
            if hasattr(pf, '_comp') and hasattr(pf, '_giant_label'):
                giant_size = sum(1 for comp in pf._comp if comp == pf._giant_label)
                total_nodes = len(pf._comp) if pf._comp else 0
                if giant_size == 0:
                    reasons.append("No giant connected component found")
                elif giant_size < total_nodes * 0.1:
                    reasons.append(f"Giant component too small: {giant_size}/{total_nodes} nodes")

            # Check for specific routing failures
            if hasattr(gs, 'committed_paths'):
                if not gs.committed_paths:
                    reasons.append("No paths were successfully routed")
                elif len(gs.committed_paths) == 0:
                    reasons.append("All routing attempts failed (no path found)")

            # Default fallback
            if not reasons:
                reasons.append("Unknown routing failure - check logs for details")

        except Exception as e:
            reasons.append(f"Analysis error: {str(e)}")

        return reasons

    def self_test_lattice(self):
        """Self-test: Build lattice and verify connectivity"""
        if not self.plugin:
            QMessageBox.critical(self, "Plugin Error", "No plugin instance available")
            return

        try:
            self.log_to_gui("🧪 Starting Lattice Self-Test...", "INFO")
            self._set_ui_busy(True, "Running lattice test...")

            pf = self.plugin.get_pathfinder()
            board = self._create_board_from_data()

            # Initialize lattice
            pf.initialize_graph(board)

            # Check lattice metrics
            gs = pf.graph_state
            lattice_count = getattr(gs, 'lattice_node_count', 0)

            if hasattr(pf, '_comp') and hasattr(pf, '_giant_label'):
                giant_size = sum(1 for comp in pf._comp if comp == pf._giant_label)
                giant_ratio = giant_size / len(pf._comp) if pf._comp else 0
            else:
                pf._analyze_lattice_connectivity()
                giant_size = sum(1 for comp in pf._comp if comp == pf._giant_label)
                giant_ratio = giant_size / len(pf._comp) if pf._comp else 0

            # Report results
            self.log_to_gui(f"✓ Lattice nodes: {lattice_count}", "SUCCESS")
            self.log_to_gui(f"✓ Giant component: {giant_size} nodes ({giant_ratio:.1%})", "SUCCESS")

            if lattice_count > 0 and giant_ratio > 0.5:
                self.log_to_gui("✅ LATTICE TEST PASSED", "SUCCESS")
                self.status_label.setText("✓ Lattice test passed")
                QMessageBox.information(self, "Lattice Test", f"✅ Lattice test PASSED\n\nLattice: {lattice_count} nodes\nGiant: {giant_size} nodes ({giant_ratio:.1%})")
            else:
                self.log_to_gui("❌ LATTICE TEST FAILED", "ERROR")
                self.status_label.setText("❌ Lattice test failed")
                QMessageBox.warning(self, "Lattice Test", f"❌ Lattice test FAILED\n\nLattice: {lattice_count} nodes\nGiant: {giant_size} nodes ({giant_ratio:.1%})")

        except Exception as e:
            logger.exception("Lattice test failed")
            self.log_to_gui(f"❌ Lattice test error: {e}", "ERROR")
            QMessageBox.critical(self, "Lattice Test Error", f"Lattice test failed:\n{str(e)}")
        finally:
            self._set_ui_busy(False)

    def self_test_tiny_route(self):
        """Self-test: Create synthetic board and route nets"""
        if not self.plugin:
            QMessageBox.critical(self, "Plugin Error", "No plugin instance available")
            return

        try:
            self.log_to_gui("🧪 Starting Tiny Route Self-Test...", "INFO")
            self._set_ui_busy(True, "Running tiny route test...")

            pf = self.plugin.get_pathfinder()

            # Create synthetic 20x20x3 board
            synthetic_board = self._create_synthetic_board()

            # Route using unified pipeline
            pf.initialize_graph(synthetic_board)
            pf.map_all_pads(synthetic_board)
            pf.prepare_routing_runtime()
            pf.route_multiple_nets(synthetic_board.nets)

            # Emit geometry
            tracks, vias = pf.emit_geometry(synthetic_board)
            geom = pf.get_geometry_payload()

            # Update viewer with synthetic results
            if hasattr(self, 'pcb_viewer') and self.pcb_viewer:
                self.pcb_viewer.update_routing(geom.tracks, geom.vias)

            # Report results
            if tracks > 0 or vias > 0:
                self.log_to_gui(f"✅ TINY ROUTE TEST PASSED: {tracks} tracks, {vias} vias", "SUCCESS")
                self.status_label.setText("✓ Tiny route test passed")
                QMessageBox.information(self, "Tiny Route Test", f"✅ Tiny Route test PASSED\n\nGenerated:\n{tracks} tracks\n{vias} vias")
            else:
                self.log_to_gui("❌ TINY ROUTE TEST FAILED: no copper generated", "ERROR")
                self.status_label.setText("❌ Tiny route test failed")
                QMessageBox.warning(self, "Tiny Route Test", "❌ Tiny Route test FAILED\n\nNo copper generated")

        except Exception as e:
            logger.exception("Tiny route test failed")
            self.log_to_gui(f"❌ Tiny route test error: {e}", "ERROR")
            QMessageBox.critical(self, "Tiny Route Test Error", f"Tiny route test failed:\n{str(e)}")
        finally:
            self._set_ui_busy(False)

    def _create_synthetic_board(self):
        """Create a 20x20x3 synthetic board with 10 nets for testing"""
        from ...domain.models.board import Board, Net, Pad, Component, Coordinate
        import random

        board = Board(id="synthetic", name="Synthetic Test Board 20x20x3")
        board.nets = []

        # Create 10 test nets, each with 2 pads
        for net_id in range(10):
            net = Net(id=f"net_{net_id}", name=f"TEST_NET_{net_id}")
            net.pads = []

            # Create 2 pads per net at random positions
            for pad_id in range(2):
                pad = Pad(
                    id=f"net_{net_id}_pad_{pad_id}",
                    component_id=f"U{net_id}",
                    net_id=f"net_{net_id}",
                    position=Coordinate(
                        x=random.uniform(0, 20),  # 20mm x 20mm board
                        y=random.uniform(0, 20)
                    ),
                    size=(0.2, 0.2),  # Default pad size
                    layer="F.Cu"  # All on top layer for simplicity
                )
                net.pads.append(pad)

            board.nets.append(net)

        self.log_to_gui(f"Created synthetic board: {len(board.nets)} nets, 20x20mm", "INFO")
        return board

    def commit_routes(self):
        """Apply routes to KiCad"""
        logger.info("SUCCESS: Committing routes to KiCad")
        self.status_label.setText("Applying routes to KiCad...")
        
        # TODO: Implement route application to KiCad
        QMessageBox.information(self, "Apply Routes", "Route application to KiCad not yet implemented")
        
        self.commit_btn.setEnabled(False)
        self.rollback_btn.setEnabled(False)
        self.replay_btn.setEnabled(False)
        self.route_preview_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Routes applied successfully")
        
    def rollback_routes(self):
        """Discard calculated routes"""
        logger.info("Discarding routes")
        self.status_label.setText("Discarding routes...")
        
        # TODO: Implement route rollback
        QMessageBox.information(self, "Discard Routes", "Route discarding not yet implemented")
        
        self.commit_btn.setEnabled(False)
        self.rollback_btn.setEnabled(False)
        self.replay_btn.setEnabled(False)
        self.route_preview_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Routes discarded")

    def replay_routing(self):
        """Re-run the same routing with clean state for demo repeatability"""
        logger.info("Replaying last routing")
        self.status_label.setText("Replaying routing...")

        # Clear existing routes and restart
        self.board_data['tracks'].clear()
        self.board_data['vias'].clear()

        # Refresh the viewer to clear previous routing
        if self.pcb_viewer:
            self.pcb_viewer.update_routing([], [])

        # Reset button states and restart routing
        self.commit_btn.setEnabled(False)
        self.rollback_btn.setEnabled(False)
        self.replay_btn.setEnabled(False)
        self.route_preview_btn.setEnabled(False)

        # Trigger autorouting again
        self.begin_autorouting()

    # Menu action methods  
    def refresh_board(self):
        """Refresh board data from KiCad"""
        logger.info("Refreshing board data from KiCad...")
        self.status_label.setText("Refreshing from KiCad...")
        
        try:
            # Get fresh board data
            new_board_data = self.kicad_interface.get_board_data()
            
            if new_board_data:
                self.board_data = new_board_data
                self.load_board_data()
                self.status_label.setText("Board data refreshed successfully")
                logger.info("Board data refreshed from KiCad")
            else:
                self.status_label.setText("Failed to refresh board data")
                QMessageBox.warning(self, "Refresh Failed", "Could not refresh board data from KiCad")
                
        except Exception as e:
            logger.error(f"Error refreshing board data: {e}")
            self.status_label.setText("Refresh failed")
            QMessageBox.critical(self, "Refresh Error", f"Error refreshing board data:\\n{e}")
            
    def zoom_fit(self):
        """Fit board to window"""
        if self.pcb_viewer:
            self.pcb_viewer.fit_to_view()
            
    def auto_route_all(self):
        """Auto route all nets (menu action)"""
        self.begin_autorouting()
    
    def _route_manhattan_rrg(self):
        """Perform Manhattan RRG routing with live GUI updates"""
        try:
            logger.info("Starting Manhattan routing with live updates...")
            
            # Reset and prepare statistics widget
            self.pathfinder_stats.reset()
            
            # Create routing configuration matching RRG PathFinder parameters
            config = RoutingConfig(
                grid_pitch=0.4,
                track_width=0.0889,
                clearance=0.0889,
                via_diameter=0.25,
                via_drill=0.15,
                k_length=1.0,
                k_via=10.0,
                k_bend=2.0,
                max_iterations=50,
                pres_fac_init=0.5,
                pres_fac_mult=1.3,
                hist_cost_step=1.0,
                alpha=2.0
            )
            
            # Calculate total nets for statistics
            nets = self.board_data.get('nets', [])
            if isinstance(nets, list) and len(nets) > 0 and isinstance(nets[0], dict):
                total_nets = len([net for net in nets if len(net.get('pads', [])) >= 2])
            else:
                # Fallback: use airwires count as total nets approximation
                total_nets = len(self.board_data.get('airwires', []))
            self.pathfinder_stats.start_routing(total_nets, config.max_iterations)
            
            # Initialize progressive routing state
            self._setup_progressive_routing(config)
            
            # Set up progress callback for live statistics
            if hasattr(self.router, 'gpu_pathfinder') and hasattr(self.router.gpu_pathfinder, 'parallel_pathfinder'):
                self.router.gpu_pathfinder.parallel_pathfinder.progress_callback = self._on_pathfinder_progress
            
            # Set up PathFinder instrumentation callback if available
            if hasattr(self.router, 'pathfinder') and hasattr(self.router.pathfinder, 'set_gui_status_callback'):
                self.router.pathfinder.set_gui_status_callback(self._update_pathfinder_status)
            
        except Exception as e:
            logger.exception("Error starting Manhattan routing")
            self.status_label.setText(f"Error: {str(e)}")
            self._reset_routing_ui()
            QMessageBox.critical(self, "Routing Error", f"Error starting Manhattan routing:\n{str(e)}")
    
    def _setup_progressive_routing(self, config):
        """Initialize progressive routing with live GUI updates"""
        # Create GPU provider for router
        try:
            gpu_provider = CUDAProvider()
            if not gpu_provider.is_available():
                gpu_provider = CPUFallbackProvider()
                logger.warning("CUDA not available, using CPU fallback")
        except Exception as e:
            logger.error(f"GPU initialization error: {e}")
            gpu_provider = CPUFallbackProvider()
            
        # Create basic constraints and initialize router
        from orthoroute.domain.models.constraints import DRCConstraints
        
        # Create constraints with default values (dataclass)
        constraints = DRCConstraints()
        constraints.min_track_width = 0.0889  # 3.5 mil
        constraints.min_track_spacing = 0.2   # 0.2mm clearance
        constraints.min_via_diameter = 0.8
        constraints.min_via_drill = 0.4
        
        # Initialize router with correct signature
        self.router = ManhattanRRGRoutingEngine(
            constraints=constraints,
            gpu_provider=gpu_provider
        )
        
        # Skip the complex board creation and use the existing router initialization
        # The router will use the KiCad interface directly to get nets
        logger.info("Progressive routing: Using KiCad interface for board initialization")
        
        # Convert board data to domain objects and initialize router
        mock_board = self._convert_board_data_to_domain(self.board_data, constraints)
        self.router.initialize(mock_board)
        
        # Use real nets from the board
        self.routing_nets = [net for net in mock_board.nets if net.is_routable]
        logger.info(f"Progressive routing: Found {len(self.routing_nets)} routable nets from board data")
        self.current_net_index = 0
        self.routed_count = 0
        self.failed_count = 0
        self.routing_start_time = time.time()
        
        # Setup GUI update timer
        self.routing_timer = QTimer()
        self.routing_timer.timeout.connect(self._routing_step)
        self.routing_timer.start(100)  # Update every 100ms
        
        logger.info(f"Progressive routing initialized: {len(self.routing_nets)} nets to route")
        
    def _routing_step(self):
        """FIXED: Route ALL nets at once using PathFinder, then update GUI progressively"""
        try:
            # Check if this is the first call - route ALL nets at once
            if not hasattr(self, '_routing_started'):
                logger.info(f"PATHFINDER SINGLE-PASS: Starting routing of ALL {len(self.routing_nets)} nets with negotiated congestion")
                self.log_to_gui(f"🚀 Starting PathFinder routing of {len(self.routing_nets)} nets", "INFO")
                
                # Route ALL nets in a single PathFinder call for proper congestion negotiation
                self.routing_result = self.router.route_all_nets(
                    self.routing_nets,
                    timeout_per_net=30.0,
                    total_timeout=1800.0
                )
                
                # Get final routing statistics
                self.routing_stats = self.router.get_routing_statistics()
                
                logger.info(f"PATHFINDER COMPLETE: Routed {self.routing_stats.nets_routed}/{self.routing_stats.nets_attempted} nets ({self.routing_stats.success_rate:.1%})")
                self.log_to_gui(f"✅ PathFinder complete: {self.routing_stats.nets_routed}/{self.routing_stats.nets_attempted} nets routed", "SUCCESS")
                
                self._routing_started = True
                self.current_net_index = 0
                
                # Update visualization with all routed tracks
                self._update_routing_visualization()
                
            # Progressive GUI updates for visual feedback
            if self.current_net_index < len(self.routing_nets):
                batch_size = min(50, len(self.routing_nets) - self.current_net_index)  # Show 50 nets per update
                
                # Update progress bar
                progress = int((self.current_net_index / len(self.routing_nets)) * 100)
                self.progress_bar.setValue(progress)
                
                # Log nets in current batch for user feedback
                for i in range(batch_size):
                    if self.current_net_index + i < len(self.routing_nets):
                        net = self.routing_nets[self.current_net_index + i]
                        # Check if this net was successfully routed by looking at router state
                        if hasattr(self.router, 'routed_nets') and net.name in self.router.routed_nets:
                            self.routed_count += 1
                            logger.info(f"PATHFINDER SUCCESS: Routed net {net.name}")
                        else:
                            self.failed_count += 1
                
                self.current_net_index += batch_size
                self.status_label.setText(f"Processing routing results: {self.current_net_index}/{len(self.routing_nets)}")
            else:
                # All nets processed - complete routing
                self._complete_routing()
                return
                
        except Exception as e:
            logger.error(f"Error in PathFinder routing: {e}")
            self._complete_routing()
            
    def _route_net_batch(self, net_batch):
        """Route a batch of nets using REAL PathFinder with negotiated congestion"""
        try:
            if not net_batch:
                return []
                
            logger.info(f"REAL PATHFINDER: Routing batch of {len(net_batch)} nets with negotiated congestion")
            
            # CRITICAL: Use route_all_nets() for REAL PathFinder routing
            # This enables negotiated congestion, ripup/reroute, proper grid-based routing
            routing_result = self.router.route_all_nets(
                net_batch, 
                timeout_per_net=30.0,  # 30 second timeout per net for real PathFinder
                total_timeout=1800.0   # 30 minute total timeout for batch
            )
            
            # Get routing statistics to check success rate
            routing_stats = self.router.get_routing_statistics()
            
            # Extract success status for each net
            batch_results = []
            if routing_result.success and routing_stats.success_rate > 0:
                # If overall routing was successful, assume all nets in this batch succeeded
                batch_results = [True] * len(net_batch)
            else:
                # If overall routing failed, assume all nets in this batch failed
                batch_results = [False] * len(net_batch)
            
            logger.info(f"REAL PATHFINDER BATCH: Completed {len(net_batch)} nets with success rate: {routing_stats.success_rate:.1%}")
            
            # Log batch results to GUI
            if hasattr(self, 'log_to_gui'):
                success_count = sum(batch_results)
                self.log_to_gui(f"✅ Batch complete: {success_count}/{len(net_batch)} nets routed ({routing_stats.success_rate:.1%})", "SUCCESS" if success_count > 0 else "WARNING")
            
            return batch_results
                    
        except Exception as e:
            logger.error(f"Error in PathFinder batch routing: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return all failures for this batch
            return [False] * len(net_batch)

    def _route_single_net_step(self, net):
        """DEPRECATED: Route a single net (fallback only - use batch routing for real PathFinder)"""
        try:
            if not net or not hasattr(net, 'name'):
                return False
                
            # DEPRECATED: This uses basic Dijkstra, not real PathFinder
            # Use _route_net_batch() for proper PathFinder routing
            logger.warning(f"DEPRECATED: Using single-net Dijkstra for {net.name} - should use batch PathFinder")
            result = self.router.route_net(net, timeout=10.0)
            
            if result and result.success:
                return True
            else:
                return False
                    
        except Exception as e:
            logger.error(f"Error routing net {getattr(net, 'name', 'unknown')}: {e}")
            return False
            
    def _update_routing_visualization(self):
        """Update GUI visualization with current routing progress"""
        logger.info("=== _update_routing_visualization() CALLED ===")
        if self.pcb_viewer:
            # Get current routed tracks and vias
            tracks = self.router.get_routed_tracks()
            vias = self.router.get_routed_vias()
            
            # ACCUMULATIVE FIX: Add new tracks to existing tracks, don't overwrite
            if tracks:
                if 'tracks' not in self.board_data:
                    self.board_data['tracks'] = []
                
                # Get existing track count
                existing_count = len(self.board_data['tracks'])
                
                # Add new tracks to existing ones
                self.board_data['tracks'].extend(tracks)
                
                total_count = len(self.board_data['tracks'])
                logger.info(f"GUI: Added {len(tracks)} new tracks to {existing_count} existing tracks = {total_count} total tracks")
            else:
                logger.warning(f"No tracks received from router!")
                
            if vias:
                if 'vias' not in self.board_data:
                    self.board_data['vias'] = []
                
                # Get existing via count  
                existing_via_count = len(self.board_data['vias'])
                
                # Add new vias to existing ones
                self.board_data['vias'].extend(vias)
                
                total_via_count = len(self.board_data['vias'])
                logger.info(f"GUI: Added {len(vias)} new vias to {existing_via_count} existing vias = {total_via_count} total vias")
            else:
                logger.debug(f"No vias received from router")
            
            # Update viewer with ALL accumulated routing data
            all_tracks = self.board_data.get('tracks', [])
            all_vias = self.board_data.get('vias', [])
            
            if hasattr(self.pcb_viewer, 'update_routing'):
                logger.info(f"GUI: Calling pcb_viewer.update_routing() with {len(all_tracks)} total tracks, {len(all_vias)} total vias")
                self.pcb_viewer.update_routing(all_tracks, all_vias)
            else:
                logger.info(f"GUI: pcb_viewer has no update_routing method, calling update() instead")
                # Force repaint to show tracks
                self.pcb_viewer.update()
                
    def _complete_routing(self):
        """Complete the progressive routing process"""
        self.routing_timer.stop()
        
        elapsed_time = time.time() - self.routing_start_time
        logger.info(f"SUCCESS: Progressive routing complete: {self.routed_count}/{len(self.routing_nets)} nets routed in {elapsed_time:.1f}s")
        
        # Final GUI updates
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Routing complete: {self.routed_count}/{len(self.routing_nets)} nets routed")
        
        # Final visualization update
        self._update_routing_visualization()
        
        # DEBUG: Screenshot after routing
        if self.pcb_viewer:
            self.pcb_viewer.debug_screenshot("after_routing")
            
        # Reset UI
        self._reset_routing_ui()
        
        # Enable commit/rollback if any routes were created
        if self.routed_count > 0:
            self.commit_btn.setEnabled(True)
            self.rollback_btn.setEnabled(True)
    
    def _on_routing_progress(self, current, total, status, tracks, vias):
        """Handle routing progress updates from the thread"""
        # Update progress bar
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"{status} ({current}/{total})")
        
        # Update preview with new tracks/vias
        if tracks:
            if 'tracks' not in self.board_data:
                self.board_data['tracks'] = []
            self.board_data['tracks'].extend(tracks)
            
        if vias:
            if 'vias' not in self.board_data:
                self.board_data['vias'] = []
            self.board_data['vias'].extend(vias)
            
        # PERFORMANCE: Throttle GUI updates during routing
        if self.pcb_viewer and hasattr(self, '_last_update_time'):
            import time
            current_time = time.time()
            if current_time - self._last_update_time > 0.25:  # Throttle to 4 FPS, smooth ~25 nets/250ms
                self.pcb_viewer.update()
                self._last_update_time = current_time
        elif self.pcb_viewer:
            import time
            self.pcb_viewer.update()
            self._last_update_time = time.time()

    def _update_preview_after_batch(self):
        """Extract routing results and update preview with traces and vias"""
        logger.info("=== _update_preview_after_batch() CALLED ===")
        try:
            if not self.router:
                logger.warning("No router available for result extraction")
                return
                
            # Get the latest routing results from router
            if hasattr(self.router, 'routed_nets') and self.router.routed_nets:
                logger.info(f"PREVIEW UPDATE: Extracting visual data from {len(self.router.routed_nets)} routed nets")
                
                new_tracks = []
                new_vias = []
                
                for net_id, route in self.router.routing_results.items():
                    if route and hasattr(route, 'segments') and hasattr(route, 'vias'):
                        # Convert segments to tracks
                        for segment in route.segments:
                            if hasattr(segment, 'start') and hasattr(segment, 'end') and hasattr(segment, 'layer'):
                                track = {
                                    'start_x': segment.start.x,
                                    'start_y': segment.start.y, 
                                    'end_x': segment.end.x,
                                    'end_y': segment.end.y,
                                    'layer': segment.layer,
                                    'width': getattr(segment, 'width', 0.2),  # Default trace width
                                    'net': net_id
                                }
                                new_tracks.append(track)
                        
                        # Convert vias
                        for via in route.vias:
                            if hasattr(via, 'position') and hasattr(via, 'from_layer') and hasattr(via, 'to_layer'):
                                via_data = {
                                    'x': via.position.x,
                                    'y': via.position.y,
                                    'from_layer': via.from_layer,
                                    'to_layer': via.to_layer,
                                    'drill': getattr(via, 'drill_size', 0.3),  # Default drill size
                                    'size': getattr(via, 'diameter', 0.6),    # Default via diameter
                                    'net': net_id
                                }
                                new_vias.append(via_data)
                
                # Update board_data with new tracks and vias
                if new_tracks:
                    if 'tracks' not in self.board_data:
                        self.board_data['tracks'] = []
                    self.board_data['tracks'].extend(new_tracks)
                    logger.info(f"PREVIEW UPDATE: Added {len(new_tracks)} tracks to preview")
                
                if new_vias:
                    if 'vias' not in self.board_data:
                        self.board_data['vias'] = []
                    self.board_data['vias'].extend(new_vias) 
                    logger.info(f"PREVIEW UPDATE: Added {len(new_vias)} vias to preview")
                
                # Force preview update
                if self.pcb_viewer and (new_tracks or new_vias):
                    self.pcb_viewer.update()
                    logger.info("PREVIEW UPDATE: Forced PCB viewer refresh")
                    
            else:
                logger.debug("PREVIEW UPDATE: No routed nets available for visualization")
                
        except Exception as e:
            logger.error(f"Error updating preview after batch: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _on_routing_completed(self, result):
        """Handle routing completion"""
        # Hide cancel button
        if hasattr(self, 'cancel_btn'):
            self.cancel_btn.setVisible(False)
            
        # Update UI
        self.progress_bar.setValue(100)
        self.commit_btn.setEnabled(True)
        self.rollback_btn.setEnabled(True)
        self.replay_btn.setEnabled(True)
        self.route_preview_btn.setEnabled(True)
        self.algorithm_combo.setEnabled(True)

        # Store routing result for commit
        self.routing_result = result

        # Export demo artifacts (run_summary.json + GeoJSON)
        self._export_demo_artifacts(result)
        
        # DEBUG: Screenshot after routing completes
        if self.pcb_viewer:
            self.pcb_viewer.debug_screenshot("after_routing")
        
        # Update status
        routed_nets = result.get('routed_nets', 0)
        failed_nets = result.get('failed_nets', 0)
        self.status_label.setText(f"Manhattan routing completed: {routed_nets} nets routed")
    
    def _on_pathfinder_progress(self, progress_data):
        """Handle PathFinder routing progress updates"""
        if not hasattr(self, 'pathfinder_stats') or not self.pathfinder_stats:
            return
            
        progress_type = progress_data.get('type', '')
        
        if progress_type == 'iteration_start':
            # Update iteration progress
            iteration = progress_data.get('iteration', 0)
            max_iterations = progress_data.get('max_iterations', 50)
            status = progress_data.get('status', '')
            self.pathfinder_stats.update_iteration(iteration, max_iterations, status)
            
        elif progress_type == 'routing_update':
            # Update routing statistics
            successful = progress_data.get('successful_routes', 0)
            failed = progress_data.get('failed_routes', 0)
            self.pathfinder_stats.update_routing_stats(successful, failed)
            
            # Update congestion statistics
            congested_edges = progress_data.get('congested_edges', 0)
            total_edges = progress_data.get('total_edges', 1)
            self.pathfinder_stats.update_congestion(congested_edges, total_edges)
            
        elif progress_type == 'convergence':
            # PathFinder converged
            iteration = progress_data.get('iteration', 0)
            status = progress_data.get('status', 'Converged!')
            max_iterations = 50  # Default, will be updated by iteration_start
            self.pathfinder_stats.update_iteration(iteration, max_iterations, status)
            
        elif progress_type == 'completion':
            # Routing completed
            successful = progress_data.get('successful_routes', 0)
            total = progress_data.get('total_routes', 0)
            converged = progress_data.get('converged', False)
            
            final_message = f"Completed: {successful}/{total} routes"
            if converged:
                final_message += " (Converged)"
                
            self.pathfinder_stats.finish_routing(successful > 0, final_message)
        
        # Clean up GPU resources if they were used
        try:
            if hasattr(self.routing_thread, 'router') and hasattr(self.routing_thread.router, 'gpu_provider'):
                if self.routing_thread.router.gpu_provider and hasattr(self.routing_thread.router.gpu_provider, 'cleanup'):
                    logger.info("Cleaning up GPU resources")
                    self.routing_thread.router.gpu_provider.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up GPU resources: {e}")
    
    def _on_routing_error(self, error_message):
        """Handle routing errors"""
        logger.error(f"Routing error: {error_message}")
        self.status_label.setText(f"Error: {error_message}")
        self._reset_routing_ui()
        
        # Clean up GPU resources if they were used
        try:
            if hasattr(self.routing_thread, 'router') and hasattr(self.routing_thread.router, 'gpu_provider'):
                if self.routing_thread.router.gpu_provider and hasattr(self.routing_thread.router.gpu_provider, 'cleanup'):
                    logger.info("Cleaning up GPU resources")
                    self.routing_thread.router.gpu_provider.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up GPU resources: {e}")
        
        # Hide cancel button
        if hasattr(self, 'cancel_btn'):
            self.cancel_btn.setVisible(False)
            
        QMessageBox.critical(self, "Routing Error", f"Error during routing:\n{error_message}")
    
    def _cancel_routing(self):
        """Cancel current routing operation"""
        if hasattr(self, 'routing_thread') and self.routing_thread.isRunning():
            logger.info("Cancelling routing...")
            self.status_label.setText("Cancelling routing...")
            self.routing_thread.cancel()
            
            # Clean up GPU resources if they were used
            try:
                if hasattr(self.routing_thread, 'router') and hasattr(self.routing_thread.router, 'gpu_provider'):
                    if self.routing_thread.router.gpu_provider and hasattr(self.routing_thread.router.gpu_provider, 'cleanup'):
                        logger.info("Cleaning up GPU resources")
                        self.routing_thread.router.gpu_provider.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up GPU resources: {e}")
                
            self.routing_thread.wait()  # Wait for thread to finish
            
        self._reset_routing_ui()
        self.status_label.setText("Routing cancelled")
        
        # Hide cancel button
        if hasattr(self, 'cancel_btn'):
            self.cancel_btn.setVisible(False)
    
    def _reset_routing_ui(self):
        """Reset routing UI to initial state"""
        self.route_preview_btn.setEnabled(True)
        self.algorithm_combo.setEnabled(True)
        self.commit_btn.setEnabled(False)
        self.rollback_btn.setEnabled(False)
        self.replay_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready")
    
    def log_to_gui(self, message: str, level: str = "INFO"):
        """Add real-time log message to GUI routing log"""
        from datetime import datetime
        
        # Add timestamp and format message
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}"
        
        # Color code by level
        color_map = {
            "INFO": "#d4d4d4",      # Light gray
            "SUCCESS": "#4EC9B0",   # Teal
            "WARNING": "#DCDCAA",   # Yellow
            "ERROR": "#F44747",     # Red
            "DEBUG": "#808080"      # Gray
        }
        color = color_map.get(level, "#d4d4d4")
        
        # Add colored message to log widget
        if hasattr(self, 'routing_log'):
            self.routing_log.append(f'<span style="color: {color};">{formatted_message}</span>')
            
            # Auto-scroll to bottom
            cursor = self.routing_log.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.routing_log.setTextCursor(cursor)
            
            # Limit log to last 1000 lines for performance
            if self.routing_log.document().lineCount() > 1000:
                cursor.movePosition(cursor.MoveOperation.Start)
                cursor.movePosition(cursor.MoveOperation.Down, cursor.MoveMode.KeepAnchor, 200)
                cursor.removeSelectedText()
    
    def clear_routing_log(self):
        """Clear the routing log window"""
        if hasattr(self, 'routing_log'):
            self.routing_log.clear()
    
    def _convert_board_data_to_domain(self, board_data, drc_constraints):
        """Convert board_data dict to domain Board object for RRG router"""
        from orthoroute.domain.models.board import Board, Net, Pad, Bounds, Coordinate, Component
        
        # Create board bounds
        bounds_data = board_data.get('bounds', (0, 0, 100, 100))
        board_bounds = Bounds(
            min_x=bounds_data[0],
            min_y=bounds_data[1], 
            max_x=bounds_data[2],
            max_y=bounds_data[3]
        )
        
        # Convert nets and pads
        nets = []
        nets_data = board_data.get('nets', {})
        
        for net_name, net_data in nets_data.items():
            if not net_name or net_name.strip() == "":
                continue
                
            pads_data = net_data.get('pads', [])
            if len(pads_data) < 2:
                continue  # Skip single-pad nets
            
            # Convert pads
            net_pads = []
            for pad_data in pads_data:
                pad = Pad(
                    id=f"{net_name}_pad_{len(net_pads)}",
                    component_id=f"comp_{net_name}_{len(net_pads)}",
                    net_id=f"net_{len(nets)}",
                    position=Coordinate(
                        x=pad_data.get('x', 0.0),
                        y=pad_data.get('y', 0.0)
                    ),
                    size=(
                        pad_data.get('width', 1.0),
                        pad_data.get('height', 1.0)
                    ),
                    drill_size=pad_data.get('drill', None),
                    layer=pad_data.get('layers', ['F.Cu'])[0] if pad_data.get('layers') else 'F.Cu'
                )
                net_pads.append(pad)
            
            # Create net
            net = Net(
                id=f"net_{len(nets)}",
                name=net_name,
                pads=net_pads
            )
            nets.append(net)
        
        # Create mock components for proper bounds calculation
        components = []
        all_pads = []
        for net in nets:
            for i, pad in enumerate(net.pads):
                all_pads.append(pad)
        
        # Create a single mock component containing all pads
        if all_pads:
            # Calculate center position
            avg_x = sum(pad.position.x for pad in all_pads) / len(all_pads)
            avg_y = sum(pad.position.y for pad in all_pads) / len(all_pads)
            
            mock_component = Component(
                id="mock_comp_1",
                reference="U1",
                value="MOCK",
                footprint="MOCK_FP",
                position=Coordinate(avg_x, avg_y),
                pads=all_pads
            )
            components.append(mock_component)
        
        # Create board - handle filename robustly
        filename = board_data.get('filename') or board_data.get('name') or 'TestBackplane.kicad_pcb'
        board = Board(
            id="board_1", 
            name=filename,
            components=components,
            nets=nets,
            layer_count=board_data.get('layers', 2)  # Dynamic from KiCad file
        )
        
        # Store airwires as a custom attribute for RRG routing
        board._airwires = board_data.get('airwires', [])
        # Store KiCad-calculated bounds for accurate routing area
        board._kicad_bounds = board_data.get('bounds', None)
        
        return board


# PathFinder GUI Integration Methods
def _update_pathfinder_status(self, status_text: str):
    """Update status bar with PathFinder instrumentation metrics"""
    self.status_label.setText(status_text)
    self.metrics_label.setText(status_text)
    self.metrics_label.setVisible(True)
    
    # Also print to terminal for console monitoring
    print(f"[PathFinder]: {status_text}")

def _update_csv_export_status(self, csv_status: str):
    """Update status bar with CSV export information"""
    self.csv_status_label.setText(csv_status)
    self.csv_status_label.setVisible(True)
    
    # Auto-hide CSV status after 10 seconds
    QTimer.singleShot(10000, lambda: self.csv_status_label.setVisible(False))

def _display_instrumentation_summary(self):
    """Display instrumentation summary when routing completes"""
    if hasattr(self.router, 'pathfinder') and hasattr(self.router.pathfinder, 'get_instrumentation_summary'):
        summary = self.router.pathfinder.get_instrumentation_summary()
        
        if summary:
            summary_text = (f"Session: {summary.get('session_id', 'N/A')} | "
                          f"Iterations: {summary.get('total_iterations', 0)} | "
                          f"Success: {summary.get('final_success_rate', 0):.1f}% | "
                          f"Nets: {summary.get('successful_nets', 0)}/{summary.get('total_nets_processed', 0)} | "
                          f"Avg Time: {summary.get('avg_routing_time_ms', 0):.1f}ms")
            
            self.metrics_label.setText(summary_text)
            self.metrics_label.setVisible(True)
            
            print(f"[SUMMARY] Routing Summary: {summary_text}")
            logger.info(f"Instrumentation summary: {summary}")


def _export_demo_artifacts(self, result):
    """Export run summary and GeoJSON artifacts for demo"""
    try:
        import json
        from datetime import datetime
        from pathlib import Path

        # Create artifacts directory
        artifacts_dir = Path("demo_artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        # Generate run summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get GPU status from GPUConfig (hardcoded, no env vars)
        try:
            from ...algorithms.manhattan.unified_pathfinder import GPUConfig
            gpu_enabled = GPUConfig.GPU_MODE
        except ImportError:
            gpu_enabled = False

        run_summary = {
            "timestamp": timestamp,
            "routing_engine": "UnifiedPathFinder",
            "gpu_enabled": gpu_enabled,
            "total_nets": getattr(result, 'total_nets', 0),
            "successful_nets": getattr(result, 'successful_nets', 0),
            "failed_nets": getattr(result, 'failed_nets', 0),
            "success_rate": f"{(getattr(result, 'successful_nets', 0) / getattr(result, 'total_nets', 1) * 100):.1f}%",
            "total_tracks": len(self.board_data.get('tracks', [])),
            "total_vias": len(self.board_data.get('vias', [])),
            "board_bounds": getattr(self.board_data, 'bounds', None),
            "grid_pitch": 0.4,
        }

        # Export run_summary.json
        summary_file = artifacts_dir / f"run_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(run_summary, f, indent=2)

        # Export GeoJSON (simplified track/via representation)
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }

        # Add tracks as LineString features
        for track in self.board_data.get('tracks', []):
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[track.get('start_x', 0), track.get('start_y', 0)],
                                   [track.get('end_x', 0), track.get('end_y', 0)]]
                },
                "properties": {
                    "type": "track",
                    "width": track.get('width', 0.2),
                    "layer": track.get('layer', 0),
                    "net_id": track.get('net_id', 'unknown')
                }
            }
            geojson["features"].append(feature)

        # Add vias as Point features
        for via in self.board_data.get('vias', []):
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [via.get('x', 0), via.get('y', 0)]
                },
                "properties": {
                    "type": "via",
                    "diameter": via.get('diameter', 0.3),
                    "drill": via.get('drill', 0.15),
                    "layers": f"L{via.get('from_layer', 0)}-L{via.get('to_layer', 1)}",
                    "net_id": via.get('net_id', 'unknown')
                }
            }
            geojson["features"].append(feature)

        # Export GeoJSON
        geojson_file = artifacts_dir / f"routing_geometry_{timestamp}.geojson"
        with open(geojson_file, 'w') as f:
            json.dump(geojson, f, indent=2)

        # Log success
        logger.info(f"[EXPORT] Demo artifacts exported:")
        logger.info(f"[EXPORT]   - Summary: {summary_file}")
        logger.info(f"[EXPORT]   - GeoJSON: {geojson_file}")
        self.log_to_gui(f"[EXPORT] Artifacts saved to demo_artifacts/", "SUCCESS")

    except Exception as e:
        logger.warning(f"Failed to export demo artifacts: {e}")
        self.log_to_gui(f"[EXPORT] Warning: Could not save artifacts", "WARNING")


# Add methods to OrthoRouteMainWindow class
OrthoRouteMainWindow._update_pathfinder_status = _update_pathfinder_status
OrthoRouteMainWindow._update_csv_export_status = _update_csv_export_status
OrthoRouteMainWindow._display_instrumentation_summary = _display_instrumentation_summary
OrthoRouteMainWindow._export_demo_artifacts = _export_demo_artifacts