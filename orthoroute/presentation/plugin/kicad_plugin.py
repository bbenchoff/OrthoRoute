"""KiCad plugin entry point for OrthoRoute."""
import sys
import os
import logging
import logging.handlers
from typing import Optional

# Add the package to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(os.path.dirname(current_dir))
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

from ...application.services.routing_orchestrator import RoutingOrchestrator
from ...infrastructure.kicad.ipc_adapter import KiCadIPCAdapter
from ...infrastructure.kicad.swig_adapter import KiCadSWIGAdapter
from ...infrastructure.gpu.cuda_provider import CUDAProvider
from ...infrastructure.gpu.cpu_fallback import CPUProvider

logger = logging.getLogger(__name__)


class KiCadPlugin:
    """Main KiCad plugin class for OrthoRoute."""
    
    def __init__(self):
        """Initialize the KiCad plugin."""
        # Setup logging
        self._setup_logging()
        
        # Initialize routing engine attributes
        self.routing_engine = None          # <- WIRING FIX: add this
        self.graph_state = None             # <- add this for Steps 7&8
        
        # Initialize services
        self._setup_services()
        
        logger.info("OrthoRoute KiCad Plugin initialized")
    
    def _setup_logging(self):
        """Setup logging configuration for plugin."""
        # NOTE: Logging already configured by init_logging() in main.py
        # init_logging() sets up: DEBUG→file (logs/), WARNING→console
        # DO NOT modify handler levels here to prevent console spam
        pass
    
    def _setup_services(self):
        """Initialize plugin services."""
        # Setup GPU provider
        try:
            self.gpu_provider = CUDAProvider()
            logger.info("CUDA GPU provider initialized for plugin")
        except Exception as e:
            logger.warning(f"CUDA not available for plugin, using CPU fallback: {e}")
            self.gpu_provider = CPUProvider()
        
        # Setup adapters - try all available options
        self.kicad_adapters = []
        
        # Try IPC adapter
        try:
            ipc_adapter = KiCadIPCAdapter()
            self.kicad_adapters.append(('IPC', ipc_adapter))
            logger.info("IPC adapter available")
        except Exception as e:
            logger.warning(f"IPC adapter unavailable: {e}")
        
        # Try SWIG adapter  
        try:
            swig_adapter = KiCadSWIGAdapter()
            self.kicad_adapters.append(('SWIG', swig_adapter))
            logger.info("SWIG adapter available")
        except Exception as e:
            logger.warning(f"SWIG adapter unavailable: {e}")
        
        # Always have file parser as fallback
        from ...infrastructure.kicad.file_parser import KiCadFileParser
        file_adapter = KiCadFileParser()
        self.kicad_adapters.append(('File', file_adapter))
        logger.info("File parser available")
        
        # Use first adapter as default (will try others in run() method)
        self.kicad_adapter = self.kicad_adapters[0][1] if self.kicad_adapters else file_adapter
        
        # Setup repositories and services
        from ...infrastructure.persistence.memory_board_repository import MemoryBoardRepository
        from ...infrastructure.persistence.memory_routing_repository import MemoryRoutingRepository
        from ...infrastructure.persistence.event_bus import EventBus
        from ...domain.models.constraints import DRCConstraints
        
        # STEP 6: Gate router selection - ensure unified_pathfinder is used by default
        PREFERRED_ROUTER = "unified_pathfinder"  # "unified_pathfinder" or "manhattan_router_rrg"
        
        logger.info(f"[ROUTER SELECTION] Selected router: {PREFERRED_ROUTER}")

        if PREFERRED_ROUTER == "unified_pathfinder":
            try:
                from ...algorithms.manhattan.unified_pathfinder import UnifiedPathFinder, PathFinderConfig
                logger.info("[ROUTER SELECTION] Loading UnifiedPathFinder (improved coordinate handling)")
                config = PathFinderConfig()
                config.strict_drc = True  # Force strict DRC for acceptance testing
                self.pf = UnifiedPathFinder(config=config, use_gpu=True)  # Enable GPU acceleration

                # Assert strict DRC is enabled
                assert getattr(self.pf.config, "strict_drc", False), "strict_drc must be True for acceptance."
                self.pf_tag = self.pf._instance_tag  # For logging
                self.routing_engine = self.pf  # Keep compatibility with existing code

                # Log router selection with instance tag
                logger.info(f"[ROUTER SELECTION] Selected router: unified_pathfinder (instance={self.pf_tag})")
                self.routing_engine_type = "unified_pathfinder"

                logger.info(f"[PIPELINE] UnifiedPathFinder created with instance_tag={self.pf_tag}")
            except Exception as e:
                logger.warning(f"[ROUTER SELECTION] Failed to load UnifiedPathFinder: {e}")
                logger.info("[ROUTER SELECTION] Falling back to ManhattanRRGRoutingEngine")
                PREFERRED_ROUTER = "manhattan_router_rrg"
        
        if PREFERRED_ROUTER == "manhattan_router_rrg":
            # DISABLE RRG during bring-up to avoid split codepaths masking bugs
            raise RuntimeError("RRG disabled during bring-up. Use unified_pathfinder.")
        
        # Initialize repositories and services
        self.board_repository = MemoryBoardRepository()
        self.routing_repository = MemoryRoutingRepository()
        self.event_bus = EventBus()
        
        logger.info(f"[ROUTER SELECTION] Active routing engine: {self.routing_engine_type}")
        
        # Setup routing orchestrator (only if we have a routing engine)
        if hasattr(self, 'routing_engine') and self.routing_engine:
            self.routing_orchestrator = RoutingOrchestrator(
                routing_engine=self.routing_engine,
                board_repository=self.board_repository,
                routing_repository=self.routing_repository,
                event_publisher=self.event_bus
            )
        else:
            self.routing_orchestrator = None

    def get_pathfinder(self):
        """Get the single pathfinder instance for GUI use."""
        return self.pf

    def route_all(self, nets):
        """GUI ROUTING: Use same three function calls as CLI."""
        pf = self.routing_engine
        pf_tag = getattr(pf, '_instance_tag', 'NO_TAG')
        logger.info(f"[GUI-STEP1] Using same UnifiedPathFinder instance {pf_tag} for GUI routing")
        
        # Create minimal board object with nets and correct layer count
        from orthoroute.domain.models.board import Board
        layer_count = getattr(self, 'layer_count', 6)  # Use detected layer count, fallback to 6
        logger.info(f"[BOARD DEBUG] About to create Board with layer_count={layer_count}")
        board = Board(id="gui-board", name="GUI Board", layer_count=layer_count)
        board.nets = nets
        logger.info(f"[BOARD DEBUG] Created Board domain object: board.layer_count={board.layer_count}")
        
        # SAME THREE CALLS AS CLI
        logger.info(f"[GUI-STEP1-CALL1] pf.initialize_graph(board) with instance {pf_tag}")
        pf.initialize_graph(board)
        
        logger.info(f"[GUI-STEP1-CALL2] pf.map_all_pads(board) with instance {pf_tag}")
        pf.map_all_pads(board)
        
        logger.info(f"[GUI-STEP1-CALL3] pf.route_multiple_nets(board.nets) with instance {pf_tag}")
        return pf.route_multiple_nets(board.nets)
    
    def run(self, board_file: str = None):
        """Main plugin execution method - ONE INSTANCE END-TO-END."""
        try:
            logger.info("Starting OrthoRoute plugin execution")
            
            # Log the instance tag at entry point
            pf = self.routing_engine
            pf_tag = getattr(pf, '_instance_tag', 'NO_TAG')
            logger.info(f"[STEP1] Plugin using UnifiedPathFinder instance: {pf_tag}")
            
            # ALWAYS try to load from KiCad first, regardless of connection status
            board = None
            
            # Try loading from KiCad APIs first (try all adapters in priority order)
            if not board_file:
                logger.info("Attempting to load board from KiCad")
                
                for adapter_name, adapter in self.kicad_adapters:
                    if adapter_name == 'File':
                        continue  # Skip file adapter for KiCad loading
                    
                    try:
                        logger.info(f"Trying {adapter_name} adapter...")
                        
                        # Try to connect and load
                        if hasattr(adapter, 'connect'):
                            if not adapter.connect():
                                logger.warning(f"{adapter_name} adapter could not connect")
                                continue
                        
                        board = adapter.load_board()
                        if board:
                            logger.info(f"Successfully loaded board from KiCad via {adapter_name}: {board.name}")
                            break
                        else:
                            logger.warning(f"{adapter_name} adapter connected but no board data")
                            
                    except Exception as e:
                        logger.warning(f"Could not load from {adapter_name} adapter: {e}")
                        continue
            
            # Fall back to file loading if KiCad loading failed or file was specified
            if not board:
                if not board_file:
                    # Find a test board file for testing
                    import glob
                    test_boards = glob.glob("testboards/**/*.kicad_pcb", recursive=True)
                    if test_boards:
                        board_file = test_boards[0]
                        logger.info(f"Falling back to test board: {board_file}")
                    else:
                        logger.error("No KiCad board available and no test boards found")
                        return False
                
                logger.info(f"Loading board from file: {board_file}")
                # Use file parser specifically for file loading
                file_adapter = None
                for adapter_name, adapter in self.kicad_adapters:
                    if adapter_name == 'File':
                        file_adapter = adapter
                        break
                
                if file_adapter:
                    board = file_adapter.load_board(board_file)
                else:
                    logger.error("File adapter not available")
                    return False
            
            if not board:
                logger.error("Failed to load board from any source")
                return False
            
            logger.info(f"Loaded board: {board.name} with {len(board.nets)} nets")
            
            # Store board in repository
            self.board_repository.save_board(board)
            
            # STEP 1: ONE INSTANCE, THREE FUNCTION CALLS
            logger.info(f"[STEP1] Using single UnifiedPathFinder instance {pf_tag} for end-to-end routing")
            
            # Call 1: Initialize graph with board data
            logger.info(f"[STEP1-CALL1] pf.initialize_graph(board) with instance {pf_tag}")
            pf.initialize_graph(board)
            
            # Call 2: Map all pads to lattice indices
            logger.info(f"[STEP1-CALL2] pf.map_all_pads(board) with instance {pf_tag}")
            pf.map_all_pads(board)
            
            # Call 3: Route all nets and get results
            logger.info(f"[STEP1-CALL3] pf.route_multiple_nets(board.nets) with instance {pf_tag}")
            results = pf.route_multiple_nets(board.nets)
            
            if results:
                logger.info(f"[STEP1] End-to-end routing completed successfully with instance {pf_tag}")
                return True
            else:
                logger.error(f"[STEP1] End-to-end routing failed with instance {pf_tag} - no results returned")
                return False
                
        except Exception as e:
            logger.error(f"[STEP1] Plugin execution failed: {e}")
            return False
    
    def _apply_routes_to_kicad(self, results) -> bool:
        """Apply routing results to the KiCad board."""
        try:
            # Get routing results from repository
            routing_results = self.routing_repository.get_all_routes()
            
            if not routing_results:
                logger.warning("No routes to apply")
                return True
            
            # Apply each route
            for route in routing_results:
                try:
                    # Create tracks for route segments
                    for segment in route.segments:
                        self.kicad_adapter.create_track(
                            start_x=segment.start.x,
                            start_y=segment.start.y,
                            end_x=segment.end.x,
                            end_y=segment.end.y,
                            layer=segment.layer,
                            width=segment.width,
                            net_id=route.net_id
                        )
                    
                    # Create vias
                    for via in route.vias:
                        self.kicad_adapter.create_via(
                            x=via.position.x,
                            y=via.position.y,
                            size=via.size,
                            drill=via.drill,
                            layers=(via.start_layer, via.end_layer),
                            net_id=route.net_id
                        )
                
                except Exception as e:
                    logger.error(f"Failed to apply route for net {route.net_id}: {e}")
                    continue
            
            # Refresh KiCad display
            self.kicad_adapter.refresh_display()
            
            logger.info(f"Applied {len(routing_results)} routes to KiCad board")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply routes to KiCad: {e}")
            return False
    
    def show_progress(self, message: str):
        """Show progress message (plugin mode doesn't show GUI)."""
        logger.info(f"Progress: {message}")
    
    def run_with_gui(self):
        """Run plugin with full interactive GUI using new architecture components."""
        try:
            logger.info("Loading board from KiCad using rich interface for GUI")
            
            from PyQt6.QtWidgets import QApplication, QMessageBox
            import sys
            
            # Create Qt application
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
                app.setApplicationName("OrthoRoute")
                app.setApplicationVersion("1.0.0")
                app.setOrganizationName("OrthoRoute")
            
            # Connect to KiCad and get board data using the rich interface
            logger.info("Connecting to KiCad to get rich board data...")
            
            try:
                from ...infrastructure.kicad.rich_kicad_interface import RichKiCadInterface
                kicad_interface = RichKiCadInterface()
                
                # Connect to running KiCad instance
                if not kicad_interface.connect():
                    QMessageBox.critical(None, "KiCad Connection Error", 
                                       "Could not connect to KiCad via IPC API.\n\n"
                                       "Make sure KiCad is running and has a PCB file open,\n"
                                       "and that the IPC API is enabled in KiCad preferences.")
                    logger.error("Failed to connect to KiCad")
                    return False
                
                logger.info("Connected to KiCad via rich IPC API")
                
                # Get rich board data from the currently open PCB
                board_data = kicad_interface.get_board_data()
                
                if not board_data or len(board_data.get('pads', [])) == 0:
                    QMessageBox.critical(None, "No Board Data", 
                                       "No valid board data found.\n\n"
                                       "Make sure you have a PCB file open in KiCad\n"
                                       "with components and pads.")
                    logger.error("No valid board data found")
                    return False
                
                logger.info(f"Loaded rich board data from KiCad: {len(board_data.get('pads', []))} pads, {len(board_data.get('nets', {}))} nets")

                # Store layer count for Board domain object creation
                layers_data = board_data.get('layers', 6)
                if isinstance(layers_data, int):
                    layer_count = layers_data  # layers is stored as count
                elif isinstance(layers_data, (list, tuple)):
                    layer_count = len(layers_data)  # layers is stored as list
                else:
                    layer_count = 6  # fallback
                logger.info(f"Board layer stack: {layer_count} copper layers detected")
                self.layer_count = layer_count

                # STEP 4: Initialize UnifiedPathFinder with the REAL board data (GUI PATH FIX)
                
                # Create and show the full-featured OrthoRoute window
                from ..gui.main_window import OrthoRouteMainWindow
                window = OrthoRouteMainWindow(board_data, kicad_interface, plugin=self)
                window.show()
                
                logger.info("OrthoRoute rich GUI launched successfully!")
                
                # Run the application
                result = app.exec()
                return result == 0
                
            except Exception as e:
                logger.error(f"Failed to connect to KiCad or get board data: {e}")
                import traceback
                logger.error(traceback.format_exc())
                QMessageBox.critical(None, "KiCad Error", 
                                   f"Failed to connect to KiCad or get board data:\n{e}\n\n"
                                   f"Make sure:\n"
                                   f"• KiCad is running\n"
                                   f"• A PCB file is open\n"
                                   f"• IPC API is enabled in KiCad preferences")
                return False
            
        except Exception as e:
            logger.error(f"GUI execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Fall back to headless mode
            return self.run()
    
    def run_with_gui_autostart(self):
        """Run plugin with GUI and automatically start routing process."""
        try:
            logger.info("Loading board from KiCad using rich interface for GUI with autostart")
            
            from PyQt6.QtWidgets import QApplication, QMessageBox
            from PyQt6.QtCore import QTimer
            import sys
            
            # Create Qt application
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
                app.setApplicationName("OrthoRoute")
                app.setApplicationVersion("1.0.0")
                app.setOrganizationName("OrthoRoute")
            
            # Connect to KiCad and get board data using the rich interface
            logger.info("Connecting to KiCad to get rich board data...")
            
            try:
                from ...infrastructure.kicad.rich_kicad_interface import RichKiCadInterface
                kicad_interface = RichKiCadInterface()
                
                # Connect to running KiCad instance
                if not kicad_interface.connect():
                    QMessageBox.critical(None, "KiCad Connection Error", 
                                       "Could not connect to KiCad via IPC API.\n\n"
                                       "Make sure KiCad is running and has a PCB file open,\n"
                                       "and that the IPC API is enabled in KiCad preferences.")
                    logger.error("Failed to connect to KiCad")
                    return False
                
                logger.info("Connected to KiCad via rich IPC API")
                
                # Get rich board data from the currently open PCB
                board_data = kicad_interface.get_board_data()
                
                if not board_data or len(board_data.get('pads', [])) == 0:
                    QMessageBox.critical(None, "No Board Data", 
                                       "No valid board data found.\n\n"
                                       "Make sure you have a PCB file open in KiCad\n"
                                       "with components and pads.")
                    logger.error("No valid board data found")
                    return False
                
                logger.info(f"Loaded rich board data from KiCad: {len(board_data.get('pads', []))} pads, {len(board_data.get('nets', {}))} nets")

                # Store layer count for Board domain object creation
                layers_data = board_data.get('layers', 6)
                if isinstance(layers_data, int):
                    layer_count = layers_data  # layers is stored as count
                elif isinstance(layers_data, (list, tuple)):
                    layer_count = len(layers_data)  # layers is stored as list
                else:
                    layer_count = 6  # fallback
                logger.info(f"Board layer stack: {layer_count} copper layers detected")
                self.layer_count = layer_count

                # STEP 4: Initialize UnifiedPathFinder with the REAL board data (GUI PATH FIX)
                
                # Create and show the full-featured OrthoRoute window
                from ..gui.main_window import OrthoRouteMainWindow
                window = OrthoRouteMainWindow(board_data, kicad_interface, plugin=self)
                window.show()
                
                logger.info("OrthoRoute rich GUI launched successfully!")
                
                # Auto-start routing after 2 seconds to let GUI load
                def auto_start_routing():
                    logger.info("AUTO-START: Triggering routing process...")
                    if hasattr(window, 'begin_autorouting'):
                        window.begin_autorouting()
                        logger.info("AUTO-START: Successfully triggered begin_autorouting")
                    else:
                        logger.warning("AUTO-START: begin_autorouting method not found")
                        # Fallback to route_all_nets if available
                        if hasattr(window, 'route_all_nets'):
                            window.route_all_nets()
                        else:
                            logger.warning("AUTO-START: No routing method found")
                
                timer = QTimer()
                timer.timeout.connect(auto_start_routing)
                timer.setSingleShot(True)
                timer.start(2000)  # Start routing after 2 seconds
                
                # Run the application
                result = app.exec()
                return result == 0
                
            except Exception as e:
                logger.error(f"Failed to connect to KiCad or get board data: {e}")
                import traceback
                logger.error(traceback.format_exc())
                QMessageBox.critical(None, "KiCad Error", 
                                   f"Failed to connect to KiCad or get board data:\n{e}\n\n"
                                   f"Make sure:\n"
                                   f"• KiCad is running\n"
                                   f"• A PCB file is open\n"
                                   f"• IPC API is enabled in KiCad preferences")
                return False
            
        except Exception as e:
            logger.error(f"GUI execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Fall back to headless mode
            return self.run()
    
    def _create_full_gui(self):
        """Create the full interactive GUI window."""
        try:
            from PyQt6.QtWidgets import (
                QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                QGroupBox, QPushButton, QLabel, QTreeWidget, QTreeWidgetItem,
                QTextEdit, QProgressBar, QTabWidget
            )
            from PyQt6.QtCore import Qt, QTimer
            from PyQt6.QtGui import QPainter, QPen, QBrush, QColor
            
            class OrthoRouteMainWindow(QMainWindow):
                def __init__(self, plugin):
                    super().__init__()
                    self.plugin = plugin
                    self.board = None
                    self.setup_ui()
                    self.load_board_data()
                
                def setup_ui(self):
                    self.setWindowTitle("OrthoRoute - PCB Autorouter")
                    self.setMinimumSize(1200, 800)
                    self.resize(1600, 1000)
                    
                    # Create main widget
                    main_widget = QWidget()
                    self.setCentralWidget(main_widget)
                    
                    # Create main layout
                    main_layout = QHBoxLayout(main_widget)
                    
                    # Create splitter
                    splitter = QSplitter(Qt.Orientation.Horizontal)
                    main_layout.addWidget(splitter)
                    
                    # Create left panel (controls)
                    left_panel = self.create_left_panel()
                    splitter.addWidget(left_panel)
                    
                    # Create right panel (board view)
                    right_panel = self.create_right_panel()
                    splitter.addWidget(right_panel)
                    
                    # Set splitter proportions
                    splitter.setSizes([400, 1000])
                
                def create_left_panel(self):
                    widget = QWidget()
                    widget.setMaximumWidth(450)
                    layout = QVBoxLayout(widget)
                    
                    # Board Info Group
                    board_group = QGroupBox("Board Information")
                    board_layout = QVBoxLayout(board_group)
                    
                    self.board_info_label = QLabel("No board loaded")
                    board_layout.addWidget(self.board_info_label)
                    
                    layout.addWidget(board_group)
                    
                    # Routing Controls Group
                    routing_group = QGroupBox("Routing Controls")
                    routing_layout = QVBoxLayout(routing_group)
                    
                    self.route_all_btn = QPushButton("Route All Nets")
                    self.route_all_btn.clicked.connect(self.route_all_nets)
                    routing_layout.addWidget(self.route_all_btn)
                    
                    self.clear_routes_btn = QPushButton("Clear Routes")
                    self.clear_routes_btn.clicked.connect(self.clear_routes)
                    routing_layout.addWidget(self.clear_routes_btn)
                    
                    self.progress_bar = QProgressBar()
                    routing_layout.addWidget(self.progress_bar)
                    
                    layout.addWidget(routing_group)
                    
                    # Nets List Group
                    nets_group = QGroupBox("Nets")
                    nets_layout = QVBoxLayout(nets_group)
                    
                    self.nets_tree = QTreeWidget()
                    self.nets_tree.setHeaderLabels(["Net Name", "Pads", "Status"])
                    nets_layout.addWidget(self.nets_tree)
                    
                    layout.addWidget(nets_group)
                    
                    # Statistics Group
                    stats_group = QGroupBox("Statistics")
                    stats_layout = QVBoxLayout(stats_group)
                    
                    self.stats_text = QTextEdit()
                    self.stats_text.setMaximumHeight(150)
                    self.stats_text.setReadOnly(True)
                    stats_layout.addWidget(self.stats_text)
                    
                    layout.addWidget(stats_group)
                    
                    return widget
                
                def create_right_panel(self):
                    widget = QWidget()
                    layout = QVBoxLayout(widget)
                    
                    # Board Viewer
                    self.board_viewer = BoardViewer()
                    layout.addWidget(self.board_viewer)
                    
                    return widget
                
                def load_board_data(self):
                    """Load board data from the plugin."""
                    try:
                        # Load board from KiCad
                        self.board = None
                        
                        # Try loading from KiCad APIs first
                        logger.info("Loading board from KiCad for GUI")
                        
                        for adapter_name, adapter in self.plugin.kicad_adapters:
                            if adapter_name == 'File':
                                continue
                            
                            try:
                                if hasattr(adapter, 'connect'):
                                    if not adapter.connect():
                                        continue
                                
                                self.board = adapter.load_board()
                                if self.board:
                                    logger.info(f"GUI loaded board via {adapter_name}: {self.board.name}")
                                    break
                            except Exception as e:
                                logger.warning(f"GUI: Could not load from {adapter_name}: {e}")
                                continue
                        
                        # Fallback to file loading
                        if not self.board:
                            for adapter_name, adapter in self.plugin.kicad_adapters:
                                if adapter_name == 'File':
                                    # Find test board
                                    import glob
                                    test_boards = glob.glob("testboards/**/*.kicad_pcb", recursive=True)
                                    if test_boards:
                                        self.board = adapter.load_board(test_boards[0])
                                        logger.info(f"GUI loaded test board: {self.board.name if self.board else 'Failed'}")
                                    break
                        
                        if self.board:
                            self.update_board_display()
                        else:
                            logger.error("GUI: Failed to load any board")
                            
                    except Exception as e:
                        logger.error(f"GUI: Error loading board: {e}")
                
                def update_board_display(self):
                    """Update the GUI with board information."""
                    if not self.board:
                        return
                    
                    # Update board info
                    info_text = f"Board: {self.board.name}\\n"
                    info_text += f"Layers: {len(self.board.layers)}\\n" 
                    info_text += f"Components: {len(self.board.components)}\\n"
                    info_text += f"Nets: {len(self.board.nets)}"
                    self.board_info_label.setText(info_text)
                    
                    # Update nets list
                    self.nets_tree.clear()
                    for net in self.board.nets:
                        item = QTreeWidgetItem([net.name, str(len(net.pads)), "Unrouted"])
                        self.nets_tree.addTopLevelItem(item)
                    
                    # Update board viewer
                    self.board_viewer.set_board(self.board)
                    
                    # Enable controls
                    self.route_all_btn.setEnabled(True)
                    self.clear_routes_btn.setEnabled(True)
                
                def route_all_nets(self):
                    """Start routing all nets."""
                    try:
                        self.route_all_btn.setEnabled(False)
                        self.progress_bar.setValue(0)
                        
                        logger.info("GUI: Starting routing process")
                        
                        # Use the plugin's routing functionality
                        result = self.plugin.run()
                        
                        self.progress_bar.setValue(100)
                        
                        if result:
                            self.stats_text.append("Routing completed successfully!")
                            logger.info("GUI: Routing completed")
                        else:
                            self.stats_text.append("Routing failed - check console for details")
                            logger.error("GUI: Routing failed")
                        
                        self.route_all_btn.setEnabled(True)
                        
                    except Exception as e:
                        logger.error(f"GUI: Routing error: {e}")
                        self.route_all_btn.setEnabled(True)
                
                def clear_routes(self):
                    """Clear all routes."""
                    try:
                        self.stats_text.append("Routes cleared")
                        logger.info("GUI: Routes cleared")
                    except Exception as e:
                        logger.error(f"GUI: Clear routes error: {e}")
            
            class BoardViewer(QWidget):
                """Simple board viewer widget."""
                
                def __init__(self):
                    super().__init__()
                    self.board = None
                    self.setMinimumSize(800, 600)
                    self.setStyleSheet("background-color: #2d2d2d; border: 1px solid #555;")
                
                def set_board(self, board):
                    self.board = board
                    self.update()
                
                def paintEvent(self, event):
                    painter = QPainter(self)
                    painter.fillRect(self.rect(), QColor(45, 45, 45))
                    
                    if not self.board:
                        painter.setPen(QColor(128, 128, 128))
                        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 
                                       "Board loaded - visualization in development")
                        return
                    
                    # Draw board info
                    painter.setPen(QColor(200, 200, 200))
                    y_pos = 50
                    painter.drawText(20, y_pos, f"Board: {self.board.name}")
                    y_pos += 30
                    painter.drawText(20, y_pos, f"Layers: {len(self.board.layers)}")
                    y_pos += 30
                    painter.drawText(20, y_pos, f"Components: {len(self.board.components)}")
                    y_pos += 30
                    painter.drawText(20, y_pos, f"Nets: {len(self.board.nets)}")
                    
                    # Draw simple representation
                    if self.board.components:
                        painter.setPen(QPen(QColor(100, 150, 255), 2))
                        painter.setBrush(QBrush(QColor(100, 150, 255, 100)))
                        
                        # Draw components as rectangles
                        for i, comp in enumerate(self.board.components[:10]):  # Limit to first 10
                            x = 100 + (i % 5) * 120
                            y = 150 + (i // 5) * 80
                            painter.drawRect(x, y, 100, 60)
                            painter.setPen(QColor(255, 255, 255))
                            painter.drawText(x + 5, y + 15, comp.reference)
                            painter.setPen(QPen(QColor(100, 150, 255), 2))
            
            # Create and return the main window
            return OrthoRouteMainWindow(self)
            
        except Exception as e:
            logger.error(f"Error creating full GUI: {e}")
            return None
    
    def get_version(self) -> str:
        """Get plugin version."""
        return "1.0.0"
    
    def get_description(self) -> str:
        """Get plugin description."""
        return "OrthoRoute - Advanced PCB Autorouter with Manhattan routing and GPU acceleration"


# Plugin entry points for KiCad
def main():
    """Main entry point for standalone plugin execution."""
    plugin = KiCadPlugin()
    return plugin.run()


def show_gui():
    """Entry point for GUI mode."""
    plugin = KiCadPlugin()
    plugin.show_gui()


# For KiCad Action Plugin compatibility
try:
    import pcbnew
    
    class OrthoRoutePlugin(pcbnew.ActionPlugin):
        """KiCad Action Plugin wrapper."""
        
        def defaults(self):
            """Set plugin defaults."""
            self.name = "OrthoRoute"
            self.category = "Routing"
            self.description = "Advanced PCB autorouter with Manhattan routing"
            self.show_toolbar_button = True
            self.icon_file_name = os.path.join(os.path.dirname(__file__), "icon.png")
        
        def Run(self):
            """Run the plugin."""
            plugin = KiCadPlugin()
            success = plugin.run()
            
            if success:
                pcbnew.Refresh()  # Refresh KiCad display
            
            return success
    
    # Register the plugin
    OrthoRoutePlugin().register()
    
except ImportError:
    # pcbnew not available - running outside KiCad
    logger.info("pcbnew not available - plugin can only run in standalone mode")