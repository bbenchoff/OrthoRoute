"""Configuration settings dataclasses."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class RoutingSettings:
    """Settings for routing operations."""
    algorithm: str = "manhattan"
    timeout_per_net: float = 30.0
    max_iterations: int = 50000
    use_gpu: bool = True
    parallel_routing: bool = True
    optimization_passes: int = 2
    
    # Track and via settings
    default_track_width: float = 0.2  # mm - KiCad default
    default_via_size: float = 0.8  # mm - KiCad default
    default_via_drill: float = 0.4  # mm - typical for 0.8mm via
    default_clearance: float = 0.2  # mm - KiCad default
    
    # Manhattan routing specific
    manhattan_grid_resolution: float = 0.05  # mm
    manhattan_layer_directions: Dict[str, str] = field(default_factory=lambda: {
        "In1.Cu": "horizontal",
        "In2.Cu": "vertical", 
        "In3.Cu": "horizontal",
        "In4.Cu": "vertical",
        "B.Cu": "vertical"
    })
    
    # Debug settings
    debug_single_roi: bool = False  # Force single ROI processing for debugging
    
    def validate(self) -> List[str]:
        """Validate settings and return list of errors."""
        errors = []
        
        if self.timeout_per_net <= 0:
            errors.append("timeout_per_net must be positive")
        
        if self.max_iterations <= 0:
            errors.append("max_iterations must be positive")
        
        if self.default_track_width <= 0:
            errors.append("default_track_width must be positive")
        
        if self.default_via_size <= 0:
            errors.append("default_via_size must be positive")
        
        if self.default_clearance < 0:
            errors.append("default_clearance must be non-negative")
        
        if self.manhattan_grid_resolution <= 0:
            errors.append("manhattan_grid_resolution must be positive")
        
        return errors


@dataclass
class DisplaySettings:
    """Settings for visualization and display."""
    show_traces: bool = True
    show_vias: bool = True
    show_components: bool = True
    show_grid: bool = False
    show_keepouts: bool = True
    
    # Colors (RGB tuples)
    background_color: tuple = (30, 30, 30)
    board_color: tuple = (40, 60, 40)
    trace_color: tuple = (0, 255, 0)
    via_color: tuple = (255, 255, 0)
    component_color: tuple = (150, 150, 150)
    grid_color: tuple = (80, 80, 80)
    
    # Rendering settings
    antialiasing: bool = True
    high_dpi_support: bool = True
    animation_enabled: bool = True
    animation_speed: float = 1.0
    
    # Window settings
    window_width: int = 1600
    window_height: int = 1000
    remember_window_position: bool = True
    window_position: Optional[tuple] = None


@dataclass
class GPUSettings:
    """Settings for GPU acceleration."""
    enabled: bool = True
    preferred_device: str = "auto"  # "auto", "cuda", "cpu"
    memory_limit_mb: Optional[int] = None
    use_memory_pool: bool = True
    
    # CUDA specific settings
    cuda_device_id: int = 0
    cuda_streams: int = 4
    cuda_async: bool = True
    
    # Performance settings
    batch_size: int = 1000
    max_parallel_nets: int = 8
    
    def validate(self) -> List[str]:
        """Validate GPU settings."""
        errors = []
        
        if self.preferred_device not in ["auto", "cuda", "cpu"]:
            errors.append("preferred_device must be 'auto', 'cuda', or 'cpu'")
        
        if self.memory_limit_mb is not None and self.memory_limit_mb <= 0:
            errors.append("memory_limit_mb must be positive if specified")
        
        if self.cuda_device_id < 0:
            errors.append("cuda_device_id must be non-negative")
        
        if self.cuda_streams <= 0:
            errors.append("cuda_streams must be positive")
        
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.max_parallel_nets <= 0:
            errors.append("max_parallel_nets must be positive")
        
        return errors


@dataclass
class KiCadSettings:
    """Settings for KiCad integration."""
    api_mode: str = "ipc"  # "ipc", "swig", "auto"
    ipc_host: str = "localhost"
    ipc_port: int = 6000
    ipc_timeout: float = 30.0
    
    # File handling
    auto_save: bool = True
    backup_before_routing: bool = True
    backup_directory: str = "backups"
    
    # Board loading
    auto_detect_netclasses: bool = True
    respect_locked_tracks: bool = True
    respect_keepout_areas: bool = True
    
    # Route creation
    create_tracks: bool = True
    create_vias: bool = True
    update_ratsnest: bool = True
    refresh_display: bool = True
    
    def validate(self) -> List[str]:
        """Validate KiCad settings."""
        errors = []
        
        if self.api_mode not in ["ipc", "swig", "auto"]:
            errors.append("api_mode must be 'ipc', 'swig', or 'auto'")
        
        if self.ipc_port <= 0 or self.ipc_port > 65535:
            errors.append("ipc_port must be between 1 and 65535")
        
        if self.ipc_timeout <= 0:
            errors.append("ipc_timeout must be positive")
        
        return errors


@dataclass
class LoggingSettings:
    """Settings for logging configuration."""
    level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    log_file: str = "orthoroute.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    
    # Format settings
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Component-specific levels
    component_levels: Dict[str, str] = field(default_factory=lambda: {
        "orthoroute.algorithms": "DEBUG",
        "orthoroute.infrastructure.gpu": "INFO",
        "orthoroute.infrastructure.kicad": "INFO",
    })
    
    def validate(self) -> List[str]:
        """Validate logging settings."""
        errors = []
        
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            errors.append(f"level must be one of {valid_levels}")
        
        if self.max_file_size_mb <= 0:
            errors.append("max_file_size_mb must be positive")
        
        if self.backup_count < 0:
            errors.append("backup_count must be non-negative")
        
        for component, level in self.component_levels.items():
            if level not in valid_levels:
                errors.append(f"component level for {component} must be one of {valid_levels}")
        
        return errors


@dataclass
class ApplicationSettings:
    """Main application settings container."""
    routing: RoutingSettings = field(default_factory=RoutingSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    gpu: GPUSettings = field(default_factory=GPUSettings)
    kicad: KiCadSettings = field(default_factory=KiCadSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    
    # Application metadata
    version: str = "0.2.0"
    config_version: int = 1
    
    def validate(self) -> Dict[str, List[str]]:
        """Validate all settings and return errors by category."""
        return {
            "routing": self.routing.validate(),
            "display": [],  # DisplaySettings doesn't have validation
            "gpu": self.gpu.validate(),
            "kicad": self.kicad.validate(),
            "logging": self.logging.validate()
        }