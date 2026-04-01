"""OrthoRoute - Advanced PCB Autorouter with Manhattan routing and GPU acceleration.

All heavy imports are lazy to avoid triggering KiCad's pcbnew action_plugin
assertion when the package is loaded inside KiCad's plugin runner.
Use explicit imports from submodules instead:
    from orthoroute.domain.models.board import Board
    from orthoroute.shared.configuration import initialize_config
"""

__version__ = "1.0.0"
__author__ = "OrthoRoute Team"
__description__ = "Advanced PCB Autorouter with Manhattan routing and GPU acceleration"


def __getattr__(name):
    """Lazy-load public symbols on first access."""
    _lazy = {
        # Domain models
        "Board": "orthoroute.domain.models.board",
        "Component": "orthoroute.domain.models.board",
        "Net": "orthoroute.domain.models.board",
        "Layer": "orthoroute.domain.models.board",
        "Route": "orthoroute.domain.models.routing",
        "Segment": "orthoroute.domain.models.routing",
        "Via": "orthoroute.domain.models.routing",
        "DRCConstraints": "orthoroute.domain.models.constraints",
        "NetClass": "orthoroute.domain.models.constraints",
        # Services
        "RoutingEngine": "orthoroute.domain.services.routing_engine",
        "RoutingOrchestrator": "orthoroute.application.services.routing_orchestrator",
        # Configuration
        "ConfigManager": "orthoroute.shared.configuration.config_manager",
        "get_config": "orthoroute.shared.configuration.config_manager",
        "initialize_config": "orthoroute.shared.configuration.config_manager",
        "ApplicationSettings": "orthoroute.shared.configuration.settings",
        # Plugin
        "KiCadPlugin": "orthoroute.presentation.plugin.kicad_plugin",
        # Algorithms
        "ManhattanRRGRoutingEngine": "orthoroute.algorithms.manhattan.manhattan_router_rrg",
        # Infrastructure
        "KiCadIPCAdapter": "orthoroute.infrastructure.kicad.ipc_adapter",
        "CUDAProvider": "orthoroute.infrastructure.gpu.cuda_provider",
        "CPUProvider": "orthoroute.infrastructure.gpu.cpu_fallback",
    }
    if name in _lazy:
        import importlib
        module = importlib.import_module(_lazy[name])
        return getattr(module, name)
    raise AttributeError(f"module 'orthoroute' has no attribute {name!r}")


__all__ = [
    # Version info
    "__version__", "__author__", "__description__",
    # Domain models
    "Board", "Component", "Net", "Layer",
    "Route", "Segment", "Via",
    "DRCConstraints", "NetClass",
    # Services
    "RoutingEngine", "RoutingOrchestrator",
    # Configuration
    "ConfigManager", "get_config", "initialize_config", "ApplicationSettings",
    # Applications
    "KiCadPlugin",
    # Algorithms
    "ManhattanRRGRoutingEngine",
    # Infrastructure
    "KiCadIPCAdapter", "CUDAProvider", "CPUProvider",
]