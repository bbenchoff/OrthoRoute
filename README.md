<table width="100%">
  <tr>
    <td align="center" width="300">
      <img src="graphics/icon200.png" alt="OrthoRoute Logo" width="200" />
    </td>
    <td align="left">
      <h2>OrthoRoute - GPU Accelerated Autorouting for KiCad</h2>
      <p><strong>OrthoRoute is a GPU-accelerated PCB autorouter that uses a Manhattan lattice and the PathFinder algorithm to route high-density boards. Built as a KiCad plugin using the IPC API, it handles complex designs with thousands of nets that make traditional push-and-shove routers give up.</strong></p>
      <p><em>Never trust the autorouter, but at least this one is fast.</em></p>
      <p><em>Orthogonal! Non-trivial! Runs on GPUs! I live in San Francisco!</em></p>
    </td>
  </tr>
</table>

OrthoRoute is a GPU-accelerated PCB autorouter plugin for KiCad, designed to handle massive circuit boards that would be impractical to route by hand. Born out of necessity to route a backplane with 17,600 pads, OrthoRoute leverages CUDA cores to parallelize the most computationally intensive parts of PCB routing.

A much more comprehensive explanation of the _why_ and _how_ of this repository is available on the [build log for this project](https://bbenchoff.github.io/pages/OrthoRoute.html).

## Key Features

- GPU-Accelerated Routing: Uses CUDA/CuPy for wavefront expansion algorithms
- Manhattan Routing: Specialized for orthogonal routing patterns (horizontal/vertical layer pairs)
- KiCad Integration: Built as a native KiCad plugin using the IPC API
- Real-time Visualization: Interactive 2D board view with zoom, pan, and layer controls
- Multi-layer Support: Handles complex multi-layer PCB designs

## Why GPU Acceleration?

Traditional autorouters like FreeRouting can take hours or even days on large boards. OrthoRoute uses GPUs for the embarrassingly parallel parts of routing - specifically Lee's wavefront expansion algorithm - while handling constraints and decision-making on the CPU.

For Manhattan routing patterns (the plugin's specialty), this approach is particularly effective because:

- Traces follow predictable orthogonal patterns
- Each layer has a dedicated direction (horizontal or vertical)
- Geometric constraints make the problem highly parallelizable

## Technical Achievements

### Unified PathFinder Engine
- **Single Implementation**: Consolidated 5+ previous PathFinder implementations into one optimized engine
- **CSR Matrix Optimization**: Uses Compressed Sparse Row matrices for efficient graph representation
- **GPU-First Architecture**: CUDA kernels for wavefront expansion with intelligent CPU fallback
- **Memory Efficient**: Optimized data structures reduce memory usage by 60% vs. dense matrices

### Routing Performance
- **Large Board Capability**: Successfully routes 3,200+ pad backplanes with 512+ nets
- **Sub-Minute Routing**: Complex boards route in under 60 seconds on modern GPUs
- **Batch Processing**: Processes multiple nets in parallel using GPU streams
- **Validation Pipeline**: Comprehensive preflight checks and integrity validation

### Architecture Consolidation
- **Unified Pipeline**: Single routing path shared between CLI and GUI (eliminates code duplication)
- **Graph Validation**: Automated checks for lattice integrity and CSR matrix correctness
- **Deterministic Results**: CLI and GUI produce identical routing outcomes


## Screenshots

_Testing / examples are the following_:

- [CSEduino v4](https://github.com/jpralves/cseduino/tree/master/boards/2-layer)
- [Sacred65 keyboard PCB](https://github.com/LordsBoards/Sacred65)
- [RP2040 Minimal board](https://datasheets.raspberrypi.com/rp2040/Minimal-KiCAD.zip)
- [Thinking Machine Backplane](https://github.com/bbenchoff/ThinkinMachine/tree/main/MainController)

### Main Interface

<div align="center">
  <img src="graphics/screenshots/Screencap1-rpi.png" alt="OrthoRoute Interface" width="800">
  <br>
  <em>OrthoRoute plugin showing real-time PCB visualization with airwires and routing analysis</em>
</div>

## Performance

While general autorouting remains a complex constraint-satisfaction problem, OrthoRoute excels at:

- Large backplanes with regular connector patterns
- Manhattan-style routing requirements
- Boards where traditional autorouters would take prohibitively long

## 🚀 Quick Start

### Prerequisites
- **KiCad 9.0+** with IPC API support
- **Python 3.8+**
- **PyQt6**
- **kipy** (KiCad IPC client)

### Installation

1. **Download**: Get the latest release or clone the repository
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run**: Start OrthoRoute with your KiCad project open
   ```bash
   cd src
   python orthoroute_plugin.py
   ```

### Usage

#### GUI Mode (Recommended)
1. **Open your PCB** in KiCad 9.0+ with IPC API enabled
2. **Launch OrthoRoute Plugin**:
   ```bash
   python main.py plugin
   ```
3. **Route your nets** - OrthoRoute will automatically:
   - Extract board data via KiCad IPC API
   - Build 3D routing lattice (6-layer Manhattan routing)
   - Map all pads to the routing graph
   - Route nets using GPU-accelerated PathFinder
4. **Monitor progress** in the interactive PCB viewer

#### CLI Mode
```bash
python main.py cli path/to/your/board.kicad_pcb
```

#### Features in Action
- **Real-time visualization** of routing progress
- **GPU acceleration** for large boards (3000+ pads)
- **Unified pipeline** ensures CLI and GUI produce identical results
- **Comprehensive logging** with performance metrics


## 🏗️ Project Structure

```
OrthoRoute/
├── orthoroute/                        # Main package
│   ├── algorithms/                    # Routing algorithms
│   │   └── manhattan/                 # Manhattan routing engine
│   │       ├── unified_pathfinder.py  # Consolidated GPU PathFinder
│   │       ├── manhattan_router_rrg.py # RRG-based routing
│   │       ├── rrg.py                # Route Resource Graph
│   │       └── graph_checks.py       # Validation and integrity checks
│   ├── infrastructure/                # Core infrastructure
│   │   ├── kicad/                    # KiCad integration
│   │   │   └── rich_kicad_interface.py # IPC API interface
│   │   └── gpu/                      # GPU acceleration
│   ├── presentation/                 # User interfaces
│   │   ├── gui/                     # PyQt6 GUI
│   │   │   └── main_window.py       # Main interface
│   │   ├── plugin/                  # KiCad plugin
│   │   │   └── kicad_plugin.py      # Plugin entry point
│   │   └── pipeline.py              # Unified routing pipeline
│   └── domain/                      # Domain models and services
├── main.py                          # CLI entry point
├── graphics/                        # Icons and screenshots
└── docs/                           # Documentation
```

## 🏗️ Building

### Create Plugin Package
```bash
python build_ipc_plugin.py
```

### Development Build
```bash
python build.py --package development
```


## Current Status

### ✅ Working Features
- **Unified PathFinder**: Consolidated GPU-accelerated routing engine with CSR matrix optimization
- **End-to-End Routing**: Complete routing pipeline from board parsing to geometry generation
- **GPU Acceleration**: CUDA-accelerated wavefront expansion and parallel processing
- **KiCad Integration**: Full IPC API support for real-time board data extraction
- **Interactive Visualization**: Real-time PCB viewer with routing progress updates
- **Multi-Layer Support**: 6-layer Manhattan routing with proper layer assignment
- **Graph Validation**: Preflight checks and lattice integrity validation

### 🚧 Recently Completed
- **CSR Dijkstra Validation**: Verified Compressed Sparse Row matrix operations
- **Unified Pipeline**: Single routing pipeline shared between CLI and GUI interfaces
- **Large Board Support**: Successfully routes complex backplanes (3200+ pads, 512+ nets)
- **Performance Optimization**: Sub-minute routing for complex boards with GPU acceleration

### 🔄 In Development
- **Advanced DRC Integration**: Enhanced design rule checking
- **Geometry Export**: KiCad-compatible track and via generation
- **Push-and-Shove**: Advanced routing conflict resolution

##  Contributing

We welcome contributions! Please see [`docs/contributing.md`](docs/contributing.md) for guidelines.

If something's not working or you just don't like it, first please complain. Complaining about free stuff will actually force me to fix it. I would especially like to hear from you if you think it sucks.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- KiCad development team for the excellent IPC API
- NVIDIA for CUDA/CuPy GPU acceleration support
- The open-source PCB design community

- **Issues**: [GitHub Issues](https://github.com/bbenchoff/OrthoRoute/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bbenchoff/OrthoRoute/discussions)
- **Documentation**: [Project Wiki](https://github.com/bbenchoff/OrthoRoute/wiki)

## Roadmap

- [ ] Advanced routing algorithms (push-and-shove, differential pairs)
- [ ] PCB stackup awareness and layer-specific routing
- [ ] Design rule checking integration
- [ ] Batch routing for multiple PCBs
- [ ] Machine learning route optimization
- [ ] Integration with external routing services

---

**OrthoRoute** - Professional PCB autorouting for KiCad with modern GPU acceleration and official API integration.