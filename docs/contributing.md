# Contributing to OrthoRoute

OrthoRoute started as a solo project to route a massive backplane that existing autorouters couldn't handle, and it's grown into something bigger. There's real work happening here -- GPU optimization, novel routing algorithms, and actual PCB routing that works -- but it's grad student-level code that needs software engineering discipline to become production-ready.

I swear to fucking god there is never going to be a discord or slack for this shit. WE'RE ALREADY ON A MESSAGING PLATFORM IT'S CALLED GITHUB YOU MAY CONTACT ME VIA PULL REQUESTS AND ISSUES

## Project Status: Research Code → Production Tool

**What's working:**
- GPU-accelerated PathFinder routing (127× speedup)
- Novel portal escape architecture (16% → 80%+ routing success)
- Routes complex multi-layer boards (tested up to 32 layers, 3,200 pads)
- Blind/buried via support (870 via pairs)

**What needs work:**
- ⚠️ No unit tests
- ⚠️ Large files (3,936-line UnifiedPathFinder class)
- ⚠️ Configuration scattered across multiple places

If you're okay working on code with rough edges but interesting algorithms, read on.

---

## Quick Start for Contributors

### Prerequisites

- **Python 3.8+**
- **Git**
- **KiCad 9.0+** (for IPC API testing)
- **CUDA-capable GPU** (optional but recommended for GPU code)
- **PyQt6** (for GUI work)

### Setup

```bash
# Clone the repo
git clone https://github.com/bbenchoff/OrthoRoute.git
cd OrthoRoute

# Install dependencies
pip install -r requirements.txt

# Run a quick test
python main.py --test-manhattan

# Or launch the GUI (requires KiCad running)
python main.py plugin
```

### Repository Structure

```
OrthoRoute/
├── orthoroute/
│   ├── algorithms/manhattan/      # Core routing algorithms
│   │   ├── unified_pathfinder.py  # Main routing engine (3,936 lines - needs refactoring)
│   │   ├── pathfinder/            # Mixin-based components
│   │   └── pad_escape_planner.py  # Portal escape architecture
│   ├── domain/                    # Domain models (Board, Net, Pad, Via)
│   ├── application/               # Use cases and orchestration
│   ├── infrastructure/            # KiCad integration, GPU providers
│   └── presentation/              # GUI and CLI
├── main.py                        # Entry point
├── docs/                          # Documentation
└── tests/                         # Tests (currently empty - hint hint)
```

---

## How to Contribute

### 1. **Start Small**

Don't try to refactor the entire 3,936-line UnifiedPathFinder on your first PR. Pick something focused:

**Good first contributions:**
- Add unit tests for lattice building
- Document the coordinate system (x, y, z mapping)
- Fix typos in comments
- Add type hints to functions missing them
- Write examples for the README

**More involved:**
- Extract classes from UnifiedPathFinder
- Implement KiCad geometry export
- Add integration tests
- Fix convergence oscillation

### 2. **Areas That Need Help**

#### **Critical Priority: Tests**
The project has **zero unit tests**. This is the biggest blocker to refactoring.

**Start here:**
```python
# tests/test_lattice_builder.py
def test_lattice_node_count():
    """Verify lattice creates correct number of nodes"""
    builder = LatticeBuilder(Nx=10, Ny=10, Nz=6)
    lattice = builder.build()
    assert lattice.num_nodes == 10 * 10 * 6

def test_manhattan_adjacency():
    """Verify no diagonal edges in graph"""
    # Check that all edges connect orthogonally
```

**What to test:**
- Lattice building (node count, adjacency)
- CSR matrix construction (integrity checks)
- Via pooling accounting (usage counts)
- Portal escape planning (DRC compliance)
- Pad mapping (nearest node finding)

#### **High Priority: Refactoring**

The `unified_pathfinder.py` file is 3,936 lines. It needs to be broken up:

**Extraction candidates:**
```python
# Current: Everything in UnifiedPathFinder
# Target: Separate classes

class LatticeManager:
    """Builds and manages the 3D routing lattice"""

class EdgeAccountant:
    """Tracks edge usage and calculates costs"""
    # (This actually exists but needs cleanup)

class ConvergenceManager:
    """Handles PathFinder negotiation loop"""

class GeometryEmitter:
    """Extracts tracks and vias from routing solution"""
```

#### **Medium Priority: Documentation**

**What's missing:**
- API documentation (Sphinx or similar)
- Coordinate system explanation (how (x,y,z) maps to mm and layers)
- PathFinder algorithm overview (it's complex)
- Architecture diagrams (Clean Architecture layers)
- GPU kernel documentation (CUDA code needs explanation)

#### **Lower Priority: Code Quality**

- Add type hints everywhere
- Remove commented-out code
- Consolidate configuration (currently in 4 places)
- Fix `hasattr()` fragility (better state management)
- Extract magic numbers to named constants

---

## Coding Standards

### Python Style

- **Follow PEP 8** (mostly—the existing code is 80% compliant)
- **Type hints encouraged** (but not required yet)
- **Docstrings for public functions** (Google style preferred)
- **Meaningful variable names** (the code is already good at this)

**Example:**
```python
def build_lattice(
    self,
    Nx: int,
    Ny: int,
    Nz: int,
    pitch: float = 0.4
) -> Lattice:
    """Build a 3D Manhattan routing lattice.

    Args:
        Nx: Grid width (number of columns)
        Ny: Grid height (number of rows)
        Nz: Number of layers (copper layers)
        pitch: Grid spacing in millimeters

    Returns:
        Lattice object with nodes and adjacency structure

    Raises:
        ValueError: If layer count is less than 2
    """
```

### Testing

**When you add tests (please do!):**
- Use `pytest`
- Test files in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- One assertion per test (mostly)

**Example:**
```python
# tests/test_via_pooling.py
import pytest
from orthoroute.algorithms.manhattan.unified_pathfinder import UnifiedPathFinder

def test_via_column_accounting():
    """Verify via column usage increments correctly"""
    pf = UnifiedPathFinder()
    # Setup...
    pf._increment_via_column_use(x=5, y=10)
    assert pf.via_col_use[5, 10] == 1
```

### Git Workflow

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b fix-via-pooling-accounting`
3. **Make focused commits:** Small, logical changes with clear messages
4. **Write descriptive commit messages:**
   ```
   Add unit tests for lattice builder

   - Test node count calculation
   - Test Manhattan adjacency (no diagonals)
   - Test layer discipline (H/V alternating)
   ```
5. **Push to your fork:** `git push origin fix-via-pooling-accounting`
6. **Open a Pull Request** with description of what and why

### Pull Request Guidelines

**Good PR description:**
```markdown
## What
Adds unit tests for the lattice builder module

## Why
The project currently has no tests, making refactoring risky. This adds
coverage for the most critical component: lattice building.

## Tests
- `test_lattice_node_count`: Verifies Nx * Ny * Nz nodes created
- `test_manhattan_adjacency`: Checks no diagonal edges
- `test_layer_discipline`: Validates H/V alternating pattern

## Checklist
- [x] Tests pass locally
- [x] Code follows PEP 8
- [x] Added docstrings to new functions
- [ ] Updated documentation (n/a for this PR)
```

**PR review process:**
- I'll try to review within 2-3 days
- Expect honest feedback (the code has issues; we're fixing them)
- Iteration is expected—first PR probably won't be perfect
- Be open to suggestions but also push back if you disagree

---

## Architecture Overview (For Larger Changes)

OrthoRoute uses **Clean Architecture** with four layers:

### 1. Domain Layer (`orthoroute/domain/`)
**Pure business logic, no dependencies**

- `models/`: Board, Net, Pad, Component, Via, Route, Segment
- `services/`: RoutingEngine interface, PathfindingService
- `events/`: Domain events (routing progress, board updates)

**Rule:** Domain never imports from other layers

### 2. Application Layer (`orthoroute/application/`)
**Use cases and orchestration**

- `commands/`: StartRouting, CancelRouting, etc.
- `queries/`: GetRoutingStatus, GetBoardStats, etc.
- `services/`: RoutingOrchestrator (coordinates domain services)
- `interfaces/`: Repository contracts, GPU provider interface

**Rule:** Application imports Domain but not Infrastructure/Presentation

### 3. Infrastructure Layer (`orthoroute/infrastructure/`)
**External integrations**

- `kicad/`: IPC API, SWIG, file parsing
- `gpu/`: CUDA provider, CPU fallback
- `persistence/`: In-memory repositories

**Rule:** Infrastructure implements Application interfaces

### 4. Presentation Layer (`orthoroute/presentation/`)
**User interfaces**

- `gui/`: PyQt6 interactive viewer
- `plugin/`: KiCad plugin entry point
- `pipeline.py`: Unified routing workflow

**Rule:** Presentation coordinates Application layer

### When to use which layer:

- **Adding a new routing algorithm?** → Domain + Infrastructure (GPU)
- **Adding KiCad export?** → Infrastructure
- **Adding a GUI feature?** → Presentation
- **Adding a test?** → Tests folder (import whatever you need)

---

## GPU Development

### CUDA Kernel Guidelines

The GPU kernels are in `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py`

**If you're modifying GPU code:**
- Test on both CUDA and CPU paths
- Use `cp.cuda.Device().synchronize()` after kernel launches for debugging
- Profile with `nvprof` or `nsys` for performance work
- Document memory layout (especially for CSR matrices)

**Example kernel pattern:**
```python
# Device-side kernel (CUDA C++)
kernel_code = r'''
extern "C" __global__
void my_kernel(int* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Do work
    }
}
'''

# Host-side launch (Python)
kernel = cp.RawKernel(kernel_code, 'my_kernel')
blocks = (N + 255) // 256
kernel((blocks,), (256,), (data, N))
```

**GPU testing:**
- Always provide CPU fallback
- Test on boards of various sizes
- Monitor memory usage (boards with 10K+ pads use significant GPU RAM)

---

## Documentation Guidelines

### Code Comments

**When to comment:**
- Why, not what (code shows what)
- Non-obvious algorithms (PathFinder iterations, via pooling logic)
- Magic numbers (why 0.6 alpha? why 1.25 multiplier?)
- Performance-critical sections

**Example:**
```python
# Good comment
alpha = 0.6  # Weight current overuse more than history to respond to recent changes

# Bad comment
alpha = 0.6  # Set alpha to 0.6
```

### Docstrings

**Required for:**
- All public functions/methods
- All classes
- Complex algorithms

**Use Google style:**
```python
def route_multiple_nets(
    self,
    nets: List[Net],
    max_iterations: int = 40
) -> RoutingResult:
    """Route multiple nets using PathFinder negotiated congestion.

    Iteratively routes nets, applying increasing congestion penalties
    to overused edges until convergence or max iterations reached.

    Args:
        nets: List of Net objects to route
        max_iterations: Maximum PathFinder iterations (default: 40)

    Returns:
        RoutingResult containing:
            - routed_nets: Successfully routed nets
            - failed_nets: Nets that couldn't be routed
            - statistics: Convergence metrics, timing data

    Raises:
        ValueError: If nets list is empty
        MemoryError: If GPU runs out of memory (large boards)

    Notes:
        This implements the PathFinder algorithm from FPGA routing,
        adapted for PCB Manhattan routing patterns. See docs/pathfinder.md
        for algorithm details.
    """
```

---

## Common Questions

### "Where should I start?"

**If you're new to the codebase:** Add tests for existing functions. This forces you to understand the code while contributing something valuable.

**If you're experienced:** Pick from the high-priority areas (refactoring, KiCad export, convergence).

### "The 3,936-line file is intimidating. How do I understand it?"

1. Start with `docs/Manhattan_Design_Doc.md`
2. Read the mixins one at a time (they're separated by concern)
3. Follow the execution flow: `main.py` → `pipeline.py` → `unified_pathfinder.py`
4. Use the extensive logging to see what happens at runtime

### "Can I add a dependency?"

**Probably yes if it's for:**
- Testing (pytest plugins, etc.)
- Documentation (Sphinx, etc.)
- Development tools (linters, formatters)

**Ask first if it's for:**
- Core functionality (routing, graph algorithms)
- GUI (PyQt6 is already heavy)
- GPU (CuPy is the standard here)

**No**
- PyTorch (because icky sparse matrices and I don't need any AI weirdos in here)
- Discord, Slack (see above)

### "I found a bug. What do I do?"

1. **Open an issue** with:
   - Description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Log files if applicable

2. **If you can fix it:**
   - Include the fix in the issue or open a PR
   - Add a test that would have caught the bug

3. **If it's a convergence/routing quality issue:**
   - Include the board file (if shareable)
   - Include logs showing the oscillation pattern
   - Check `PATHFINDER_CONVERGENCE_DEBUG_GUIDE.md`

### "I disagree with a design decision. Can I change it?"

**Open an issue to discuss first.** Some things that look weird have reasons:

- Mixins instead of inheritance: Allows composition, GPU/CPU code separation
- Multiple config sources: Legacy from experimentation (agree it's messy)
- Large files: Result of consolidation effort (agree it needs further breakdown)

That said, if you have a better approach with clear benefits, let's discuss it.

### "Will my PR be accepted?"

**Likely yes if:**
- It adds tests
- It fixes a real bug
- It improves code quality without breaking functionality
- It adds documentation
- It's well-explained in the PR description

**Maybe if:**
- It's a large refactoring (needs discussion first)
- It changes core algorithms (needs validation)
- It adds significant dependencies

**Probably not if:**
- It's a style-only change with no functional benefit
- It breaks existing functionality
- It's not explained (what/why)

---

## Communication

### Where to ask questions:

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, architecture discussion
- **Pull Request comments**: Code-specific questions

### Response time expectations:

This is a solo project but I'm unemployed, so:
- **Issues/PRs:** 2-3 day response time (usually faster)
- **Urgent bugs:** Same day if possible
- **Long discussions:** May take a week to think through

### Code of Conduct:

There is no code of conduct for a reason. I will kick out and ignore assholes. There will be no rule lawyering (rule chicanery?). I reserve the right to be arbitrary and capricious.  

---

## Getting Credit

- Contributors will be listed in the README (if you want to be)
- Significant contributions may warrant co-authorship on any academic papers
- Your GitHub profile gets commits → looks good for jobs/grad school

---

## Final Notes

This project exists because existing autorouters couldn't handle a massive backplane. It's grown into something bigger: novel algorithms (portal escapes), serious GPU optimization (127× speedup), and research-level work that could be production-ready with some polish.

The code has rough edges because it's research code. That's fine—we're making it better. Your contributions, whether tests, refactoring, or new features, move it closer to being a real tool that real people use for real PCB routing.

If you're excited about GPU programming, routing algorithms, or just want to work on something that solves actual problems, welcome aboard.

---

**Questions?** Open an issue or start a discussion. Thanks for contributing!

— BBenchoff
