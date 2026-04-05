---
name: Refactor UnifiedPathFinder
description: Safe refactoring patterns for breaking down the 3,936-line UnifiedPathFinder monolith
category: refactoring
keywords: [refactoring, pathfinder, extraction, clean-architecture, single-responsibility]
---

# Refactoring UnifiedPathFinder Skill

This skill provides safe patterns for refactoring the massive `unified_pathfinder.py` (3,936 lines) into maintainable components.

## ⚠️ Prerequisites Before Refactoring

**CRITICAL**: Do NOT attempt major refactoring without tests!

Required before proceeding:
- [ ] Characterization tests for current behavior
- [ ] Unit tests for extracted components
- [ ] Integration tests with real board files
- [ ] CI/CD pipeline to catch regressions

See `.github/agents/testing.agent.md` for test creation guidance.

## Current State Analysis

**File**: `orthoroute/algorithms/manhattan/unified_pathfinder.py`
- **Lines**: 3,936
- **Classes**: 1 monolithic class (`UnifiedPathFinder`)
- **Methods**: ~60+
- **Responsibilities**: Everything (lattice building, pathfinding, congestion, via logic, grid management)

**Existing Mixin Architecture** (proof that extraction is possible):
```
orthoroute/algorithms/manhattan/pathfinder/
├── config.py              # Configuration dataclass
├── lattice_mixins.py      # Lattice building methods
├── congestion_mixins.py   # Congestion calculation
├── pathfinding_mixins.py  # Core routing logic
└── via_mixins.py          # Via pathfinding
```

The mixins prove the code CAN be decomposed - but mixins are a code smell indicating classes doing too much.

## Refactoring Strategy: Extract and Test

### Phase 1: Extract Pure Functions (Safest)

**Target**: Methods with no state dependencies
**Risk**: Low
**Tests Required**: Unit tests only

**Example - Extract Coordinate Transformations:**

```python
# BEFORE: Method in UnifiedPathFinder (lines 450-460)
def xyz_to_gid(self, x, y, z):
    """Convert (x,y,z) to global ID"""
    return z * self.Nx * self.Ny + y * self.Nx + x

# AFTER: Standalone utility class
# orthoroute/algorithms/manhattan/coordinate_mapper.py
class CoordinateMapper:
    """Immutable coordinate transformation utilities"""
    
    def __init__(self, nx: int, ny: int, nz: int):
        self.nx = nx
        self.ny = ny
        self.nz = nz
    
    def xyz_to_gid(self, x: int, y: int, z: int) -> int:
        """Convert (x,y,z) to global ID"""
        if not (0 <= x < self.nx and 0 <= y < self.ny and 0 <= z < self.nz):
            raise ValueError(f"Coordinates out of bounds: ({x},{y},{z})")
        return z * self.nx * self.ny + y * self.nx + x
    
    def gid_to_xyz(self, gid: int) -> tuple[int, int, int]:
        """Convert global ID to (x,y,z)"""
        z = gid // (self.nx * self.ny)
        remainder = gid % (self.nx * self.ny)
        y = remainder // self.nx
        x = remainder % self.nx
        return x, y, z

# tests/test_coordinate_mapper.py
def test_roundtrip():
    mapper = CoordinateMapper(nx=10, ny=10, nz=6)
    gid = mapper.xyz_to_gid(5, 3, 2)
    assert mapper.gid_to_xyz(gid) == (5, 3, 2)
```

**Migration in UnifiedPathFinder:**
```python
class UnifiedPathFinder:
    def __init__(self, ...):
        self.coord_mapper = CoordinateMapper(self.Nx, self.Ny, self.Nz)
    
    def xyz_to_gid(self, x, y, z):
        # Delegate to extracted class
        return self.coord_mapper.xyz_to_gid(x, y, z)
```

### Phase 2: Extract Value Objects (Low Risk)

**Target**: Data structures with behavior
**Risk**: Low
**Tests Required**: Unit tests + integration tests

**Example - Extract Edge Cost Calculation:**

```python
# orthoroute/algorithms/manhattan/cost_calculator.py
from dataclasses import dataclass

@dataclass(frozen=True)
class EdgeCostParams:
    """Parameters for edge cost calculation"""
    base_cost: float
    overuse_count: int
    historical_congestion: float
    present_congestion_factor: float = 1.0
    history_factor: float = 0.5

class CongestionCostCalculator:
    """Calculate edge costs based on congestion"""
    
    @staticmethod
    def calculate_cost(params: EdgeCostParams) -> float:
        """
        PathFinder cost formula:
        cost = base_cost * (1 + present_factor * overuse + history_factor * h_cost)
        """
        if params.overuse_count <= 1:
            return params.base_cost
        
        overuse_penalty = params.present_congestion_factor * (params.overuse_count - 1)
        history_penalty = params.history_factor * params.historical_congestion
        
        return params.base_cost * (1.0 + overuse_penalty + history_penalty)

# tests/test_cost_calculator.py
def test_no_overuse_no_penalty():
    params = EdgeCostParams(base_cost=1.0, overuse_count=1, historical_congestion=0.0)
    cost = CongestionCostCalculator.calculate_cost(params)
    assert cost == 1.0

def test_overuse_increases_cost():
    params = EdgeCostParams(base_cost=1.0, overuse_count=3, historical_congestion=0.0)
    cost = CongestionCostCalculator.calculate_cost(params)
    assert cost > 1.0
```

### Phase 3: Extract Service Objects (Medium Risk)

**Target**: Cohesive groups of related methods
**Risk**: Medium
**Tests Required**: Unit + integration + characterization tests

**Example - Extract Via Pathfinding:**

```python
# orthoroute/algorithms/manhattan/via_pathfinder.py
from typing import Optional, List
from .types import ViaCandidate, LayerPair

class ViaPathfinder:
    """Handles via pathfinding between layer pairs"""
    
    def __init__(self, graph, coord_mapper, cost_calculator):
        self.graph = graph
        self.coord_mapper = coord_mapper
        self.cost_calculator = cost_calculator
    
    def find_via_path(
        self, 
        start_layer: int, 
        end_layer: int,
        x: int, 
        y: int
    ) -> Optional[List[ViaCandidate]]:
        """
        Find path through via barrel from start_layer to end_layer.
        
        Handles blind, buried, and through vias according to board stackup.
        """
        if start_layer == end_layer:
            return None
        
        # Extract via pathfinding logic from UnifiedPathFinder
        # (currently lines 2100-2400 circa)
        ...
        
        return via_path

# tests/test_via_pathfinder.py
def test_adjacent_layer_via():
    """Test via between adjacent layers"""
    # Setup mock graph, coord_mapper, cost_calculator
    finder = ViaPathfinder(graph, coord_mapper, cost_calculator)
    path = finder.find_via_path(start_layer=0, end_layer=1, x=5, y=5)
    assert path is not None
    assert len(path) == 1  # Single via hop

def test_via_skips_keepout_layers():
    """Verify via path respects keepout constraints"""
    # Test critical invariant
    pass
```

### Phase 4: Extract Strategy Pattern (Higher Risk)

**Target**: Algorithm variants
**Risk**: High (requires interface design)
**Tests Required**: Full test suite

**Example - Extract Pathfinding Algorithms:**

```python
# orthoroute/domain/services/pathfinding_strategy.py
from abc import ABC, abstractmethod

class PathfindingStrategy(ABC):
    """Abstract pathfinding algorithm"""
    
    @abstractmethod
    def find_path(self, source, target, graph, cost_fn):
        """Find shortest path from source to target"""
        pass

# orthoroute/algorithms/manhattan/dijkstra_pathfinder.py
class DijkstraPathfinder(PathfindingStrategy):
    """CPU-based Dijkstra implementation"""
    
    def find_path(self, source, target, graph, cost_fn):
        # Extract from UnifiedPathFinder._cpu_pathfind()
        pass

# orthoroute/algorithms/manhattan/cuda_pathfinder.py
class CUDAPathfinder(PathfindingStrategy):
    """GPU-accelerated parallel Dijkstra"""
    
    def find_path(self, source, target, graph, cost_fn):
        # Extract from UnifiedPathFinder._gpu_pathfind()
        pass

# In UnifiedPathFinder:
class UnifiedPathFinder:
    def __init__(self, use_gpu=True):
        self.pathfinder = (
            CUDAPathfinder() if use_gpu and has_cuda()
            else DijkstraPathfinder()
        )
    
    def route_net(self, net):
        path = self.pathfinder.find_path(
            source=net.start_node,
            target=net.end_node,
            graph=self.graph,
            cost_fn=self.get_edge_cost
        )
```

## Refactoring Anti-Patterns (Avoid These!)

### ❌ Anti-Pattern 1: Big Bang Refactoring
```python
# DON'T: Rewrite entire UnifiedPathFinder in one PR
class NewPathFinder:  # 2,000 lines of new code
    # Complete rewrite with no backward compatibility
```

**Why it fails**: Impossible to review, no incremental testing, high risk

### ❌ Anti-Pattern 2: Premature Abstraction
```python
# DON'T: Create abstract interfaces before understanding patterns
class AbstractPathfinderFactory(ABC):
    @abstractmethod
    def create_path_resolver(self) -> PathResolver:
        pass

class PathResolver(ABC):
    @abstractmethod  
    def resolve(self, strategy: PathStrategy) -> Path:
        pass
```

**Why it fails**: Over-engineering, no clear benefit, harder to understand

### ❌ Anti-Pattern 3: Breaking the Graph Freeze Invariant
```python
# DON'T: Modify CSR structure during refactoring
class ExtractedComponent:
    def some_method(self):
        self.graph.add_edge(...)  # FORBIDDEN - graph is frozen!
```

**Why it fails**: Violates critical invariant, causes "LATTICE INTEGRITY failed" error

### ❌ Anti-Pattern 4: Ignoring Mixin Structure
```python
# DON'T: Duplicate logic already in mixins
class NewCongestionHandler:
    def calculate_congestion(self):
        # Re-implementing logic from congestion_mixins.py
```

**Why it fails**: Creates inconsistency, wasted effort

## Safe Refactoring Checklist

Before merging any extraction:

- [ ] New component has >80% test coverage
- [ ] Integration tests pass with real board files
- [ ] Performance benchmarks show no regression
- [ ] UnifiedPathFinder still works (backward compatibility)
- [ ] Documentation updated
- [ ] No graph structure mutations
- [ ] Clean architecture boundaries respected
- [ ] Code review approved

## Incremental Migration Path

**Week 1-2**: Extract pure functions
- `CoordinateMapper`
- `LayerDirectionCalculator`
- Geometry utilities

**Week 3-4**: Extract value objects
- `CongestionCostCalculator`
- `EdgeCostParams`
- `ViaConstraints`

**Week 5-8**: Extract service objects
- `ViaPathfinder`
- `PadEscapeRouter`
- `ObstacleManager`

**Month 3+**: Strategy pattern refactoring
- `PathfindingStrategy` hierarchy
- `CongestionStrategy` variants
- `GraphBuilder` interface

## Measuring Progress

Track these metrics:

```python
# Script to measure refactoring progress
import ast

def count_lines_in_class(filepath, classname):
    with open(filepath) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == classname:
            return node.end_lineno - node.lineno
    return 0

# Goal: Reduce UnifiedPathFinder from 3,936 → <500 lines
current_lines = count_lines_in_class('unified_pathfinder.py', 'UnifiedPathFinder')
print(f"Progress: {3936 - current_lines} lines extracted")
```

**Target**: Reduce to <500 lines (coordinator/facade only)

## Remember

- **Tests first, refactor second** - Never extract without tests
- **One responsibility at a time** - Extract smallest coherent unit
- **Preserve behavior** - Refactoring changes structure, not behavior
- **Incremental delivery** - Small PRs that can be reviewed
- **Respect invariants** - Graph freeze, coordinate mapping, layer consistency

When in doubt:
1. Write characterization test
2. Extract smallest possible component
3. Verify tests still pass
4. Commit and move to next extraction

**Success = UnifiedPathFinder becomes a thin coordinator delegating to well-tested components.**
