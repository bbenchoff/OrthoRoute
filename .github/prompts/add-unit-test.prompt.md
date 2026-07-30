---
name: add-unit-test
description: Scaffold a pytest unit test for a given OrthoRoute function or class
---

# Add Unit Test

Scaffold a pytest unit test for the specified OrthoRoute symbol.

## Inputs

- **symbol**: The fully-qualified name of the function or class to test (e.g. `orthoroute.domain.models.board.Board`, `orthoroute.algorithms.manhattan.real_global_grid.GlobalGrid.gid_roundtrip`)
- **test_type** *(optional)*: `unit` (default) or `integration`

## Instructions

1. Locate `${symbol}` in the codebase and read its implementation.
2. Identify:
   - All inputs/outputs and their types
   - Edge cases (empty board, single pad, zero nets, OOM path, etc.)
   - Any existing inline "test" scripts (`if __name__ == "__main__"`) that document expected behavior — convert those into proper pytest assertions
3. Create the test file at `tests/<mirrored_module_path>/test_<module_name>.py`, creating the directory and an `__init__.py` if they do not exist.
4. Write tests following these conventions:
   - One `test_` function per behavior, not per method
   - Use `pytest.fixture` for shared setup (Board instances, config, grid)
   - Use `@pytest.mark.parametrize` for input/output tables
   - Prefer real domain objects over mocks; mock only I/O boundaries (KiCad IPC, file system, GPU)
   - Assert specific values, not just "no exception raised"
5. Add a `# TODO:` comment for any behavior that needs deeper integration (e.g., full routing pipeline) — do not expand scope.
6. Run `pytest tests/<new_file> -v` and confirm all tests pass.

## Example

**Input:** `symbol = "orthoroute.domain.models.board.Board"`

**Output location:** `tests/domain/models/test_board.py`

```python
import pytest
from orthoroute.domain.models.board import Board

@pytest.fixture
def empty_board():
    return Board(layers=[], nets=[], pads=[])

def test_board_has_no_nets_by_default(empty_board):
    assert empty_board.nets == []

def test_board_layer_count(empty_board):
    assert empty_board.layer_count == 0

@pytest.mark.parametrize("layer_names,expected", [
    (["F.Cu", "B.Cu"], 2),
    (["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"], 4),
])
def test_board_layer_count_parametrized(layer_names, expected):
    board = Board(layers=layer_names, nets=[], pads=[])
    assert board.layer_count == expected
```
