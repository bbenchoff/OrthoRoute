"""Synthetic backplane generator.

Builds parameterized Board objects shaped like the real target hardware:
N connectors, each a grid of pads, wired connector-to-connector. Small
instances are fast CPU regression tests; large instances approach the
"monster board" (16 connectors x 1100 pins, 32 layers) for GPU benchmarks.

Pads are placed on multiples of the 0.4mm routing pitch so the escape
planner's column snapping behaves the same as on the smoke-test boards
(use portal_x_snap_max=0.75, matching tests/conftest.py).
"""

import random
from typing import List

from orthoroute.domain.models.board import (
    Board, Component, Coordinate, Net, Pad,
)

# Multiples of the 0.4mm GRID_PITCH so pads land on lattice columns.
DEFAULT_PIN_PITCH_MM = 2.4
DEFAULT_ROW_PITCH_MM = 2.4


def make_backplane(connectors: int = 2,
                   pins_per: int = 20,
                   layers: int = 4,
                   *,
                   rows: int = 2,
                   pin_pitch_mm: float = DEFAULT_PIN_PITCH_MM,
                   row_pitch_mm: float = DEFAULT_ROW_PITCH_MM,
                   connector_spacing_mm: float = 16.0,
                   pattern: str = "pairs",
                   seed: int = 42) -> Board:
    """Build a synthetic backplane Board.

    Args:
        connectors: Number of connector footprints, placed along X.
        pins_per: Pads per connector (must divide evenly by rows).
        layers: Copper layer count for the board.
        rows: Pad columns per connector (2 = DIN-style two-row header).
        pin_pitch_mm: Pad pitch along Y within a column.
        row_pitch_mm: Spacing between the pad columns of one connector.
        connector_spacing_mm: X spacing between connector origins.
        pattern: Net wiring pattern:
            "pairs"    - random pad-to-pad matching across the whole board
                         (seeded). Mirrors the real backplane's ~2 pads/net.
            "neighbor" - pin i of connector j wired to pin i of connector
                         j+1, for even j (disjoint straight-across pairs).
            "bus"      - pin i of EVERY connector on one net (multi-pad
                         nets, connectors pads each).
        seed: RNG seed for the "pairs" pattern.

    Returns:
        A routable domain Board with every pad assigned to exactly one net.
    """
    if pins_per % rows != 0:
        raise ValueError(f"pins_per={pins_per} must divide evenly by rows={rows}")
    if pattern not in ("pairs", "neighbor", "bus"):
        raise ValueError(f"Unknown pattern: {pattern}")
    if pattern == "neighbor" and connectors % 2 != 0:
        raise ValueError("neighbor pattern needs an even connector count")

    board = Board(id="synthetic_backplane",
                  name=f"synthetic-{connectors}x{pins_per}-{layers}L-{pattern}")
    board.layer_count = layers

    pins_per_column = pins_per // rows
    # pad_grid[j][i] = Pad for pin i of connector j
    pad_grid: List[List[Pad]] = []

    for j in range(connectors):
        comp_x = 4.0 + j * connector_spacing_mm
        comp_y = 4.0
        comp = Component(id=f"J{j+1}", reference=f"J{j+1}", value="CONN",
                         footprint=f"Backplane-{pins_per}",
                         position=Coordinate(x=comp_x, y=comp_y))
        conn_pads: List[Pad] = []
        for pin in range(pins_per):
            row = pin % rows
            slot = pin // rows
            pad = Pad(id=f"J{j+1}-{pin+1}",
                      component_id=comp.id,
                      position=Coordinate(x=comp_x + row * row_pitch_mm,
                                          y=comp_y + slot * pin_pitch_mm),
                      layer="F.Cu",
                      size=(1.2, 1.2),
                      net_id=None)
            comp.pads.append(pad)
            conn_pads.append(pad)
        pad_grid.append(conn_pads)
        board.add_component(comp)

    for net_index, net_pads in enumerate(_wire(pad_grid, pattern, seed)):
        net = Net(id=f"net{net_index}", name=f"NET_{net_index}", pads=net_pads)
        for pad in net_pads:
            pad.net_id = net.id
        board.add_net(net)

    return board


def _wire(pad_grid: List[List[Pad]], pattern: str, seed: int) -> List[List[Pad]]:
    """Group pads into nets according to the wiring pattern."""
    connectors = len(pad_grid)
    pins_per = len(pad_grid[0])

    if pattern == "bus":
        return [[pad_grid[j][i] for j in range(connectors)]
                for i in range(pins_per)]

    if pattern == "neighbor":
        return [[pad_grid[j][i], pad_grid[j + 1][i]]
                for j in range(0, connectors - 1, 2)
                for i in range(pins_per)]

    # "pairs": seeded random matching over the whole pad pool
    rng = random.Random(seed)
    pool = [pad for conn in pad_grid for pad in conn]
    rng.shuffle(pool)
    return [[pool[k], pool[k + 1]] for k in range(0, len(pool) - 1, 2)]
