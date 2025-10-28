#!/usr/bin/env python3
"""
Create synthetic test board for auto-configuration testing
"""

import sys
from pathlib import Path

# Add package to path
package_dir = Path(__file__).parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

from orthoroute.domain.models.board import Board, Net, Pad, Component, Coordinate
import random


def create_synthetic_board(
    net_count: int = 512,
    layer_count: int = 12,
    board_width_mm: float = 188.5,
    board_height_mm: float = 23.8,
    name: str = "Synthetic Test Board"
) -> Board:
    """Create a synthetic test board with random nets"""

    board = Board(id="synthetic", name=name)
    board.layer_count = layer_count

    # Create components and pads distributed across the board
    components = []
    pads = []
    nets = []

    print(f"Creating synthetic board: {net_count} nets, {layer_count} layers")
    print(f"Board size: {board_width_mm}mm x {board_height_mm}mm")

    # Create grid of component positions
    grid_cols = int((net_count / 2) ** 0.5)
    grid_rows = (net_count * 2) // grid_cols

    pad_id_counter = 0

    # Create components with pads
    for row in range(grid_rows):
        for col in range(grid_cols):
            if len(components) >= net_count * 2:
                break

            # Position in grid
            x = (col / max(1, grid_cols - 1)) * board_width_mm if grid_cols > 1 else board_width_mm / 2
            y = (row / max(1, grid_rows - 1)) * board_height_mm if grid_rows > 1 else board_height_mm / 2

            # Add some random jitter
            x += random.uniform(-1.0, 1.0)
            y += random.uniform(-0.5, 0.5)

            comp_id = f"U{len(components) + 1}"
            comp = Component(
                id=comp_id,
                reference=comp_id,
                value="IC",
                footprint="TestIC",
                position=Coordinate(x=x, y=y)
            )
            components.append(comp)

            # Each component gets one pad
            pad = Pad(
                id=f"pad_{pad_id_counter}",
                component_id=comp.id,
                position=Coordinate(x=x, y=y),
                layer="F.Cu",
                size=(0.5, 0.5),
                net_id=None
            )
            pads.append(pad)
            pad_id_counter += 1

    print(f"Created {len(components)} components with {len(pads)} pads")

    # Create nets by pairing pads
    for i in range(0, min(len(pads) - 1, net_count * 2), 2):
        net = Net(id=f"net_{len(nets)}", name=f"NET_{len(nets)}")

        # Assign two pads to this net
        pads[i].net_id = net.id
        pads[i + 1].net_id = net.id
        net.pad_ids = [pads[i].id, pads[i + 1].id]

        # Store pad objects in the net for routing
        net.pads = [pads[i], pads[i + 1]]

        nets.append(net)

    board.nets = nets
    board.components = components
    board.pads = pads

    # Create pad lookup dict
    board.pad_dict = {pad.id: pad for pad in pads}

    # Set board bounds based on actual pad positions
    if pads:
        min_x = min(pad.position.x for pad in pads)
        max_x = max(pad.position.x for pad in pads)
        min_y = min(pad.position.y for pad in pads)
        max_y = max(pad.position.y for pad in pads)

        # Add margin
        margin = 3.0
        board._kicad_bounds = (min_x - margin, min_y - margin, max_x + margin, max_y + margin)

    print(f"Created {len(nets)} nets")
    print(f"Board bounds: {board._kicad_bounds if hasattr(board, '_kicad_bounds') else 'None'}")

    return board


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Create synthetic test board")
    parser.add_argument('--nets', type=int, default=512, help='Number of nets')
    parser.add_argument('--layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--width', type=float, default=188.5, help='Board width (mm)')
    parser.add_argument('--height', type=float, default=23.8, help='Board height (mm)')

    args = parser.parse_args()

    board = create_synthetic_board(
        net_count=args.nets,
        layer_count=args.layers,
        board_width_mm=args.width,
        board_height_mm=args.height
    )

    print(f"\nSynthetic board created successfully!")
    print(f"  Name: {board.name}")
    print(f"  Nets: {len(board.nets)}")
    print(f"  Layers: {board.layer_count}")
    print(f"  Components: {len(board.components) if hasattr(board, 'components') else 0}")
    print(f"  Pads: {len(board.pads) if hasattr(board, 'pads') else 0}")
