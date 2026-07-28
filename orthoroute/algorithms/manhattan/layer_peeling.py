"""Deterministic layer-pair peeling helpers for routed lattice paths.

The functions in this module are deliberately independent of a live
``UnifiedPathFinder``.  They make the destructive part of a layer peel
auditable before a new routing graph is allocated:

* rank stack-symmetric internal layer pairs by planar occupancy;
* identify exactly the nets whose planar copper is displaced;
* renumber surviving paths while compressing adjacent via chains; and
* reject any remapped path that is not legal on the reduced lattice.
"""

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from .hdi_stack import HDIStack


LatticeShape = Tuple[int, int, int]
LayerPair = Tuple[int, int]


@dataclass(frozen=True)
class PeelPlan:
    """One symmetric two-copper-layer reduction."""

    source_layers: int
    target_layers: int
    removed_layers: LayerPair
    removed_planar_occupancy: int
    planar_occupancy_by_layer: Tuple[int, ...]
    displaced_nets: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "source_layers": self.source_layers,
            "target_layers": self.target_layers,
            "removed_layers": list(self.removed_layers),
            "removed_planar_occupancy": (
                self.removed_planar_occupancy
            ),
            "planar_occupancy_by_layer": list(
                self.planar_occupancy_by_layer
            ),
            "displaced_net_count": len(self.displaced_nets),
            "displaced_nets": list(self.displaced_nets),
        }


def node_to_coord(node: int, shape: LatticeShape) -> Tuple[int, int, int]:
    """Decode the router's layer-major flat node index."""
    x_steps, y_steps, layers = map(int, shape)
    plane = x_steps * y_steps
    node = int(node)
    if node < 0 or node >= plane * layers:
        raise ValueError(f"node {node} is outside lattice shape {shape}")
    layer, local = divmod(node, plane)
    y_idx, x_idx = divmod(local, x_steps)
    return x_idx, y_idx, layer


def coord_to_node(
    x_idx: int,
    y_idx: int,
    layer: int,
    shape: LatticeShape,
) -> int:
    """Encode one coordinate using the router's layer-major layout."""
    x_steps, y_steps, layers = map(int, shape)
    x_idx = int(x_idx)
    y_idx = int(y_idx)
    layer = int(layer)
    if not 0 <= x_idx < x_steps:
        raise ValueError(f"x index {x_idx} is outside lattice shape {shape}")
    if not 0 <= y_idx < y_steps:
        raise ValueError(f"y index {y_idx} is outside lattice shape {shape}")
    if not 0 <= layer < layers:
        raise ValueError(f"layer {layer} is outside lattice shape {shape}")
    return layer * x_steps * y_steps + y_idx * x_steps + x_idx


def stack_symmetric_internal_pairs(layer_count: int) -> Tuple[LayerPair, ...]:
    """Return every two-layer peel that preserves outer copper symmetry."""
    layer_count = int(layer_count)
    if layer_count < 4 or layer_count % 2:
        raise ValueError("layer peeling requires an even stack of at least 4")
    return tuple(
        (layer, layer_count - 1 - layer)
        for layer in range(1, layer_count // 2)
    )


def _planar_layers(path: Sequence[int], shape: LatticeShape) -> set[int]:
    layers = set()
    for first, second in zip(path, path[1:]):
        x1, y1, z1 = node_to_coord(first, shape)
        x2, y2, z2 = node_to_coord(second, shape)
        if z1 == z2 and (x1 != x2 or y1 != y2):
            layers.add(z1)
    return layers


def planar_node_occupancy(
    net_paths: Mapping[str, Sequence[int]],
    shape: LatticeShape,
) -> Tuple[int, ...]:
    """Count per-net planar node uses on every layer.

    A node used twice by the same net on one layer is one occupied resource;
    the same node used by two nets counts twice.  This mirrors PathFinder's
    capacity-one node accounting while ignoring via-only transit nodes.
    """
    occupancy = [0] * int(shape[2])
    for path in net_paths.values():
        planar_nodes: Dict[int, set[int]] = {}
        for first, second in zip(path, path[1:]):
            x1, y1, z1 = node_to_coord(first, shape)
            x2, y2, z2 = node_to_coord(second, shape)
            if z1 != z2 or (x1 == x2 and y1 == y2):
                continue
            planar_nodes.setdefault(z1, set()).update(
                (int(first), int(second))
            )
        for layer, nodes in planar_nodes.items():
            occupancy[layer] += len(nodes)
    return tuple(occupancy)


def build_peel_plan(
    net_paths: Mapping[str, Sequence[int]],
    shape: LatticeShape,
) -> PeelPlan:
    """Select the least occupied stack-symmetric internal layer pair."""
    occupancy = planar_node_occupancy(net_paths, shape)
    pairs = stack_symmetric_internal_pairs(shape[2])
    removed = min(
        pairs,
        key=lambda pair: (
            occupancy[pair[0]] + occupancy[pair[1]],
            max(occupancy[pair[0]], occupancy[pair[1]]),
            pair,
        ),
    )
    removed_set = set(removed)
    displaced = tuple(sorted(
        str(net_id)
        for net_id, path in net_paths.items()
        if _planar_layers(path, shape) & removed_set
    ))
    return PeelPlan(
        source_layers=int(shape[2]),
        target_layers=int(shape[2]) - 2,
        removed_layers=removed,
        removed_planar_occupancy=(
            occupancy[removed[0]] + occupancy[removed[1]]
        ),
        planar_occupancy_by_layer=occupancy,
        displaced_nets=displaced,
    )


def remap_layer(layer: int, removed_layers: Iterable[int]) -> int:
    """Renumber one surviving layer after a peel."""
    layer = int(layer)
    removed = tuple(sorted(int(item) for item in removed_layers))
    if layer in removed:
        raise ValueError(f"removed layer {layer} has no direct mapping")
    return layer - sum(item < layer for item in removed)


def remap_surviving_path(
    path: Sequence[int],
    source_shape: LatticeShape,
    removed_layers: Iterable[int],
) -> list[int]:
    """Drop peeled via-only nodes and renumber a surviving route."""
    removed = tuple(sorted(int(item) for item in removed_layers))
    if len(removed) != 2 or len(set(removed)) != 2:
        raise ValueError("a peel must remove exactly two distinct layers")
    target_shape = (
        int(source_shape[0]),
        int(source_shape[1]),
        int(source_shape[2]) - 2,
    )
    remapped = []
    for node in path:
        x_idx, y_idx, layer = node_to_coord(node, source_shape)
        if layer in removed:
            continue
        mapped = coord_to_node(
            x_idx,
            y_idx,
            remap_layer(layer, removed),
            target_shape,
        )
        if not remapped or mapped != remapped[-1]:
            remapped.append(mapped)
    return remapped


def validate_reduced_path(
    path: Sequence[int],
    shape: LatticeShape,
    stack: HDIStack,
) -> None:
    """Raise if a remapped path is not an adjacent legal graph walk."""
    if int(shape[2]) != stack.layer_count:
        raise ValueError("lattice and fabrication stack layer counts differ")
    if not path:
        raise ValueError("surviving path is empty")
    for node in path:
        node_to_coord(node, shape)
    for first, second in zip(path, path[1:]):
        x1, y1, z1 = node_to_coord(first, shape)
        x2, y2, z2 = node_to_coord(second, shape)
        delta = (abs(x2 - x1), abs(y2 - y1), abs(z2 - z1))
        if sum(delta) != 1:
            raise ValueError(
                f"path step {first}->{second} is not lattice-adjacent"
            )
        if z1 != z2:
            stack.process_for_span(z1, z2)


def remap_surviving_paths(
    net_paths: Mapping[str, Sequence[int]],
    plan: PeelPlan,
    source_shape: LatticeShape,
    stack: HDIStack,
) -> Dict[str, list[int]]:
    """Transform and validate every non-displaced committed route."""
    if int(source_shape[2]) != plan.source_layers:
        raise ValueError("peel plan does not match source lattice")
    target_shape = (
        int(source_shape[0]),
        int(source_shape[1]),
        plan.target_layers,
    )
    displaced = set(plan.displaced_nets)
    survivors = {}
    for net_id, path in net_paths.items():
        if str(net_id) in displaced:
            continue
        # A KiCad-accepted (<100) source may still contain a small number of
        # unrouted nets.  They have no committed occupancy to preserve.
        if not path:
            continue
        remapped = remap_surviving_path(
            path,
            source_shape,
            plan.removed_layers,
        )
        validate_reduced_path(remapped, target_shape, stack)
        survivors[str(net_id)] = remapped
    return survivors


def infer_terminal_entry_layers(
    path: Sequence[int],
    shape: LatticeShape,
) -> LayerPair:
    """Infer the two terminal-via landing layers from an attached path."""
    if not path:
        raise ValueError("cannot infer portal layers from an empty path")

    def infer(nodes: Sequence[int]) -> int:
        coordinates = [node_to_coord(node, shape) for node in nodes]
        terminal_xy = coordinates[0][:2]
        previous_layer = coordinates[0][2]
        for x_idx, y_idx, layer in coordinates[1:]:
            if (x_idx, y_idx) != terminal_xy:
                return previous_layer
            previous_layer = layer
        return previous_layer

    return infer(path), infer(tuple(reversed(path)))


def rebuild_selected_portals(
    router,
    net_paths: Mapping[str, Sequence[int]],
    shape: LatticeShape,
) -> Tuple[Dict[str, tuple], Dict[str, LayerPair]]:
    """Match remapped path terminals to deterministic portal candidates.

    The old full-board artifacts did not serialize mutable selected portal
    objects.  Their terminal X/Y coordinates are nevertheless part of every
    committed path.  A freshly initialized reduced router can therefore
    recover the exact candidate and assign its new landing layer.
    """
    selected: Dict[str, tuple] = {}
    portal_layers: Dict[str, LayerPair] = {}
    for net_id, path in net_paths.items():
        pad_ids = router.net_pad_ids.get(net_id)
        if pad_ids is None or len(pad_ids) != 2:
            raise ValueError(f"net {net_id} has no two-pad portal mapping")
        layers = infer_terminal_entry_layers(path, shape)
        endpoint_coordinates = (
            node_to_coord(path[0], shape),
            node_to_coord(path[-1], shape),
        )
        matched = []
        for pad_id, coordinate, entry_layer in zip(
            pad_ids,
            endpoint_coordinates,
            layers,
        ):
            candidates = list(router.portal_candidates.get(pad_id, ()))
            primary = router.portals.get(pad_id)
            if primary is not None and all(
                candidate is not primary for candidate in candidates
            ):
                candidates.insert(0, primary)
            portal = next(
                (
                    candidate for candidate in candidates
                    if (
                        int(candidate.x_idx),
                        int(candidate.y_idx),
                    ) == coordinate[:2]
                ),
                None,
            )
            if portal is None:
                raise ValueError(
                    f"net {net_id} terminal {coordinate[:2]} does not "
                    f"match a portal candidate for {pad_id}"
                )
            matched.append(replace(
                portal,
                entry_layer=int(entry_layer),
            ))
        selected[net_id] = tuple(matched)
        portal_layers[net_id] = tuple(map(int, layers))
    return selected, portal_layers


def remap_selected_portals(
    selected_portals: Mapping[str, Sequence[object]],
    net_paths: Mapping[str, Sequence[int]],
    shape: LatticeShape,
    source_layer_count: int,
) -> Tuple[Dict[str, tuple], Dict[str, LayerPair]]:
    """Preserve serialized portal geometry while updating reduced layers."""
    selected: Dict[str, tuple] = {}
    portal_layers: Dict[str, LayerPair] = {}
    for net_id, path in net_paths.items():
        portals = selected_portals.get(net_id)
        if portals is None or len(portals) != 2:
            raise ValueError(
                f"net {net_id} has no serialized two-terminal portals"
            )
        layers = infer_terminal_entry_layers(path, shape)
        remapped = []
        for portal, entry_layer in zip(portals, layers):
            pad_layer = int(portal.pad_layer)
            if pad_layer == int(source_layer_count) - 1:
                pad_layer = int(shape[2]) - 1
            elif pad_layer not in (0,):
                raise ValueError(
                    f"net {net_id} portal pad layer {pad_layer} is not outer"
                )
            remapped.append(replace(
                portal,
                pad_layer=pad_layer,
                entry_layer=int(entry_layer),
            ))
        selected[net_id] = tuple(remapped)
        portal_layers[net_id] = tuple(map(int, layers))
    return selected, portal_layers
