"""Warm-start layer-pair peeling regressions."""

from types import SimpleNamespace

import pytest

from orthoroute.algorithms.manhattan.hdi_stack import (
    pcbway_mechanical_stack,
)
from orthoroute.algorithms.manhattan.layer_peeling import (
    build_peel_plan,
    coord_to_node,
    infer_terminal_entry_layers,
    node_to_coord,
    planar_node_occupancy,
    rebuild_selected_portals,
    remap_selected_portals,
    remap_surviving_path,
    remap_surviving_paths,
    stack_symmetric_internal_pairs,
    validate_reduced_path,
)
from orthoroute.algorithms.manhattan.pad_escape_planner import Portal


SHAPE = (4, 3, 8)


def node(x, y, z):
    return coord_to_node(x, y, z, SHAPE)


def test_node_coordinate_round_trip():
    for z in range(SHAPE[2]):
        for y in range(SHAPE[1]):
            for x in range(SHAPE[0]):
                assert node_to_coord(node(x, y, z), SHAPE) == (x, y, z)


def test_symmetric_pairs_exclude_outer_copper():
    assert stack_symmetric_internal_pairs(8) == (
        (1, 6),
        (2, 5),
        (3, 4),
    )


def test_planar_occupancy_counts_distinct_per_net_node_uses():
    paths = {
        "A": [node(0, 0, 1), node(1, 0, 1), node(2, 0, 1)],
        "B": [node(1, 0, 1), node(2, 0, 1)],
        "VIA": [node(3, 2, 0), node(3, 2, 1), node(3, 2, 2)],
    }

    occupancy = planar_node_occupancy(paths, SHAPE)

    assert occupancy[1] == 5
    assert sum(occupancy) == 5


def test_peel_plan_chooses_least_occupied_symmetric_pair_and_displaced_set():
    paths = {
        "L1": [node(0, 0, 1), node(1, 0, 1)],
        "L6": [node(0, 0, 6), node(1, 0, 6)],
        "L2": [node(0, 1, 2), node(1, 1, 2)],
        "VERTICAL_ONLY": [
            node(3, 2, z) for z in range(SHAPE[2])
        ],
    }

    plan = build_peel_plan(paths, SHAPE)

    assert plan.removed_layers == (3, 4)
    assert plan.removed_planar_occupancy == 0
    assert plan.displaced_nets == ()
    assert plan.as_dict()["target_layers"] == 6


def test_displaced_nets_are_exactly_planar_users_of_removed_layers():
    paths = {
        "L1": [node(0, 0, 1), node(1, 0, 1)],
        "L6": [node(0, 0, 6), node(1, 0, 6)],
        "L2": [node(0, 1, 2), node(1, 1, 2)],
        "L5": [node(0, 1, 5), node(1, 1, 5)],
        "VERTICAL_ONLY": [
            node(3, 2, z) for z in range(SHAPE[2])
        ],
    }

    plan = build_peel_plan(paths, SHAPE)

    assert plan.removed_layers == (3, 4)
    assert plan.displaced_nets == ()

    paths["L3"] = [node(0, 2, 3), node(1, 2, 3)]
    paths["L4"] = [node(0, 2, 4), node(1, 2, 4)]
    plan = build_peel_plan(paths, SHAPE)

    assert plan.removed_layers == (1, 6)
    assert plan.displaced_nets == ("L1", "L6")


def test_via_chain_compresses_across_removed_layers():
    path = [node(2, 1, z) for z in range(SHAPE[2])]

    remapped = remap_surviving_path(path, SHAPE, (2, 5))
    coords = [
        node_to_coord(item, (SHAPE[0], SHAPE[1], 6))
        for item in remapped
    ]

    assert coords == [(2, 1, z) for z in range(6)]
    validate_reduced_path(
        remapped,
        (SHAPE[0], SHAPE[1], 6),
        pcbway_mechanical_stack(6),
    )


def test_terminal_entry_layers_stop_at_first_planar_step():
    path = [
        node(0, 1, 0),
        node(0, 1, 1),
        node(0, 1, 2),
        node(1, 1, 2),
        node(1, 1, 3),
        node(1, 1, 4),
        node(1, 1, 5),
        node(2, 1, 5),
        node(2, 1, 6),
        node(2, 1, 7),
    ]

    assert infer_terminal_entry_layers(path, SHAPE) == (2, 5)


def test_selected_portals_rebuild_from_committed_terminal_coordinates():
    source_portal = Portal(
        x_idx=0, y_idx=1, pad_layer=0, delta_steps=3,
        direction=1, pad_x=0.0, pad_y=0.0,
    )
    target_portal = Portal(
        x_idx=2, y_idx=1, pad_layer=7, delta_steps=3,
        direction=-1, pad_x=1.0, pad_y=1.0,
    )
    router = SimpleNamespace(
        net_pad_ids={"NET": ("P1", "P2")},
        portal_candidates={
            "P1": [source_portal],
            "P2": [target_portal],
        },
        portals={"P1": source_portal, "P2": target_portal},
    )
    path = [
        node(0, 1, 0),
        node(0, 1, 1),
        node(0, 1, 2),
        node(1, 1, 2),
        node(2, 1, 2),
        node(2, 1, 3),
        node(2, 1, 4),
        node(2, 1, 5),
        node(2, 1, 6),
        node(2, 1, 7),
    ]

    selected, layers = rebuild_selected_portals(
        router, {"NET": path}, SHAPE
    )

    assert layers["NET"] == (2, 2)
    assert selected["NET"][0].entry_layer == 2
    assert selected["NET"][1].entry_layer == 2
    assert selected["NET"][0] is not source_portal


def test_serialized_back_portal_moves_to_new_outer_layer():
    front = Portal(
        x_idx=0, y_idx=1, pad_layer=0, delta_steps=3,
        direction=1, pad_x=0.0, pad_y=0.0,
    )
    back = Portal(
        x_idx=2, y_idx=1, pad_layer=9, delta_steps=3,
        direction=-1, pad_x=1.0, pad_y=1.0,
    )
    shape = (4, 3, 8)
    path = [
        coord_to_node(0, 1, 0, shape),
        coord_to_node(0, 1, 1, shape),
        coord_to_node(1, 1, 1, shape),
        coord_to_node(2, 1, 1, shape),
        coord_to_node(2, 1, 2, shape),
        coord_to_node(2, 1, 3, shape),
        coord_to_node(2, 1, 4, shape),
        coord_to_node(2, 1, 5, shape),
        coord_to_node(2, 1, 6, shape),
        coord_to_node(2, 1, 7, shape),
    ]

    selected, layers = remap_selected_portals(
        {"NET": (front, back)},
        {"NET": path},
        shape,
        source_layer_count=10,
    )

    assert layers["NET"] == (1, 1)
    assert selected["NET"][0].pad_layer == 0
    assert selected["NET"][1].pad_layer == 7


def test_remap_rejects_nonadjacent_original_walk():
    bad = [node(0, 0, 1), node(3, 0, 1)]

    with pytest.raises(ValueError, match="not lattice-adjacent"):
        validate_reduced_path(
            bad,
            SHAPE,
            pcbway_mechanical_stack(8),
        )


def test_bulk_remap_excludes_displaced_and_validates_survivors():
    paths = {
        "DISPLACED_A": [node(0, 0, 1), node(1, 0, 1)],
        "DISPLACED_B": [node(0, 0, 6), node(1, 0, 6)],
        "BUSY_2": [
            node(0, 1, 2), node(1, 1, 2), node(2, 1, 2)
        ],
        "BUSY_5": [
            node(0, 1, 5), node(1, 1, 5), node(2, 1, 5)
        ],
        "BUSY_3": [
            node(0, 2, 3), node(1, 2, 3), node(2, 2, 3)
        ],
        "BUSY_4": [
            node(0, 2, 4), node(1, 2, 4), node(2, 2, 4)
        ],
        "SURVIVOR": [node(2, 1, z) for z in range(SHAPE[2])],
        "UNROUTED": [],
    }
    plan = build_peel_plan(paths, SHAPE)

    survivors = remap_surviving_paths(
        paths,
        plan,
        SHAPE,
        pcbway_mechanical_stack(6),
    )

    assert set(survivors) == {
        "BUSY_2",
        "BUSY_3",
        "BUSY_4",
        "BUSY_5",
        "SURVIVOR",
    }
