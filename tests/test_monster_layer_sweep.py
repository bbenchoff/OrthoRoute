from pathlib import Path

from benchmarks.run_monster_layer_sweep import (
    _backfill_candidates,
    _defect_census_paths,
    _drc_counts,
    _fabrication_manifest_paths,
    _parse_candidate_layers,
    _qualification_label,
    _remaining_candidates,
)


def test_drc_counts_include_unconnected_items_in_reported_errors():
    report = {
        "violations": [
            {"severity": "error"},
            {"severity": "warning"},
            {"severity": "error"},
        ],
        "unconnected_items": [{}, {}, {}],
    }

    assert _drc_counts(report) == {
        "drc_errors": 2,
        "drc_warnings": 1,
        "unconnected_items": 3,
        "reported_errors": 5,
    }


def test_drc_counts_tolerate_missing_sections():
    assert _drc_counts({}) == {
        "drc_errors": 0,
        "drc_warnings": 0,
        "unconnected_items": 0,
        "reported_errors": 0,
    }


def test_successful_reduced_candidate_qualifies_fourteen_layers():
    selected = (16, {"status": "complete"})

    assert _remaining_candidates(16, selected, 20) == [14]


def test_successful_twenty_layer_candidate_backfills_eighteen():
    selected = (20, {"status": "complete"})

    assert _remaining_candidates(
        20,
        selected,
        32,
        candidate_layers=[24, 28, 32],
    ) == [18]


def test_incomplete_candidate_expands_layers_in_order():
    assert _remaining_candidates(16, None, 20) == [18, 20]


def test_explicit_coarse_ladder_skips_to_requested_layers():
    assert _remaining_candidates(
        16,
        None,
        32,
        candidate_layers=[20, 24, 28, 32],
    ) == [20, 24, 28, 32]


def test_coarse_ladder_backfills_minimum_even_layer_after_success():
    assert _backfill_candidates(16, 20) == [18]
    assert _backfill_candidates(20, 24) == [22]
    assert _backfill_candidates(28, 32) == [30]


def test_candidate_layer_parser_requires_ordered_even_bounded_layers():
    assert _parse_candidate_layers(
        "20,24,28,32",
        initial_layers=16,
        max_layers=32,
    ) == [20, 24, 28, 32]


def test_qualification_label_preserves_run_timestamp():
    assert _qualification_label({
        "run_name": "Backplane-16L-stuff-20260727_134657",
    }) == "QUAL-20260727_134657"


def test_deliverable_records_fabrication_manifests():
    paths = _fabrication_manifest_paths(Path("candidate.kicad_pcb"))

    assert paths == {
        "fabrication_manifest_json": "candidate-fabrication.json",
        "fabrication_manifest_csv": "candidate-fabrication.csv",
        "fabrication_manifest_markdown": "candidate-fabrication.md",
    }


def test_deliverable_records_defect_site_census():
    paths = _defect_census_paths(Path("candidate-drc.json"))

    assert paths == {
        "defect_sites_json": "candidate-drc-defect-sites.json",
        "defect_sites_csv": "candidate-drc-defect-sites.csv",
        "defect_sites_markdown": "candidate-drc-defect-sites.md",
    }
