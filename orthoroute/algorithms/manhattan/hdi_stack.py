"""Explicit HDI stack and via-process models.

The routing graph works in copper-layer indices.  This module turns a
fabrication stack into the exact layer pairs, drill geometries, and spacing
requirements that the graph and geometry emitter must preserve.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Tuple


LayerPair = Tuple[int, int]


def canonical_pair(a: int, b: int) -> LayerPair:
    """Return one ordered copper-layer pair."""
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class ViaProcess:
    """Fabrication geometry for one legal via span."""

    name: str
    drill_mm: float
    diameter_mm: float
    kind: str

    @property
    def annular_ring_mm(self) -> float:
        return 0.5 * (self.diameter_mm - self.drill_mm)


@dataclass(frozen=True)
class HDIStack:
    """A fixed copper stack and its legal physical interconnects."""

    name: str
    layer_count: int
    build_up_steps: int
    core_pair: LayerPair
    allowed_via_spans: FrozenSet[LayerPair]
    via_processes: Tuple[Tuple[LayerPair, ViaProcess], ...]
    track_width_mm: float = 0.1
    clearance_mm: float = 0.1
    hole_to_conductor_mm: float = 0.1524
    hole_edge_spacing_mm: float = 0.2794

    def __post_init__(self) -> None:
        if self.layer_count < 4 or self.layer_count % 2:
            raise ValueError("HDI stacks require an even layer count of at least four")
        expected = frozenset(
            (layer, layer + 1) for layer in range(self.layer_count - 1)
        )
        if self.allowed_via_spans != expected:
            raise ValueError(
                "ELIC stack must explicitly define every adjacent layer pair"
            )
        process_pairs = frozenset(pair for pair, _ in self.via_processes)
        if process_pairs != expected:
            raise ValueError("every legal via span requires one process geometry")

    @property
    def notation(self) -> str:
        return f"{self.build_up_steps}+2+{self.build_up_steps}"

    @property
    def process_by_span(self) -> Dict[LayerPair, ViaProcess]:
        return dict(self.via_processes)

    def process_for_span(self, a: int, b: int) -> ViaProcess:
        pair = canonical_pair(int(a), int(b))
        try:
            return self.process_by_span[pair]
        except KeyError as exc:
            raise ValueError(
                f"via span {pair} is not legal for {self.name}"
            ) from exc

    def expand_span(self, a: int, b: int) -> Tuple[LayerPair, ...]:
        """Expand a vertical run into legal adjacent physical vias."""
        a = int(a)
        b = int(b)
        if not 0 <= a < self.layer_count or not 0 <= b < self.layer_count:
            raise ValueError(f"layer span {(a, b)} is outside {self.name}")
        if a == b:
            return ()
        step = 1 if b > a else -1
        return tuple((layer, layer + step) for layer in range(a, b, step))

    def center_spacing_by_span(self) -> Dict[LayerPair, float]:
        """Minimum center spacing implied by copper and finished-hole rules."""
        return {
            pair: max(
                process.diameter_mm + self.clearance_mm,
                process.drill_mm + self.hole_edge_spacing_mm,
            )
            for pair, process in self.via_processes
        }

    def design_rules(self) -> Dict[str, float]:
        """Conservative board-wide rules; span-specific sizes remain explicit."""
        largest = max(
            (process for _, process in self.via_processes),
            key=lambda process: (process.diameter_mm, process.drill_mm),
        )
        return {
            "default_track_width": self.track_width_mm,
            "default_clearance": self.clearance_mm,
            "default_via_drill": largest.drill_mm,
            "default_via_diameter": largest.diameter_mm,
            "min_via_annular_width": min(
                process.annular_ring_mm
                for _, process in self.via_processes
            ),
            "min_hole_to_hole": self.hole_edge_spacing_mm,
            "min_hole_clearance": self.hole_to_conductor_mm,
        }


def pcbway_elic_stack(
    layer_count: int,
    *,
    microvia_drill_mm: float = 0.1,
    core_drill_mm: float = 0.2,
    annular_ring_mm: float = 0.0762,
) -> HDIStack:
    """Build the reduced-layer PCBWay ELIC family used by OrthoRoute.

    Every build-up transition is a copper-filled stacked laser microvia.
    The one central core transition is a mechanical buried interconnect.
    PCBWay must still approve the final material stack and DFM package.
    """
    layer_count = int(layer_count)
    if layer_count < 4 or layer_count % 2:
        raise ValueError("PCBWay ELIC target must have an even layer count >= 4")

    core_pair = (layer_count // 2 - 1, layer_count // 2)
    allowed = frozenset(
        (layer, layer + 1) for layer in range(layer_count - 1)
    )
    microvia = ViaProcess(
        name="laser_microvia",
        drill_mm=float(microvia_drill_mm),
        diameter_mm=float(microvia_drill_mm + 2.0 * annular_ring_mm),
        kind="microvia",
    )
    core = ViaProcess(
        name="mechanical_core",
        drill_mm=float(core_drill_mm),
        diameter_mm=float(core_drill_mm + 2.0 * annular_ring_mm),
        kind="buried",
    )
    processes: Iterable[Tuple[LayerPair, ViaProcess]] = (
        (pair, core if pair == core_pair else microvia)
        for pair in sorted(allowed)
    )
    build_up_steps = (layer_count - 2) // 2
    return HDIStack(
        name=f"pcbway_elic_{build_up_steps}+2+{build_up_steps}",
        layer_count=layer_count,
        build_up_steps=build_up_steps,
        core_pair=core_pair,
        allowed_via_spans=allowed,
        via_processes=tuple(processes),
    )
