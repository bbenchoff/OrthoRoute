"""Explicit HDI stack and via-process models.

The routing graph works in copper-layer indices.  This module turns a
fabrication stack into the exact layer pairs, drill geometries, and spacing
requirements that the graph and geometry emitter must preserve.

Two topologies are expressible:

* Adjacent-span stacks (PCBWay ELIC and PCBWay mechanical).  Every legal
  graph via joins two neighbouring copper layers.  A deep hole is a run of
  adjacent hops that the KiCad exporter may consolidate.
* Core-span stacks (JLCPCB k+N+k HDI).  Laser microvias join neighbouring
  layers inside the outer build-ups only.  The N-layer core is served by one
  mechanical buried hole that spans the entire core.  Graph transitions
  inside the core all pass through one hub layer, so every core layer change
  at a site shares that single hole, and the hole's barrel occupies every
  core layer at that site.  ``physical_extent`` reports that occupancy so via
  accounting, barrel ownership, and keepouts stay physically honest.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Optional, Tuple


LayerPair = Tuple[int, int]

# JLCPCB HDI ordering ceiling.  Blind/buried structures are offered up to
# this many copper layers at the time of writing.  The published HDI FAQ
# does not state the ceiling; it comes from the order form.
JLCPCB_HDI_MAX_LAYERS = 28


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
    # Core-span topology.  ``core_span`` is the one mechanical hole that
    # serves every core layer; graph transitions inside the core are hub
    # pairs (hub, z).  ``None`` keeps the adjacent-only ELIC/CNC model.
    core_span: Optional[LayerPair] = None
    core_hub_layer: Optional[int] = None
    core_process: Optional[ViaProcess] = None
    # Multiplier applied on top of the adjacent-hop via cost for each hub
    # edge.  Two hub edges make one mid-route core layer change.
    core_via_cost_scale: float = 1.0
    # Core hole site grid. 1 allows every DRC-legal lattice site; 2 keeps
    # holes on even rows and columns so every odd row and column is a
    # hole-free channel on every core layer (a classic backplane via grid).
    core_site_stride: int = 1
    vendor: str = "pcbway"
    notation_override: Optional[str] = None

    def __post_init__(self) -> None:
        if self.layer_count < 4 or self.layer_count % 2:
            raise ValueError("HDI stacks require an even layer count of at least four")
        process_pairs = frozenset(pair for pair, _ in self.via_processes)
        if self.core_span is None:
            expected = frozenset(
                (layer, layer + 1) for layer in range(self.layer_count - 1)
            )
            if self.allowed_via_spans != expected:
                raise ValueError(
                    "ELIC stack must explicitly define every adjacent layer pair"
                )
            if process_pairs != expected:
                raise ValueError("every legal via span requires one process geometry")
            return

        core_lo, core_hi = self.core_span
        if not 0 <= core_lo < core_hi < self.layer_count:
            raise ValueError(
                f"core span {self.core_span} is outside {self.name}"
            )
        if (
            self.core_hub_layer is None
            or not core_lo <= self.core_hub_layer <= core_hi
        ):
            raise ValueError(
                "core-span stacks need a hub layer inside the core"
            )
        if self.core_process is None:
            raise ValueError("core-span stacks need a core via process")
        expected = self._expected_core_span_pairs()
        if self.allowed_via_spans != expected:
            raise ValueError(
                "core-span stack must define exactly the build-up adjacent "
                "pairs plus the core hub pairs"
            )
        if process_pairs != expected:
            raise ValueError("every legal via span requires one process geometry")

    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------
    @property
    def is_core_span_stack(self) -> bool:
        return self.core_span is not None

    def _expected_core_span_pairs(self) -> FrozenSet[LayerPair]:
        core_lo, core_hi = self.core_span
        hub = int(self.core_hub_layer)
        pairs = {(layer, layer + 1) for layer in range(0, core_lo)}
        pairs |= {
            (layer, layer + 1)
            for layer in range(core_hi, self.layer_count - 1)
        }
        pairs |= {
            canonical_pair(hub, layer)
            for layer in range(core_lo, core_hi + 1)
            if layer != hub
        }
        return frozenset(pairs)

    def is_core_layer(self, layer: int) -> bool:
        if self.core_span is None:
            return False
        core_lo, core_hi = self.core_span
        return core_lo <= int(layer) <= core_hi

    def is_hub_pair(self, a: int, b: int) -> bool:
        """Return whether one graph pair rides the shared core hole."""
        if self.core_span is None:
            return False
        lo, hi = canonical_pair(int(a), int(b))
        core_lo, core_hi = self.core_span
        return core_lo <= lo and hi <= core_hi

    @property
    def notation(self) -> str:
        if self.notation_override:
            return self.notation_override
        return f"{self.build_up_steps}+2+{self.build_up_steps}"

    @property
    def process_by_span(self) -> Dict[LayerPair, ViaProcess]:
        return dict(self.via_processes)

    def process_for_span(self, a: int, b: int) -> ViaProcess:
        pair = canonical_pair(int(a), int(b))
        process = self.process_by_span.get(pair)
        if process is not None:
            return process
        if self.core_span is not None:
            # Any transition that touches the core is realised by the core
            # hole; report the largest hole for provisional geometry.
            spans = self.physical_spans(*pair)
            if self.core_span in spans:
                return self.core_process
            if spans:
                return self.process_by_span[spans[0]]
        raise ValueError(
            f"via span {pair} is not legal for {self.name}"
        )

    def physical_spans(self, a: int, b: int) -> Tuple[LayerPair, ...]:
        """Physical holes realising one vertical run, ordered top-down."""
        a = int(a)
        b = int(b)
        if not 0 <= a < self.layer_count or not 0 <= b < self.layer_count:
            raise ValueError(f"layer span {(a, b)} is outside {self.name}")
        lo, hi = canonical_pair(a, b)
        if lo == hi:
            return ()
        if self.core_span is None:
            return tuple((layer, layer + 1) for layer in range(lo, hi))
        core_lo, core_hi = self.core_span
        spans = [
            (layer, layer + 1)
            for layer in range(lo, min(hi, core_lo))
        ]
        if lo < core_hi and hi > core_lo:
            spans.append((core_lo, core_hi))
        spans.extend(
            (layer, layer + 1)
            for layer in range(max(lo, core_hi), hi)
        )
        return tuple(spans)

    def expand_span(self, a: int, b: int) -> Tuple[LayerPair, ...]:
        """Expand a vertical run into legal physical vias, in travel order."""
        spans = self.physical_spans(a, b)
        if int(a) > int(b):
            return tuple((high, low) for low, high in reversed(spans))
        return spans

    def is_physical_span(self, a: int, b: int) -> bool:
        """Return whether one emitted via is itself a legal drilled hole."""
        pair = canonical_pair(int(a), int(b))
        if pair[0] == pair[1]:
            return False
        try:
            return self.physical_spans(*pair) == (pair,)
        except ValueError:
            return False

    def physical_extent(self, a: int, b: int) -> LayerPair:
        """Copper layers physically occupied by one vertical run's holes."""
        spans = self.physical_spans(a, b)
        if not spans:
            layer = int(a)
            return (layer, layer)
        return (
            min(low for low, _ in spans),
            max(high for _, high in spans),
        )

    # ------------------------------------------------------------------
    # Geometry and cost tables consumed by the graph
    # ------------------------------------------------------------------
    def center_spacing_by_span(self) -> Dict[LayerPair, float]:
        """Minimum center spacing implied by copper and finished-hole rules."""
        return {
            pair: max(
                process.diameter_mm + self.clearance_mm,
                process.drill_mm + self.hole_edge_spacing_mm,
            )
            for pair, process in self.via_processes
        }

    def site_policy_by_span(self) -> Dict[LayerPair, Tuple[str, int]]:
        """Lattice site policy overrides for graph pairs that ride the core hole."""
        stride = int(self.core_site_stride)
        if self.core_span is None or stride <= 1:
            return {}
        return {
            pair: ("stride", stride)
            for pair, _ in self.via_processes
            if self.is_hub_pair(*pair)
        }

    def via_cost_scale_by_span(
        self, adjacent_step_scale: float
    ) -> Dict[LayerPair, float]:
        """Multiplier on the base via cost for every legal graph pair."""
        scale = float(adjacent_step_scale)
        return {
            pair: (
                scale * float(self.core_via_cost_scale)
                if self.is_hub_pair(*pair) else scale
            )
            for pair, _ in self.via_processes
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


def pcbway_mechanical_stack(
    layer_count: int,
    *,
    drill_mm: float = 0.15,
    annular_ring_mm: float = 0.0762,
) -> HDIStack:
    """Build an adjacent-via cost comparison using CNC geometry.

    This keeps the ELIC route topology for a direct capacity comparison,
    but every transition uses the larger mechanical blind/buried geometry.
    A fabricator must still define and approve the actual lamination and
    drill sequence.
    """
    layer_count = int(layer_count)
    if layer_count < 4 or layer_count % 2:
        raise ValueError(
            "PCBWay mechanical target must have an even layer count >= 4"
        )

    core_pair = (layer_count // 2 - 1, layer_count // 2)
    allowed = frozenset(
        (layer, layer + 1) for layer in range(layer_count - 1)
    )
    mechanical = ViaProcess(
        name="mechanical_blind_buried",
        drill_mm=float(drill_mm),
        diameter_mm=float(drill_mm + 2.0 * annular_ring_mm),
        kind="blind_buried",
    )
    build_up_steps = (layer_count - 2) // 2
    return HDIStack(
        name=f"pcbway_mechanical_adjacent_{layer_count}L",
        layer_count=layer_count,
        build_up_steps=build_up_steps,
        core_pair=core_pair,
        allowed_via_spans=allowed,
        via_processes=tuple(
            (pair, mechanical) for pair in sorted(allowed)
        ),
    )


def core_span_hdi_stack(
    layer_count: int,
    *,
    vendor: str,
    build_up_steps: int,
    max_build_up_steps: int,
    max_layer_count: int,
    laser_drill_mm: float,
    laser_annular_ring_mm: float,
    core_drill_mm: float,
    core_annular_ring_mm: float,
    hole_edge_spacing_mm: float,
    hole_to_conductor_mm: float,
    core_via_cost_scale: float = 2.0,
    core_site_stride: int = 1,
) -> HDIStack:
    """Build any k+N+k stack: laser build-ups over one whole-core hole."""
    layer_count = int(layer_count)
    build_up_steps = int(build_up_steps)
    if not 1 <= build_up_steps <= int(max_build_up_steps):
        raise ValueError(
            f"{vendor} offers 1..{max_build_up_steps} sequential build-ups"
        )
    if layer_count % 2 or layer_count < 2 * build_up_steps + 2:
        raise ValueError(
            f"{vendor} HDI target must have an even layer count of at "
            f"least {2 * build_up_steps + 2} for {build_up_steps} build-ups"
        )
    if layer_count > int(max_layer_count):
        raise ValueError(
            f"{vendor} HDI target exceeds the {max_layer_count}-layer ceiling"
        )
    if int(core_site_stride) < 1:
        raise ValueError("core_site_stride must be at least 1")

    core_lo = build_up_steps
    core_hi = layer_count - 1 - build_up_steps
    hub = core_lo
    laser = ViaProcess(
        name="laser_microvia",
        drill_mm=float(laser_drill_mm),
        diameter_mm=float(laser_drill_mm + 2.0 * laser_annular_ring_mm),
        kind="microvia",
    )
    core = ViaProcess(
        name="mechanical_buried_core",
        drill_mm=float(core_drill_mm),
        diameter_mm=float(core_drill_mm + 2.0 * core_annular_ring_mm),
        kind="buried",
    )
    laser_pairs = {(layer, layer + 1) for layer in range(0, core_lo)}
    laser_pairs |= {
        (layer, layer + 1) for layer in range(core_hi, layer_count - 1)
    }
    hub_pairs = {
        canonical_pair(hub, layer)
        for layer in range(core_lo, core_hi + 1)
        if layer != hub
    }
    processes = [(pair, laser) for pair in sorted(laser_pairs)]
    processes.extend((pair, core) for pair in sorted(hub_pairs))
    core_layers = core_hi - core_lo + 1
    notation = f"{build_up_steps}+{core_layers}+{build_up_steps}"
    return HDIStack(
        name=f"{vendor}_hdi_{notation}",
        layer_count=layer_count,
        build_up_steps=build_up_steps,
        core_pair=(core_lo, core_hi),
        allowed_via_spans=frozenset(laser_pairs | hub_pairs),
        via_processes=tuple(processes),
        track_width_mm=0.1,
        clearance_mm=0.1,
        hole_to_conductor_mm=float(hole_to_conductor_mm),
        hole_edge_spacing_mm=float(hole_edge_spacing_mm),
        core_span=(core_lo, core_hi),
        core_hub_layer=hub,
        core_process=core,
        core_via_cost_scale=float(core_via_cost_scale),
        core_site_stride=int(core_site_stride),
        vendor=vendor,
        notation_override=notation,
    )


def pcbway_hdi_stack(
    layer_count: int,
    *,
    build_up_steps: int = 6,
    core_via_cost_scale: float = 2.0,
    core_site_stride: int = 1,
) -> HDIStack:
    """PCBWay k+N+k HDI with laser build-ups over one whole-core hole.

    PCBWay's HDI page lists builds through 6+N+6 (evaluation required at
    order six and above). Geometry follows the PCBWay ELIC profile: 0.10 mm
    laser holes with a 3 mil ring, 0.20 mm mechanical core hole, 11 mil
    finished-hole spacing, 6 mil hole-to-conductor. Six laser layers per
    side carry escape fan-out and much of the routing without core holes,
    which is the structural fix the 28-layer JLCPCB experiments point to.
    """
    return core_span_hdi_stack(
        layer_count,
        vendor="pcbway",
        build_up_steps=build_up_steps,
        max_build_up_steps=6,
        max_layer_count=32,
        laser_drill_mm=0.1,
        laser_annular_ring_mm=0.0762,
        core_drill_mm=0.2,
        core_annular_ring_mm=0.0762,
        hole_edge_spacing_mm=0.2794,
        hole_to_conductor_mm=0.1524,
        core_via_cost_scale=core_via_cost_scale,
        core_site_stride=core_site_stride,
    )


def jlcpcb_hdi_stack(
    layer_count: int,
    *,
    build_up_steps: int = 3,
    laser_drill_mm: float = 0.1,
    laser_annular_ring_mm: float = 0.075,
    core_drill_mm: float = 0.15,
    core_annular_ring_mm: float = 0.075,
    core_via_cost_scale: float = 2.0,
    core_site_stride: int = 1,
    max_layer_count: int = JLCPCB_HDI_MAX_LAYERS,
) -> HDIStack:
    """Build a JLCPCB k+N+k HDI stack (1st- to 3rd-order HDI).

    JLCPCB published HDI capability (help article "HDI PCB Capabilities &
    FAQ", read 2026-09-16):

    * Blind vias are laser drilled, 0.075-0.15 mm finished hole (0.1 mm
      default), pad at least 0.15 mm larger than the hole (0.075 mm ring),
      dielectric-to-hole aspect ratio <= 1:1, copper filled and planarised
      so stacked microvias are allowed.
    * Buried vias in the core are mechanical, resin plugged and capped,
      aspect ratio <= 16:1 with a 0.15 mm minimum hole on boards up to
      2.4 mm thick.
    * 1st/2nd/3rd order means one, two, or three sequential lamination and
      laser drilling cycles per side, i.e. 1+N+1, 2+N+2, 3+N+3.  There is no
      arbitrary mechanical blind/buried span: a core hole spans the whole
      core.
    * Hole-to-hole spacing between different nets >= 0.24 mm; 3/3 mil
      trace/space is available with 3.5/3.5 mil recommended.  OrthoRoute
      keeps 4/4 mil for yield.

    JLCPCB engineering confirmed on 2026-09-16 (see
    docs/jlcpcb_hdi_stack.md): the core cannot be split into sub-cores,
    controlled-depth mechanical blind vias need >1 mm holes and do not
    apply, stacked 0.10/0.25 mm laser vias L1-L4 are fine, and blind/buried
    hole-wall spacing between nets is 8 mil minimum, 10 mil recommended.

    The routing graph exposes laser pairs inside each build-up and hub pairs
    ``(core_top, z)`` for every core layer ``z``.  A mid-route core layer
    change therefore costs two hub edges and one physical hole whose barrel
    occupies every core layer at the site.
    """
    return core_span_hdi_stack(
        layer_count,
        vendor="jlcpcb",
        build_up_steps=build_up_steps,
        max_build_up_steps=3,
        max_layer_count=max_layer_count,
        laser_drill_mm=laser_drill_mm,
        laser_annular_ring_mm=laser_annular_ring_mm,
        core_drill_mm=core_drill_mm,
        core_annular_ring_mm=core_annular_ring_mm,
        hole_edge_spacing_mm=0.24,
        hole_to_conductor_mm=0.2,
        core_via_cost_scale=core_via_cost_scale,
        core_site_stride=core_site_stride,
    )


def jlcpcb_adjacent_stack(
    layer_count: int,
    *,
    drill_mm: float = 0.15,
    annular_ring_mm: float = 0.075,
) -> HDIStack:
    """Comparison-only adjacent-span model with JLCPCB via geometry.

    This is NOT an orderable JLCPCB structure.  It keeps the PCBWay
    any-adjacent-span topology while using the JLCPCB 0.15 mm mechanical
    hole, 0.075 mm ring, and 0.24 mm hole spacing.  Routing it alongside the
    real ``jlcpcb_hdi_stack`` isolates how much capacity the whole-core
    buried hole costs at the same layer count.
    """
    layer_count = int(layer_count)
    if layer_count < 4 or layer_count % 2:
        raise ValueError(
            "JLCPCB comparison target must have an even layer count >= 4"
        )
    core_pair = (layer_count // 2 - 1, layer_count // 2)
    allowed = frozenset(
        (layer, layer + 1) for layer in range(layer_count - 1)
    )
    mechanical = ViaProcess(
        name="mechanical_blind_buried",
        drill_mm=float(drill_mm),
        diameter_mm=float(drill_mm + 2.0 * annular_ring_mm),
        kind="blind_buried",
    )
    build_up_steps = (layer_count - 2) // 2
    return HDIStack(
        name=f"jlcpcb_adjacent_{layer_count}L_comparison_only",
        layer_count=layer_count,
        build_up_steps=build_up_steps,
        core_pair=core_pair,
        allowed_via_spans=allowed,
        via_processes=tuple(
            (pair, mechanical) for pair in sorted(allowed)
        ),
        hole_to_conductor_mm=0.2,
        hole_edge_spacing_mm=0.24,
        vendor="jlcpcb",
    )


def stack_vendor(name: Optional[str]) -> str:
    """Map a stack name recorded in geometry back to its fabricator."""
    normalized = str(name or "").strip().lower()
    if normalized.startswith("jlcpcb"):
        return "jlcpcb"
    return "pcbway"


def vendor_project_rules(vendor: str) -> Dict[str, float]:
    """KiCad project rule floors that match one fabricator envelope."""
    if str(vendor).strip().lower() == "jlcpcb":
        return {
            "min_clearance": 0.1,
            "min_hole_clearance": 0.2,
            "min_hole_to_hole": 0.24,
        }
    return {
        "min_clearance": 0.1,
        "min_hole_clearance": 0.1524,
        "min_hole_to_hole": 0.2794,
    }


# Human-readable manufacturing profiles shared by the GUI and the CLI.
# Keys are the canonical profile identifiers accepted by stack_for_profile.
MANUFACTURING_PROFILES: Tuple[Tuple[str, str], ...] = (
    ("board_default", "Board defaults (through vias)"),
    ("pcbway_mechanical", "PCBWay mechanical blind/buried (any adjacent span)"),
    ("pcbway_elic", "PCBWay ELIC microvia (any adjacent span)"),
    ("jlcpcb_3n3", "JLCPCB HDI 3+N+3 (laser build-ups, one core hole)"),
    ("jlcpcb_2n2", "JLCPCB HDI 2+N+2 (laser build-ups, one core hole)"),
    ("jlcpcb_1n1", "JLCPCB HDI 1+N+1 (laser build-ups, one core hole)"),
    ("jlcpcb_adjacent", "JLCPCB geometry, adjacent spans (bound only, not orderable)"),
    ("pcbway_6n6", "PCBWay HDI 6+N+6 (laser build-ups, one core hole; needs PCBWay review)"),
)


def stack_for_profile(
    profile: str,
    layer_count: int,
    **overrides,
) -> Optional[HDIStack]:
    """Resolve a UI/CLI fabrication profile to an explicit stack."""
    normalized = str(profile).strip().lower().replace("-", "_")
    if normalized in {"", "default", "board_default", "none"}:
        return None
    if normalized in {"pcbway_mechanical", "mechanical"}:
        return pcbway_mechanical_stack(layer_count)
    if normalized in {"pcbway_elic", "elic"}:
        return pcbway_elic_stack(layer_count)
    if normalized in {"jlcpcb", "jlcpcb_hdi", "jlcpcb_3n3", "jlcpcb_hdi_3n3"}:
        return jlcpcb_hdi_stack(layer_count, build_up_steps=3, **overrides)
    if normalized in {"jlcpcb_2n2", "jlcpcb_hdi_2n2"}:
        return jlcpcb_hdi_stack(layer_count, build_up_steps=2, **overrides)
    if normalized in {"jlcpcb_1n1", "jlcpcb_hdi_1n1"}:
        return jlcpcb_hdi_stack(layer_count, build_up_steps=1, **overrides)
    if normalized in {"jlcpcb_adjacent", "jlcpcb_comparison"}:
        return jlcpcb_adjacent_stack(layer_count)
    if normalized in {"pcbway_6n6", "pcbway_hdi", "pcbway_hdi_6n6"}:
        return pcbway_hdi_stack(layer_count, build_up_steps=6, **overrides)
    if normalized in {"pcbway_4n4", "pcbway_hdi_4n4"}:
        return pcbway_hdi_stack(layer_count, build_up_steps=4, **overrides)
    raise ValueError(f"unknown fabrication profile: {profile}")
