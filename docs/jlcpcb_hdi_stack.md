# JLCPCB HDI stack targets

OrthoRoute's JLCPCB manufacturing profiles model the k+N+k HDI structures
JLCPCB actually sells (1st- to 3rd-order HDI), rather than the arbitrary
adjacent-span topology used by the PCBWay profiles.

| Profile | Structure at 28 layers | Graph via pairs | Orderable |
| --- | --- | --- | --- |
| `jlcpcb_3n3` | 3+22+3 | 6 laser + 21 hub | yes, pending DFM |
| `jlcpcb_2n2` | 2+24+2 | 4 laser + 23 hub | yes, pending DFM |
| `jlcpcb_1n1` | 1+26+1 | 2 laser + 25 hub | yes, pending DFM |
| `jlcpcb_adjacent` | any adjacent span | 27 adjacent | **no** (feasibility bound) |

The profiles appear in the GUI fabrication dropdown, in
`benchmarks/run_benchmark.py --hdi-stack`, and in
`benchmarks/results/monster_full_route.py` through `ORTHO_HDI_STACK` with
`ORTHO_FAB_PROFILE=jlcpcb_hdi`.

## What JLCPCB offers

Revalidated against JLCPCB's published HDI FAQ on 2026-09-16
(https://jlcpcb.com/help/article/hdi-pcb-capabilities-faq):

- Blind vias are laser drilled: 0.075-0.15 mm finished hole, 0.1 mm
  default. Pad diameter at least 0.15 mm larger than the hole (0.075 mm
  annular ring). Dielectric thickness to hole diameter at most 1:1. Blind
  vias are copper filled and planarised, so stacked microvias are allowed.
- Buried vias are mechanical, resin plugged and capped with plating.
  Aspect ratio for mechanical buried and through vias at most 16:1, with a
  0.15 mm minimum hole on HDI boards up to 2.4 mm thick.
- 1st-, 2nd-, and 3rd-order HDI mean one, two, or three lamination and
  laser drilling cycles. In other words 1+N+1, 2+N+2, and 3+N+3. There is
  no arbitrary controlled-depth mechanical blind via and no sub-core
  lamination on the published page.
- Hole-to-hole spacing between different nets at least 0.24 mm.
- 3/3 mil trace and space is available; 3.5/3.5 mil is recommended.
  OrthoRoute keeps 4/4 mil (0.1 mm) for yield.

The 28-layer ceiling for HDI orders comes from the JLCPCB order form, not
from the FAQ. JLCPCB's general rigid capability page lists 1-32 layers but
marks blind/buried vias as unsupported outside the HDI service, so obtain
written engineering confirmation before assuming a 28-layer HDI build.

Values JLCPCB does not publish for HDI and that this model sets
conservatively: hole-to-conductor clearance 0.2 mm (their standard-board
via-to-track figure).

### JLCPCB engineering answers (2026-09-16)

Asked through their support chat about a 28-layer 3+N+3 backplane; the
engineering team replied:

- **Core lamination:** the 22-layer core cannot be split into sub-cores
  with their own buried vias and an extra lamination. One press, one core
  hole. This is the constraint the core-span model encodes.
- **Controlled-depth mechanical blind vias:** offered only through a PTH
  controlled-depth blind-slot process requiring dielectric over 0.2 mm,
  holes over 1 mm, and aspect ratio at most 1. Irrelevant at this pitch;
  laser drilling is the only blind via that applies.
- **Stacked microvias:** laser vias may be stacked at the same XY, e.g.
  L1-L2 / L2-L3 / L3-L4, and a 0.10 mm hole / 0.25 mm pad laser via with
  matching pad requirements is supported.
- **Hole density:** 30,000-40,000 mechanical and 40,000 laser holes must be
  evaluated on the real design file. Different-net blind-to-blind or
  blind-to-buried hole-wall spacing at least 8 mil (0.203 mm), 10 mil
  (0.254 mm) recommended. On the 0.4 mm lattice adjacent core holes have
  0.25 mm wall spacing: above the minimum, 4 um under the recommendation.
  The model keeps the 0.24 mm rule so core holes may use every site;
  raising it to 0.254 mm would force a checkerboard and halve core via
  sites.
- **Engineering review** of the actual design files is required before a
  production order. Not yet answered: panel size (250 x 240 mm), thickness
  and dielectrics at 28 layers, and price.

## How the router models it

Copper layers are indexed from 0 (F.Cu). For 3+22+3 on 28 layers the core
is layers 3..24.

- Laser pairs `(0,1) (1,2) (2,3)` and `(24,25) (25,26) (26,27)` are
  ordinary adjacent via edges with the 0.1 mm / 0.25 mm microvia process.
- The core is served by one mechanical hole spanning layers 3..24. The
  graph exposes hub pairs `(3, z)` for every core layer `z`. A route that
  changes from core layer 10 to 15 traverses `10 -> 3 -> 15`; both hops
  are the same physical hole. The geometry emitter expands any hop that
  touches the core into that single buried via, and duplicate holes at one
  site collapse in the existing HDI dedupe.
- `HDIStack.physical_extent(a, b)` reports the copper layers a hop's
  hole(s) occupy. The router's `_via_physical_extent` feeds that into
  barrel ownership, via keepouts, via column/segment pooling, the via edge
  metadata used by the CUDA penalty and hard-block kernels, and portal
  clearance. A core hop therefore blocks every core layer at its site,
  which is the physical truth of a 22-layer buried via.
- Hub edges cost `via_cost * adjacent_via_step_scale * core_via_cost_scale`
  each (default `core_via_cost_scale = 2.0`, override with
  `ORTHO_CORE_VIA_SCALE`). Two hub edges make one mid-route layer change,
  so a core layer change costs four adjacent hops of the PCBWay model, and
  entering the core from the build-up costs two.
- Core via sites use every 0.4 mm lattice point: 0.30 mm pad plus 0.1 mm
  clearance equals the pitch exactly, and 0.15 mm hole plus 0.24 mm hole
  spacing is 0.39 mm. Laser sites also use every point.
- Hole placement sees the copper it would cut. The first 28-layer attempt
  diverged: exact barrel conflicts rose from 330k to 490k in twelve
  iterations because the search priced a core hole only by other holes and
  the via edge's own congestion, not by the tracks on the other 21 core
  layers at that site. `hole_track_pressure_weight` (default 1.0 for
  core-span stacks, `ORTHO_HOLE_TRACK_WEIGHT`) folds the committed
  path-node occupancy of each layer into that layer's via-segment present
  congestion, so a hub edge spanning the core is penalised by
  `pres_fac * weight * (tracks in the column)`. Holes then migrate to empty
  columns and tracks route around the holes that stay.
- Optional core hole grid. `core_site_stride=2` (`ORTHO_CORE_SITE_STRIDE`,
  sweep flag `--core-site-stride`) restricts hub edges to even rows and
  columns, so every odd row and column is a hole-free channel on every
  core layer, the way a press-fit backplane leaves routing channels between
  pins. Escape columns then stop at the hub layer and the core is entered
  only through grid-legal hub edges. A quarter of the lattice remains for
  holes, about 63,000 sites against roughly 25,000 needed. Off by default.
- Early physical cleanup. The router works barrel conflicts in large
  one-sided waves only after negotiated overuse is down to the three-edge
  tail. With whole-core holes the conflicts are almost all barrel conflicts
  while negotiated overuse tails off slowly, and the ordinary hotset cap
  (100-256 nets) moves them at about half a percent per iteration.
  `physical_cleanup_overuse_threshold` (`ORTHO_PHYSICAL_CLEANUP_THRESHOLD`)
  lets that phase start once negotiated overuse is merely at or below the
  given value. Unset keeps the ordinary tail gate.
- Planned core escapes (`planned_core_escapes`, `ORTHO_PLANNED_ESCAPES=1`).
  The escape planner already lays dense connectors out deterministically:
  one via at the exact pad x, three or four rows into the copper-free band
  beside the pad row, staggered by column. With the core grid active the
  first attempts forced that column to stop at the hub layer and fan out
  on L1-L3 to a grid-legal site, which is what saturated those layers. In
  planned mode the column drops straight into the core at its own site,
  the entry is priced as one laser column plus one core hole for every
  core layer, and each pad is pinned to its primary portal
  (`portal_candidate_count = 1`) so the escape hole field is fixed from
  iteration 1. Only mid-route corner holes are confined to the grid.
- Live hole pricing within a pass (`hole_pressure_refresh_every`,
  `ORTHO_HOLE_REFRESH=N`). Every run's first pass produced the same
  conflict count regardless of penalties, because hole-track pressure was
  computed once per iteration from the previous pass: a hole placed late in
  a pass could not see the tracks committed earlier in that pass, while a
  track placed late did see earlier holes through the owner penalty. With
  `N > 0` the via pooling penalties are rebuilt every N committed nets on
  the GPU-resident node usage and applied as a delta on top of the live
  present costs. Sixty-four is a reasonable value on the backplane.
- Escape columns reserved before routing. With planned escapes every
  pad's column is known before iteration 1, so `_reserve_planned_escape_barrels`
  marks their clearance nodes as owned (and re-marks them after every
  ownership rebuild). Without this, early nets in the first pass run
  tracks through columns where later nets' escape holes land, which is
  most of the first-pass conflict count.
- Two-row escape stagger (`dynamic_escape_stagger_rows`,
  `ORTHO_ESCAPE_STAGGER=2`). The planner staggers adjacent columns'
  escape vias by one row, so a band's two hole rows are adjacent and every
  column is blocked on every via layer: a wall for vertical tracks across
  each connector block. With a two-row stagger there is a hole-free row
  between them and a vertical track crosses with one jog. Rows two and
  three of a connector are only four lattice rows apart, so the stagger
  fits in the eight-row bands beside rows one and four and between rows
  one-two and three-four.
- Where planned core escapes apply. A hole reserves every lattice node
  strictly closer than its centre spacing. PCBWay's mechanical core hole
  needs 0.4794 mm, so its footprint includes the four 0.4 mm neighbours and
  a band of escape holes at every other column is a solid wall on every
  layer the holes span, stagger or not. Sending PCBWay escapes through the
  core therefore walls off every connector band on all 26 layers and leaves
  only the 9 mm block gaps for vertical traffic (about 2,500 crossings
  against a mid-board demand of 2,700 to 3,800); measured on the 28-layer
  board the first pass went from 170k to 217k exact conflicts with 118k node
  overuse. Keep PCBWay escapes in the build-up (the default when
  `planned_core_escapes` is off) and let only grid-site corner holes reach
  the core. JLCPCB's core hole needs exactly 0.40 mm, which excludes the
  neighbours, so planned core escapes with the two-row stagger leave a
  free jog row and are the configuration to use on the JLCPCB stack.
- One hole is charged once. A route that turns inside the core walks
  `10 -> 3 -> 15` at one site; both hops are the same drilled hole, so via
  column and segment accounting charge that column visit once, in the full
  rebuild, the incremental update, and the escape-via tracker alike. Before
  this the second hop counted the same 21 segments again and the run
  reported thousands of self-inflicted via-segment overuse.

## Why this is much harder than the PCBWay model

The qualified 32-layer PCBWay-model route uses 327 distinct via spans and
193,713 holes, 177,329 of them buried between inner layers. Drilling that
schedule would need 15 sequential lamination cycles on the top side and 13
on the bottom. No fabricator sells that build; the PCBWay profile doc calls
it a routing-feasibility profile for that reason. Only 27,547 of those
holes are legal laser microvias in a 3+N+3 structure.

Under the JLCPCB structure every escape from the surface-mount connector
pads reaches the core through a stacked laser column and one whole-core
hole, and every mid-route layer change is another whole-core hole. The
router must fan out on the build-up layers, drop into the core as few
times as possible, and route mostly planar with wrong-way jogs. Whether
8,192 nets converge at 28 layers under that constraint is an experiment,
not a given.

## Experiment log: 28 layers, 2026-09-16/17

All runs: `BackplaneBlank.kicad_pcb`, 8,192 nets, 0.4 mm lattice
(494 x 513 x 28), RTX 5080, one run at a time (two concurrent monster
routes exhaust host RAM). "Total load" is exact barrel conflicts plus
negotiated (edge + path-node) overuse.

| Run | Model | Result |
| --- | --- | --- |
| bound | `jlcpcb_adjacent` (any adjacent span, not orderable) | converging: overuse 60, exact barrel 5,965 at iteration 22, falling 15 %/iteration |
| v1 | `jlcpcb_3n3`, core scale 2 | diverged: exact barrel 330k -> 490k by iteration 12 |
| v2 | + hole-track pressure | flat at ~310k by iteration 4 |
| v3 | + core via grid (stride 2) | 221k -> 159k by iteration 16, ~1 %/iteration; physical cleanup gated behind negotiated overuse tail |
| v4/v5 | + early physical repair wave | wave works (12 %/iteration on barrels) but negotiated overuse refills 10k -> 26k; total load -2 %/iteration by iteration 15 |
| v6 | core scale 0.5, wrong-way 2 | total 225k -> 122k by iteration 7, decay 15 % -> 8 %/iteration |
| v7 | bidirectional layers | worse: iteration-2 total 449k vs 225k |
| v8 | v6 + ordinary hotset floor 1,024 | best: total 261k -> 80k over 11 iterations (2 h), decay 25 % -> 3 %/iteration, plateau forming near 70k; exact barrel 43.8k, negotiated 36k |
| v9 | v8 + pressure ramp 1.25 | tracks slightly below v8: total 102.5k at iteration 7 vs 97.4k; stopped |

Day two, PCBWay 6+16+6 (`pcbway_6n6`) and the escape-side levers:

| Run | Change | Result |
| --- | --- | --- |
| P1 | 6+16+6, v8 knobs, escapes stop in the build-up | total 247k -> 139k -> 85k -> 59k -> 45k -> 37k -> 33k -> 31k (iterations 1-8), slope decaying 44 % -> 7 % |
| P2 | + planned core escapes | identical to P1 through iteration 7 |
| P3 | + owner penalty 500 | identical through iteration 2 (first pass sees no ownership) |
| P4 | + live hole pricing every 64 nets | first pass 170k vs 184k, iteration 2 106k vs 114k |
| P5 | + reserved escape columns, two-row stagger | worse: 217k exact, 118k node overuse (bands become walls, see above) |
| P6 | P1 + live hole pricing, run long | plateau 30k at iteration 11 (18.8k barrel + 11.7k negotiated), pressure factor only 3.5 |
| P7 | P6 + pressure ramp 1.25, history gain 0.8 | same curve through iteration 5 |

The PCBWay structure halves the plateau (30k against 70k) because six laser
layers per side absorb the fan-out and much of the routing, but the
negotiation still stalls with the core at 37 % occupancy and the build-up
and hub layers at 57-68 %.

Planned escapes on the JLCPCB stack (J10: planned, reserved, two-row
stagger, owner penalty 10; J11: the same with owner penalty 500) were
worse still: first-pass exact conflicts 364k and 436k against 221k for the
plain stack, with J11 also carrying 242k node and 90k edge overuse. Fixing
16,384 escape columns before routing and making them near-hard forces the
greedy first pass into detours the board cannot absorb, on either vendor's
spacing. The switches stay in the code because they converge the small
test board cleanly, but they are not the answer for this backplane.

Per-layer occupancy in the stalled runs is the diagnostic that matters:
core layers L4-L23 at 37-45 %, laser-reachable layers L2/L3 at 62-70 %
and L25/L26 at 54-78 %, with most node conflicts on L1-L3 where 16,384
escapes fan out to grid sites. Nets avoid the core because every core
transition is a whole-core hole and every core layer carries other nets'
barrels at 25 % site density. The flexible bound model, which needs no such
holes, converges on the same lattice.

Capacity bound (`analyze_layer_capacity.py`, 28 layers): worst vertical cut
5,120 of 8,208 (62 %), worst horizontal cut 2,660 of 4,940 (54 %), node
lower bound 35 %. With the via grid, an even-column cut loses about 40 % of
the even rows on core layers, putting the worst cut near 75 %; if tracks
confine themselves to the hole-free odd rows the cut demand exceeds
capacity. The design is therefore feasible on paper only with the core
packed hard, which is exactly what PathFinder fails to do.

Conclusion for this connector (17,600 SMD pads on F.Cu at 0.4 mm pitch)
and JLCPCB's single-piece core: no configuration tried produces a
conflict-free 28-layer route; the best stalls with about 44,000 barrel
shorts and 30,000 node conflicts after two hours. This is an empirical
negative, not a proof of impossibility. Levers not yet tried: pre-planned
escape holes (assign each pad a dedicated grid site so L1-L3 fan-out is
computed, not negotiated), a 6+N+6 build (PCBWay offers it with review;
six laser layers per side would carry most of the fan-out and much of the
routing without core holes), or a fabricator that laminates the core in
two halves.

## Running it

Feasibility bound (adjacent spans, JLCPCB geometry, not orderable):

```powershell
$env:ORTHO_SOURCE_BOARD = "C:\Users\Benchoff\Documents\GitHub\ThinkinMachine\Backplane\BackplaneBlank.kicad_pcb"
$env:ORTHO_DIAGNOSTIC_LAYER_LIMIT = "28"
$env:ORTHO_DIAGNOSTIC_MAX_ITERATIONS = "200"
$env:ORTHO_DIAGNOSTIC_OWNER_PENALTY = "10"
$env:ORTHO_DIAGNOSTIC_PATH_NODE_PENALTY = "10"
$env:ORTHO_FAB_PROFILE = "jlcpcb_hdi"
$env:ORTHO_HDI_STACK = "jlcpcb_adjacent"
$env:ORTHO_DIRECTION_MODE = "guided"
$env:ORTHO_WRONG_WAY_MULTIPLIER = "4"
$env:ORTHO_GRID_PITCH = "0.4"
python benchmarks\results\monster_full_route.py
```

Real 3+22+3 structure: the same, with `ORTHO_HDI_STACK = "jlcpcb_3n3"` and
optionally `ORTHO_CORE_VIA_SCALE`.

The layer sweep supervisor takes the same choices:

```powershell
python benchmarks\run_monster_layer_sweep.py <initial-progress.json> <results-dir> <BackplaneBlank.kicad_pcb> `
    --fab-profile jlcpcb_hdi --hdi-stack jlcpcb_3n3 --core-via-scale 2 `
    --max-layers 28 --max-iterations 200 --layer-depth-bias 0
```

Benchmark metrics report `topology_valid` by asking the stack whether every
emitted via is one legal drilled hole, so the whole-core buried via counts
as valid for JLCPCB stacks while an In7-In12 hop would not.

Export and DRC the result with `benchmarks/export_kicad_board.py`; the
fabrication manifest and KiCad project rules switch to JLCPCB values when
the geometry carries a `jlcpcb_*` stack name.

## Important qualification

This model prevents the router from inventing via spans JLCPCB does not
drill, but it is not a substitute for JLCPCB stack engineering or DFM
approval. At 28 layers inside the 2.4 mm HDI thickness limit the
dielectrics are roughly 70 um; core thickness, copper weights, resin fill,
and the exact lamination sequence must be confirmed with the factory before
the selected result becomes an orderable stack.
