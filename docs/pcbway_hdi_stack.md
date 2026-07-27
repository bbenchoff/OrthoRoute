# PCBWay reduced-layer stack targets

OrthoRoute's reduced-layer fabrication target is an explicit ELIC
(Every Layer Interconnect) family:

| Copper layers | Stack notation | Central core pair |
| --- | --- | --- |
| 8 | 3+2+3 | L3-L4 |
| 10 | 4+2+4 | L4-L5 |
| 12 | 5+2+5 | L5-L6 |
| 14 | 6+2+6 | L6-L7 |

All build-up transitions are emitted as individual, copper-filled stacked
laser microvias. The central core transition is emitted as a mechanical
buried via. Consecutive graph transitions are not collapsed into an
arbitrary deep blind/buried hole.

For cost comparison, `pcbway_mechanical_stack()` preserves the same
adjacent-only topology but assigns a 0.15 mm CNC drill and 0.3024 mm pad to
every blind/buried transition. This is deliberately more restrictive on the
0.4 mm grid: the published hole spacing requires about 0.4294 mm between
centres, so adjacent grid sites cannot both carry vias on the same span.
The current full-board capacity sweep evaluates 14, 16, 18, and 20 copper
layers with this geometry. The KiCad exporter consolidates touching,
co-located, same-net adjacent steps into the deepest equivalent mechanical
blind/buried span. This reduces drill objects without changing the occupied
copper interval.

The mechanical model is a routing-feasibility profile, not a claim that
every resulting span can be fabricated in one inexpensive lamination cycle.
PCBWay must define the actual blind/buried drill spans and lamination
sequence before the selected result becomes an orderable stack.

## Routing rules

- Track width: 0.1000 mm
- Track clearance: 0.1000 mm
- Laser microvia finished hole: 0.1000 mm
- Laser microvia pad: 0.2524 mm (0.0762 mm / 3 mil annular ring)
- Central mechanical finished hole: 0.2000 mm
- Central mechanical pad: 0.3524 mm (0.0762 mm / 3 mil annular ring)
- Hole-to-conductor clearance: 0.1524 mm / 6 mil
- Finished hole edge-to-edge spacing: 0.2794 mm / 11 mil

The graph applies the resulting center spacing separately for every layer
pair. At 0.2 mm lattice pitch, laser microvias occupy a 2x2 sublattice
(0.4 mm center pitch), while the larger central mechanical process uses a
3x3 sublattice (0.6 mm center pitch). At 0.4 mm lattice pitch, laser
microvias can use every site and the core process uses a checkerboard.

These values intentionally use PCBWay's advanced-capability envelope.
Revalidated against PCBWay's published pages on 2026-07-27: PCBWay lists
0.10 mm laser blind/buried holes, 0.15 mm minimum mechanical holes, 3 mil via
annular rings, 6 mil hole-to-conductor spacing, and an 11 mil finished-hole
edge-to-edge rule for vias up to 0.45 mm. Its dedicated HDI page lists
builds through 6+N+6, with evaluation required at order six and above, while
the broader advanced-capability page lists HDI through seven steps. PCBWay
also shows a 14-layer production example using 0.15 mm laser and 0.20 mm
mechanical holes.

References:

- https://www.pcbway.com/hdi-pcb.html
- https://www.pcbway.com/advanced-pcb-capabilities.html
- https://www.pcbway.com/capabilities.html
- https://www.pcbway.com/blog/PCB_Design_Layout/How_to_Select_the_Right_HDI_Stack_Up_at_the_HDI_Design_Stage_7dcacd33.html

## Important qualification

This model prevents the router from inventing unsupported via spans, but it
is not a substitute for PCBWay stack engineering or DFM approval. Dielectric
thicknesses, copper weights, impedance requirements, resin fill, lamination
sequence, and final drill compensation must be reviewed with the fabricator.
PCBWay's own 2026 stack-selection guidance identifies lamination count as a
major cost multiplier and recommends confirming the preliminary stack with
the factory before manufacture.
