# PCBWay HDI stack target

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

These values intentionally use PCBWay's advanced-capability envelope. PCBWay
currently publishes 0.10 mm laser holes, 0.15 mm minimum mechanical holes,
3 mil via annular rings, 6 mil hole-to-conductor spacing, HDI structures
through 7 build-up steps, and examples of 14-layer boards using 0.10 mm laser
and 0.20 mm mechanical holes.

References:

- https://www.pcbway.com/hdi-pcb.html
- https://www.pcbway.com/advanced-pcb-capabilities.html
- https://www.pcbway.com/pcb_prototype/What_is_HDI_PCB.html

## Important qualification

This model prevents the router from inventing unsupported via spans, but it
is not a substitute for PCBWay stack engineering or DFM approval. Dielectric
thicknesses, copper weights, impedance requirements, resin fill, lamination
sequence, and final drill compensation must be reviewed with the fabricator.
