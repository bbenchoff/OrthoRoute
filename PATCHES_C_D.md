# Remaining Patches C & D for unified_pathfinder.py

## Patch C: Hotset incremental routing

Add this method after `_route_all`:

```python
def _build_hotset(self, tasks: Dict[str, Tuple[int, int]]) -> Set[str]:
    """Return the set of nets whose paths touch currently-overused edges."""
    # CPU views of arrays
    present = self.accounting.present.get() if self.accounting.use_gpu else self.accounting.present
    capacity = self.accounting.capacity.get() if self.accounting.use_gpu else self.accounting.capacity
    over_idx = set(map(int, np.where(present > capacity)[0]))

    if not over_idx:
        return set(tasks.keys())  # first iter or no overuse: route all

    # Build edge->nets reverse index from existing committed paths
    edge_to_nets = defaultdict(set)
    for net_id, path in self.net_paths.items():
        if not path:
            continue
        for ei in self._path_to_edges(path):
            edge_to_nets[ei].add(net_id)

    hot = set()
    for ei in over_idx:
        hot |= edge_to_nets.get(ei, set())

    # Fallback: if nothing maps, route all to avoid deadlock
    return hot or set(tasks.keys())
```

## Patch D: Geometry emission

Add these methods to PathFinderRouter `__init__`:

```python
self._geometry_payload = {'tracks': [], 'vias': []}
```

Add layer_names to config (after line ~350):

```python
layer_names: List[str] = field(default_factory=lambda: ['F.Cu', 'In1.Cu', 'In2.Cu', 'In3.Cu', 'In4.Cu', 'B.Cu'])
```

Add these helper methods before `emit_geometry`:

```python
def _segment_world(self, a_idx: int, b_idx: int, layer: int, net: str):
    ax, ay, _ = self.lattice.idx_to_coord(a_idx)
    bx, by, _ = self.lattice.idx_to_coord(b_idx)
    (ax_mm, ay_mm) = self.lattice.geom.lattice_to_world(ax, ay)
    (bx_mm, by_mm) = self.lattice.geom.lattice_to_world(bx, by)
    return {
        'net': net,
        'layer': self.config.layer_names[layer] if layer < len(self.config.layer_names) else f"L{layer}",
        'x1': ax_mm, 'y1': ay_mm, 'x2': bx_mm, 'y2': by_mm,
        'width': self.config.grid_pitch * 0.6,
    }

def _via_world(self, at_idx: int, net: str, from_layer: int, to_layer: int):
    x, y, _ = self.lattice.idx_to_coord(at_idx)
    (x_mm, y_mm) = self.lattice.geom.lattice_to_world(x, y)
    return {
        'net': net,
        'x': x_mm, 'y': y_mm,
        'from_layer': self.config.layer_names[from_layer] if from_layer < len(self.config.layer_names) else f"L{from_layer}",
        'to_layer': self.config.layer_names[to_layer] if to_layer < len(self.config.layer_names) else f"L{to_layer}",
        'diameter': self.config.grid_pitch * 1.5,
        'drill': self.config.grid_pitch * 0.8,
    }
```

Replace the entire `emit_geometry` method:

```python
def emit_geometry(self, board: Board) -> Tuple[int, int]:
    """Convert routed node paths into drawable segments and vias."""
    logger.info("[EMIT] Converting")
    tracks, vias = [], []

    for net_id, path in self.net_paths.items():
        if not path:
            continue
        run_start = path[0]
        prev = path[0]
        prev_dir = None
        prev_layer = self.lattice.idx_to_coord(prev)[2]

        for node in path[1:]:
            x0, y0, z0 = self.lattice.idx_to_coord(prev)
            x1, y1, z1 = self.lattice.idx_to_coord(node)

            if z1 != z0:
                # flush any pending straight run before via
                if prev != run_start:
                    tracks.append(self._segment_world(run_start, prev, prev_layer, net_id))
                vias.append(self._via_world(prev, net_id, z0, z1))
                run_start = node
                prev_dir = None
            else:
                dir_vec = (np.sign(x1 - x0), np.sign(y1 - y0))
                if prev_dir is None or dir_vec == prev_dir:
                    # keep extending run
                    pass
                else:
                    # direction changed: flush previous run
                    tracks.append(self._segment_world(run_start, prev, prev_layer, net_id))
                    run_start = prev
                prev_dir = dir_vec

            prev = node
            prev_layer = z1

        # flush final run
        if prev != run_start:
            tracks.append(self._segment_world(run_start, prev, prev_layer, net_id))

    self._geometry_payload = {'tracks': tracks, 'vias': vias}
    return (len(tracks), len(vias))

def get_geometry_payload(self):
    return self._geometry_payload
```
