"""existing_copper.py -- existing-copper obstacles for OrthoRoute (a dense mixed-signal 6-layer board local patch).

OrthoRoute 9e06f2d routes on an empty lattice: headless mode never loads the
board's existing copper, and the pad escape planner DRC-checks stubs only
against the pads in the export. For an incremental route on a board that is
already 90 % routed that guarantees shorts. This module adds both checks.

Activated by env ORTHO_OBSTACLES=<obstacles.json>, written by kicad7_to_orp2.py:
  {"clearance": mm, "track_radius": mm, "via_radius": mm,
   "lattice_layers": ["F.Cu","In2.Cu","In4.Cu","B.Cu"],
   "pad_net": {"<comp>_<pad id>": "<net>"},
   "segs":  [[layer|"*", x1, y1, x2, y2, half_width, net], ...],
   "polys": [[layer|"*", [[x, y], ...], net], ...],
   "holes": [[x, y, drill_radius], ...],          # hole-to-hole spacing (JLC 0.5 mm)
   "hole_to_hole": mm,
   "require": {"<layer>": [[[x, y], ...], ...]},  # reference-plane copper a trace must sit over
   "void":    {"<layer>": [[[x, y], ...], ...]}}  # holes in that reference copper
Layer "*" = the item occupies every copper layer (vias, PTH pads, holes, edge).

Two hooks:
  apply_to_graph(pf)   from UnifiedPathFinder._apply_keepout_obstacles:
      planar edges on layer z are blocked when either end node is within
      (item half width + clearance + track_radius) of copper on z;
      via edges at (x, y) are blocked when any copper on ANY layer is within
      (half width + clearance + via_radius) -- every via is imported as a
      through via.
  portal_ok(...)       from PadEscapePlanner._try_create_portal:
      the portal via and the F.Cu escape stub must clear other-net copper.
"""
import json
import logging
import math
import os

import numpy as np

log = logging.getLogger(__name__)
_D = None
BUCKET = 1.0  # mm


def active():
    return bool(os.environ.get("ORTHO_OBSTACLES"))


def _load():
    global _D
    if _D is not None:
        return _D
    d = json.load(open(os.environ["ORTHO_OBSTACLES"]))
    segs = d.get("segs", [])
    polys = d.get("polys", [])
    grid = {}

    def put(i, x0, y0, x1, y1):
        for bx in range(int(math.floor(x0 / BUCKET)), int(math.floor(x1 / BUCKET)) + 1):
            for by in range(int(math.floor(y0 / BUCKET)), int(math.floor(y1 / BUCKET)) + 1):
                grid.setdefault((bx, by), []).append(i)

    items = []
    for s in segs:
        L, x1, y1, x2, y2, hw, net = s
        items.append(("s", L, (x1, y1, x2, y2), hw, net))
        put(len(items) - 1, min(x1, x2) - hw, min(y1, y2) - hw, max(x1, x2) + hw, max(y1, y2) + hw)
    for p in polys:
        L, pts, net = p
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        items.append(("p", L, pts, 0.0, net))
        put(len(items) - 1, min(xs), min(ys), max(xs), max(ys))
    d["_items"] = items
    d["_grid"] = grid
    _D = d
    log.warning(f"[OBSTACLES] loaded {len(segs)} segments, {len(polys)} polygons, "
                f"{len(d.get('pad_net', {}))} pad->net entries")
    return d


def _pt_seg(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _seg_seg(a, b):
    (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = a, b

    def ccw(x1, y1, x2, y2, x3, y3):
        return (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)
    d1 = ccw(bx1, by1, bx2, by2, ax1, ay1)
    d2 = ccw(bx1, by1, bx2, by2, ax2, ay2)
    d3 = ccw(ax1, ay1, ax2, ay2, bx1, by1)
    d4 = ccw(ax1, ay1, ax2, ay2, bx2, by2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return 0.0
    return min(_pt_seg(ax1, ay1, bx1, by1, bx2, by2), _pt_seg(ax2, ay2, bx1, by1, bx2, by2),
               _pt_seg(bx1, by1, ax1, ay1, ax2, ay2), _pt_seg(bx2, by2, ax1, ay1, ax2, ay2))


def _pip(x, y, pts):
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _near(x0, y0, x1, y1, r):
    d = _load()
    seen = set()
    for bx in range(int(math.floor((min(x0, x1) - r) / BUCKET)), int(math.floor((max(x0, x1) + r) / BUCKET)) + 1):
        for by in range(int(math.floor((min(y0, y1) - r) / BUCKET)), int(math.floor((max(y0, y1) + r) / BUCKET)) + 1):
            for i in d["_grid"].get((bx, by), ()):
                if i not in seen:
                    seen.add(i)
                    yield d["_items"][i]


def seg_clear(x0, y0, x1, y1, radius, layer, net):
    """True when a track (centre x0,y0-x1,y1, half width radius) on `layer`
    ("*" = every layer) keeps `clearance` from all copper of other nets."""
    d = _load()
    c = d["clearance"]
    for kind, L, geo, hw, inet in _near(x0, y0, x1, y1, radius + c + 2.0):
        if inet == net and net is not None:
            continue
        if layer != "*" and L != "*" and L != layer:
            continue
        if kind == "s":
            if _seg_seg((x0, y0, x1, y1), geo) < hw + radius + c - 1e-9:
                return False
        else:
            if _pip(x0, y0, geo) or _pip(x1, y1, geo):
                return False
            for k in range(len(geo)):
                q0, q1 = geo[k], geo[(k + 1) % len(geo)]
                if _seg_seg((x0, y0, x1, y1), (q0[0], q0[1], q1[0], q1[1])) < radius + c - 1e-9:
                    return False
    return True


def hole_ok(x, y, r):
    """JLC hole-to-hole: drill edges must stay hole_to_hole apart."""
    d = _load()
    h2h = d.get("hole_to_hole", 0.5)
    for hx, hy, hr in d.get("holes", ()):
        if math.hypot(x - hx, y - hy) < hr + r + h2h - 1e-9:
            return False
    return True


REF_H = 0.05          # mm, resolution of the cached reference-plane bitmaps
_REF = {}


def _ref_bitmap(layer):
    """Rasterise this layer's reference-plane copper ONCE (fills minus their
    holes) at REF_H. Point-in-polygon over ~150 outlines and thousands of
    antipad holes per query was the exporter's bottleneck."""
    if layer in _REF:
        return _REF[layer]
    d = _load()
    req = d.get("require", {}).get(layer)
    if not req:
        _REF[layer] = None
        return None
    x0, y0, x1, y1 = d["bounds"]
    nx = int((x1 - x0) / REF_H) + 2
    ny = int((y1 - y0) / REF_H) + 2

    # PIL's scanline fill is C; the GND plane is ONE 85k-point outline whose
    # antipads are keyhole cut-ins, so even-odd filling reproduces the voids.
    from PIL import Image, ImageDraw
    img = Image.new("1", (ny, nx), 0)
    drw = ImageDraw.Draw(img)
    for poly in req:
        drw.polygon([((q[1] - y0) / REF_H, (q[0] - x0) / REF_H) for q in poly], fill=1)
    for poly in d.get("void", {}).get(layer, ()):
        drw.polygon([((q[1] - y0) / REF_H, (q[0] - x0) / REF_H) for q in poly], fill=0)
    m = np.array(img, dtype=bool)
    # Shrink by the track half width (and one cell for the raster edge): a
    # trace needs reference copper under its whole width, and a boundary cell
    # rounded outward would otherwise claim copper KiCad does not have there.
    from scipy import ndimage
    erode_cells = int(math.ceil(d["track_radius"] / REF_H)) + 1
    m = ndimage.binary_erosion(m, np.ones((3, 3), dtype=bool), iterations=erode_cells)
    log.warning(f"[OBSTACLES] reference bitmap {layer}: {100 * m.mean():.1f}% copper "
                f"({nx}x{ny} at {REF_H} mm)")
    _REF[layer] = (x0, y0, m)
    return _REF[layer]


def over_reference(x, y, layer):
    """True when (x, y) sits over this layer's reference-plane copper."""
    r = _ref_bitmap(layer)
    if r is None:
        return True
    x0, y0, m = r
    i = int(round((x - x0) / REF_H))
    j = int(round((y - y0) / REF_H))
    if not (0 <= i < m.shape[0] and 0 <= j < m.shape[1]):
        return False
    return bool(m[i, j])


def stub_ok(x0, y0, x1, y1, layer):
    n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 0.1))
    return all(over_reference(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, layer) for i in range(n + 1))


def pad_net(pad_id):
    return _load().get("pad_net", {}).get(pad_id)


def portal_ok(pad_id, via_x, via_y, segments):
    d = _load()
    net = pad_net(pad_id)
    if not seg_clear(via_x, via_y, via_x, via_y, d["via_radius"], "*", net):
        return False
    if not hole_ok(via_x, via_y, d.get("via_drill_radius", 0.125)):
        return False
    for (sx, sy), (ex, ey) in segments:
        if not seg_clear(sx, sy, ex, ey, d["track_radius"], "F.Cu", net):
            return False
        if not stub_ok(sx, sy, ex, ey, "F.Cu"):
            return False
    return True


def _pip_np(gx, gy, pts):
    """Vectorised even-odd point-in-polygon over grid arrays."""
    inside = np.zeros(gx.shape, dtype=bool)
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if yi != yj:
            cross = ((yi > gy) != (yj > gy)) & (gx < (xj - xi) * (gy - yi) / (yj - yi) + xi)
            inside ^= cross
        j = i
    return inside


def _raster(Nx, Ny, x0, y0, pitch, layer_filter, extra):
    """Boolean (Nx, Ny) mask of lattice nodes within hw+clearance+extra of copper."""
    d = _load()
    c = d["clearance"]
    m = np.zeros((Nx, Ny), dtype=bool)
    xs = x0 + np.arange(Nx) * pitch
    ys = y0 + np.arange(Ny) * pitch
    for kind, L, geo, hw, net in d["_items"]:
        if not layer_filter(L):
            continue
        r = hw + c + extra
        if kind == "s":
            ax, ay, bx, by = geo
            bb = (min(ax, bx) - r, min(ay, by) - r, max(ax, bx) + r, max(ay, by) + r)
        else:
            px = [q[0] for q in geo]
            py = [q[1] for q in geo]
            bb = (min(px) - r, min(py) - r, max(px) + r, max(py) + r)
        i0 = max(0, int(math.ceil((bb[0] - x0) / pitch)))
        i1 = min(Nx - 1, int(math.floor((bb[2] - x0) / pitch)))
        j0 = max(0, int(math.ceil((bb[1] - y0) / pitch)))
        j1 = min(Ny - 1, int(math.floor((bb[3] - y0) / pitch)))
        if i0 > i1 or j0 > j1:
            continue
        gx, gy = np.meshgrid(xs[i0:i1 + 1], ys[j0:j1 + 1], indexing="ij")
        if kind == "s":
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            if L2 == 0:
                dist = np.hypot(gx - ax, gy - ay)
            else:
                t = np.clip(((gx - ax) * dx + (gy - ay) * dy) / L2, 0.0, 1.0)
                dist = np.hypot(gx - (ax + t * dx), gy - (ay + t * dy))
            m[i0:i1 + 1, j0:j1 + 1] |= dist < r
        else:
            hit = _pip_np(gx, gy, geo)
            # grow by r: also block nodes within r of the outline
            dmin = np.full(gx.shape, np.inf)
            for k in range(len(geo)):
                qa, qb = geo[k], geo[(k + 1) % len(geo)]
                ex, ey = qb[0] - qa[0], qb[1] - qa[1]
                E2 = ex * ex + ey * ey
                if E2 == 0:
                    continue
                t = np.clip(((gx - qa[0]) * ex + (gy - qa[1]) * ey) / E2, 0.0, 1.0)
                dmin = np.minimum(dmin, np.hypot(gx - (qa[0] + t * ex), gy - (qa[1] + t * ey)))
            m[i0:i1 + 1, j0:j1 + 1] |= hit | (dmin < r)
    return m


def _require_mask_win(X, Y, layer):
    """Reference-plane mask over an arbitrary grid, read from the cached bitmap."""
    r = _ref_bitmap(layer)
    if r is None:
        return np.ones(X.shape, dtype=bool)
    x0, y0, m = r
    i = np.rint((X - x0) / REF_H).astype(np.int64)
    j = np.rint((Y - y0) / REF_H).astype(np.int64)
    ok = (i >= 0) & (i < m.shape[0]) & (j >= 0) & (j < m.shape[1])
    out = np.zeros(X.shape, dtype=bool)
    out[ok] = m[i[ok], j[ok]]
    return out


def _require_mask(Nx, Ny, x0, y0, pitch, layer):
    """False where this layer has no reference-plane copper under it."""
    d = _load()
    req = d.get("require", {}).get(layer)
    if not req:
        return np.ones((Nx, Ny), dtype=bool)
    xs = x0 + np.arange(Nx) * pitch
    ys = y0 + np.arange(Ny) * pitch
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return _require_mask_win(X, Y, layer)


def _hole_mask(Nx, Ny, x0, y0, pitch, r_drill):
    d = _load()
    h2h = d.get("hole_to_hole", 0.5)
    m = np.zeros((Nx, Ny), dtype=bool)
    xs = x0 + np.arange(Nx) * pitch
    ys = y0 + np.arange(Ny) * pitch
    for hx, hy, hr in d.get("holes", ()):
        r = hr + r_drill + h2h
        i0 = max(0, int(math.ceil((hx - r - x0) / pitch)))
        i1 = min(Nx - 1, int(math.floor((hx + r - x0) / pitch)))
        j0 = max(0, int(math.ceil((hy - r - y0) / pitch)))
        j1 = min(Ny - 1, int(math.floor((hy + r - y0) / pitch)))
        if i0 > i1 or j0 > j1:
            continue
        gx, gy = np.meshgrid(xs[i0:i1 + 1], ys[j0:j1 + 1], indexing="ij")
        m[i0:i1 + 1, j0:j1 + 1] |= np.hypot(gx - hx, gy - hy) < r
    return m


def apply_to_graph(pf):
    d = _load()
    g = pf.graph
    Nx, Ny, Nz = pf._Nx, pf._Ny, pf._Nz
    plane = Nx * Ny
    geom = pf.lattice.geom
    x0, y0 = geom.grid_min_x, geom.grid_min_y
    pitch = pf.lattice.pitch
    names = d["lattice_layers"]
    assert len(names) == Nz, f"obstacles lattice_layers {names} != Nz {Nz}"
    tr, vr = d["track_radius"], d["via_radius"]
    planar = np.zeros((Nz, Nx, Ny), dtype=bool)
    for z in range(Nz):
        planar[z] = _raster(Nx, Ny, x0, y0, pitch, lambda L, n=names[z]: L == "*" or L == n, tr)
        planar[z] |= ~_require_mask(Nx, Ny, x0, y0, pitch, names[z])
    via = _raster(Nx, Ny, x0, y0, pitch, lambda L: True, vr)
    via |= _hole_mask(Nx, Ny, x0, y0, pitch, d.get("via_drill_radius", 0.125))
    # node index = z*plane + y*Nx + x  ->  flatten masks in (z, y, x) order
    pflat = planar.transpose(0, 2, 1).reshape(-1)
    vflat = np.tile(via.T.reshape(-1), Nz)
    indptr = g.indptr.get() if hasattr(g.indptr, "get") else g.indptr
    indices = g.indices.get() if hasattr(g.indices, "get") else g.indices
    src = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    dst = np.asarray(indices)
    is_via = (src // plane) != (dst // plane)
    ex = d.get("exempt") or []
    n_ex = 0
    for exy in ex:
        ix = int(round((exy[0] - x0) / pitch))
        iy = int(round((exy[1] - y0) / pitch))
        if not (0 <= ix < Nx and 0 <= iy < Ny):
            continue
        # A 3rd element names the ONE layer to free: an anchor whose escape stub
        # lies on In4 has no business unblocking F.Cu, where the copper that
        # blocks it may belong to someone else.
        zs_free = range(Nz)
        if len(exy) >= 3 and exy[2] in names:
            zs_free = (names.index(exy[2]),)
        for z in zs_free:
            n = z * plane + iy * Nx + ix
            if pflat[n] or vflat[n]:
                n_ex += 1
            pflat[n] = False
            vflat[n] = False
    bn = d.get("block_nodes") or []
    n_bn = 0
    for b3 in bn:
        ix = int(round((b3[0] - x0) / pitch))
        iy = int(round((b3[1] - y0) / pitch))
        if not (0 <= ix < Nx and 0 <= iy < Ny):
            continue
        zs_blk = range(Nz)
        if len(b3) >= 3 and b3[2] in names:
            zs_blk = (names.index(b3[2]),)
        for z in zs_blk:
            n = z * plane + iy * Nx + ix
            if not pflat[n]:
                n_bn += 1
            pflat[n] = True
    if bn:
        log.warning(f"[OBSTACLES] blocked {len(bn)} anchor-halo nodes "
                    f"({n_bn} newly blocked) so no other net's track lands "
                    f"0.40 mm from an anchor via")
    if ex:
        log.warning(f"[OBSTACLES] exempted {len(ex)} seed sites "
                    f"({n_ex} blocked node-layers freed)")
    blk = np.where(is_via, vflat[src] | vflat[dst], pflat[src] | pflat[dst])
    bc = g.base_costs
    gpu = hasattr(bc, "get")
    bcc = bc.get() if gpu else bc
    bcc[blk] = 1e9
    if gpu:
        import cupy as cp
        g.base_costs = cp.asarray(bcc)
    else:
        g.base_costs = bcc
    g.blocked_edges = bcc >= 1e8
    # EdgeAccountant.update_costs is what turns these into inf.
    # It has no reference to the graph, so hand it the mask directly.
    _acct = getattr(pf, "accounting", None)
    if _acct is not None:
        _acct._blocked_edges = blk
        if os.environ.get("ORTHO_VIA_HALO", "0") == "1":
            # A 0.60 mm via and another net's 0.20 mm track need 0.60 mm between
            # centres (0.30 + 0.10 + 0.20); the lattice pitch is 0.40 mm, so a
            # track on the NEXT node is legal in the model and a KiCad
            # hole_clearance error on the board (measured: 206 of them from 216
            # imported vias). Reserving the planar edges around a via's column
            # makes the negotiation see the halo it actually occupies.
            _acct._via_halo = {
                "indptr": np.asarray(indptr), "indices": np.asarray(indices),
                "src": np.asarray(src), "plane": int(plane), "Nz": int(Nz),
            }
            log.warning("[OBSTACLES] via halo reservation ENABLED")
        log.warning(f"[OBSTACLES] blocked mask handed to EdgeAccountant ({int(blk.sum())} edges -> inf)")
    else:
        log.error("[OBSTACLES] no pf.accounting -- blocked edges stay FINITE (1e9)")
    log.warning(f"[OBSTACLES] blocked {int(blk.sum())}/{len(blk)} edges "
                f"(planar nodes blocked per layer {[int(planar[z].sum()) for z in range(Nz)]}, "
                f"via sites blocked {int(via.sum())}/{Nx * Ny})")
