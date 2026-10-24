#!/usr/bin/env python3
"""
epsmap.py - make, inspect, convert and combine dielectric maps for DelPhiPKa.

All maps hold the ABSOLUTE relative dielectric constant (water ~ 80, protein ~ 2-8, lipid core ~ 2)
on a regular axis-aligned grid, in the coordinate frame of the PDB you give DelPhiPKa.
Formats (chosen by file extension, .cube/.cub or .dx):
  cube : Gaussian/DelPhi text cube, written in Bohr (like DelPhi's "gridpoint-epsmap"), 6 values/line
  dx   : OpenDX, Angstrom, 3 values/line
Only numpy is required.

  epsmap.py info     MAP
  epsmap.py convert  IN OUT
  epsmap.py membrane OUT (--like REF | --box X0 X1 Y0 Y1 Z0 Z1 --spacing H)  [slab options]
  epsmap.py combine  PROTEIN MEMBRANE OUT [--mode blend|mask|min|max] [options]

Typical membrane workflow (membrane normal = z, membrane centre at z = 0, e.g. an OPM/PPM-oriented PDB):

  epsmap.py membrane membrane.cube --like protein.cube --center 0 --core-thickness 30 \\
            --head-thickness 8 --eps-core 2 --eps-head 30
  epsmap.py combine  protein.cube membrane.cube combined.cube --interior 8
  # run.prm:  Custom dielectric map : combined.cube   /   Custom dielectric map outside : EXTEND

The combine rule "blend" (default) is the Gaussian model's own rule with the environment generalised
from water to your membrane:   eps = f*eps_interior + (1-f)*eps_membrane(r),
with the protein occupancy f = (eps_bg - eps_protein_map) / (eps_bg - eps_interior) clipped to [0,1].
"""
import argparse
import json
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BOHR = 0.529177210903   # Angstrom per Bohr


# --------------------------------------------------------------------------------------------- map
class EpsMap:
    """data[ix,iy,iz] at (origin + (ix,iy,iz)*delta); all lengths in Angstrom."""
    def __init__(self, origin, delta, data, note=""):
        self.origin = np.asarray(origin, float)
        self.delta = np.asarray(delta, float)
        self.data = data
        self.note = note

    @property
    def n(self):
        return self.data.shape

    def axis(self, a):
        return self.origin[a] + self.delta[a] * np.arange(self.data.shape[a])

    def upper(self):
        return self.origin + self.delta * (np.array(self.n) - 1)


# ---------------------------------------------------------------------------------------------- IO

def _write_rows(f, arr, fmt1, sep=" ", rows=20000):
    """Write a 2-D array one row per line, like np.savetxt(fmt=fmt1, delimiter=sep) but ~3x faster
    (formats a block of rows with a single % operation)."""
    line = sep.join([fmt1] * arr.shape[1]) + "\n"
    blk = line * rows
    n = arr.shape[0]; full = (n // rows) * rows
    for i in range(0, full, rows):
        f.write(blk % tuple(arr[i:i + rows].ravel().tolist()))
    if full < n:
        f.write((line * (n - full)) % tuple(arr[full:].ravel().tolist()))

def read_cube(path):
    with open(path, "r") as f:
        f.readline(); note = f.readline().strip()
        t = f.readline().split()
        natoms = int(t[0]); org = [float(x) for x in t[1:4]]
        if natoms < 0:
            raise ValueError("cube files with orbital data are not supported")
        cnt, vox = [], []
        for _ in range(3):
            t = f.readline().split()
            cnt.append(int(t[0])); vox.append([float(x) for x in t[1:4]])
        if all(c > 0 for c in cnt): unit = BOHR
        elif all(c < 0 for c in cnt): unit = 1.0
        else: raise ValueError("mixed Bohr/Angstrom voxel counts")
        n = [abs(c) for c in cnt]
        for a in range(3):
            for b in range(3):
                if a != b and abs(vox[a][b]) > 1e-9:
                    raise ValueError("rotated/skewed cube grids are not supported")
        for _ in range(natoms):
            f.readline()
        data = np.fromstring(f.read(), dtype=np.float64, sep=" ")
    if data.size != n[0] * n[1] * n[2]:
        raise ValueError("%s: expected %d values, found %d" % (path, n[0] * n[1] * n[2], data.size))
    return EpsMap(np.array(org) * unit, [vox[a][a] * unit for a in range(3)],
                  data.astype(np.float32).reshape(n), note)


def read_dx(path):
    n = org = None; delta = []; data = None
    with open(path, "r") as f:
        for line in f:
            w = line.split()
            if not w or w[0].startswith("#"):
                continue
            if w[0] == "object" and "counts" in w and n is None:
                i = w.index("counts"); n = [int(x) for x in w[i + 1:i + 4]]
            elif w[0] == "origin":
                org = [float(x) for x in w[1:4]]
            elif w[0] == "delta":
                v = [float(x) for x in w[1:4]]; a = len(delta)
                if any(abs(v[b]) > 1e-12 for b in range(3) if b != a):
                    raise ValueError("rotated/skewed DX grids are not supported")
                delta.append(v[a])
            elif w[0] == "object" and "data" in w and "follows" in w:
                text = f.read()
                cuts = [i for i in (text.find("attribute"), text.find("object")) if i >= 0]
                if cuts:                      # drop the DX trailer lines after the numbers
                    text = text[:min(cuts)]
                data = np.fromstring(text, dtype=np.float64, sep=" ")
                break
    if n is None or org is None or len(delta) != 3 or data is None:
        raise ValueError("%s is not a valid OpenDX scalar grid" % path)
    tot = n[0] * n[1] * n[2]
    if data.size < tot:
        raise ValueError("%s: expected %d values, found %d" % (path, tot, data.size))
    return EpsMap(org, delta, data[:tot].astype(np.float32).reshape(n))


def read_map(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dx":
        return read_dx(path)
    if ext in (".cube", ".cub"):
        return read_cube(path)
    # unknown extension: sniff
    with open(path) as f:
        head = [f.readline() for _ in range(50)]
    return read_dx(path) if any(l.startswith("object") for l in head) else read_cube(path)


def write_cube(path, m, note="epsmap"):
    n = m.n
    ctr = m.origin + m.delta * (np.array(n) - 1) / 2.0
    with open(path, "w") as f:
        f.write("  %f  %d %f %f %f\n" % (float(m.data.min()), n[0], ctr[0], ctr[1], ctr[2]))
        f.write("Gaussian cube gridpoint-epsmap: %s (text format)\n" % note)
        f.write("%5d %16.8f %16.8f %16.8f\n" % (1, *(m.origin / BOHR)))
        for a in range(3):
            v = [0.0, 0.0, 0.0]; v[a] = m.delta[a] / BOHR
            f.write("%5d %16.8f %16.8f %16.8f\n" % (n[a], *v))
        f.write("    1       0.000000       0.000000       0.000000       0.000000\n")
        flat = m.data.ravel()
        full = (flat.size // 6) * 6
        if full:
            _write_rows(f, flat[:full].reshape(-1, 6), "%13.5e", "")
        if full < flat.size:
            f.write("".join("%13.5e" % v for v in flat[full:]) + "\n")


def write_dx(path, m, note="epsmap"):
    n = m.n
    with open(path, "w") as f:
        f.write("# %s\n" % note)
        f.write("object 1 class gridpositions counts %d %d %d\n" % tuple(n))
        f.write("origin %.9g %.9g %.9g\n" % tuple(m.origin))
        for a in range(3):
            v = [0.0, 0.0, 0.0]; v[a] = m.delta[a]
            f.write("delta %.9g %.9g %.9g\n" % tuple(v))
        f.write("object 2 class gridconnections counts %d %d %d\n" % tuple(n))
        f.write("object 3 class array type double rank 0 items %d data follows\n" % m.data.size)
        flat = m.data.ravel()
        full = (flat.size // 3) * 3
        if full:
            _write_rows(f, flat[:full].reshape(-1, 3), "%.7g", " ")
        if full < flat.size:
            f.write(" ".join("%.7g" % v for v in flat[full:]) + "\n")
        f.write('attribute "dep" string "positions"\nobject "regular positions regular connections" class field\n')


def write_map(path, m, note="epsmap"):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dx":
        write_dx(path, m, note)
    elif ext in (".cube", ".cub"):
        write_cube(path, m, note)
    else:
        sys.exit("output file must end in .cube, .cub or .dx")


# ------------------------------------------------------------------------------------ resampling
def resample(src, axes, outside="edge", fill=80.0):
    """Trilinear resampling of `src` onto the regular grid given by three 1-D coordinate arrays.
    outside='edge' : continue the nearest edge value;  'value': use `fill` outside the source box."""
    arr = src.data.astype(np.float32)
    inside = []
    for a in range(3):
        f = (np.asarray(axes[a], float) - src.origin[a]) / src.delta[a]
        nsrc = src.n[a]
        tol = 1e-6
        inside.append((f >= -tol) & (f <= nsrc - 1 + tol))
        f = np.clip(f, 0.0, nsrc - 1)
        i0 = np.minimum(np.floor(f).astype(np.int64), max(nsrc - 2, 0))
        i1 = np.minimum(i0 + 1, nsrc - 1)
        t = (f - i0).astype(np.float32)
        shp = [1, 1, 1]; shp[a] = -1
        arr = np.take(arr, i0, axis=a) * (1 - t.reshape(shp)) + np.take(arr, i1, axis=a) * t.reshape(shp)
    if outside == "value":
        mask = inside[0][:, None, None] & inside[1][None, :, None] & inside[2][None, None, :]
        arr = np.where(mask, arr, np.float32(fill))
    return arr.astype(np.float32)


# ----------------------------------------------------------------------------------- membrane slab
def _step(x, x0, w):
    """smooth 0 -> 1 step at x0 (Gaussian-smoothed, width w in Angstrom)"""
    if w <= 0:
        return (x >= x0).astype(float)
    return 0.5 * (1.0 + np.vectorize(math.erf)((x - x0) / (math.sqrt(2.0) * w)))


def slab_profile(x, centre, core, head, eps_core, eps_head, eps_water, smooth):
    """eps along the membrane normal: water | head | core | head | water"""
    hc = core / 2.0
    hh = hc + head
    core_win = _step(x, centre - hc, smooth) - _step(x, centre + hc, smooth)
    outer_win = _step(x, centre - hh, smooth) - _step(x, centre + hh, smooth)
    return eps_water + (eps_head - eps_water) * outer_win + (eps_core - eps_head) * core_win


def cmd_membrane(a):
    if a.like:
        ref = read_map(a.like)
        origin, delta, n = ref.origin.copy(), ref.delta.copy(), list(ref.n)
    elif a.box and a.spacing:
        b = a.box
        origin = np.array([b[0], b[2], b[4]], float)
        delta = np.array([a.spacing] * 3, float)
        n = [int(math.ceil((b[1] - b[0]) / a.spacing)) + 1,
             int(math.ceil((b[3] - b[2]) / a.spacing)) + 1,
             int(math.ceil((b[5] - b[4]) / a.spacing)) + 1]
    else:
        sys.exit("membrane: give either --like REFMAP or --box X0 X1 Y0 Y1 Z0 Z1 together with --spacing")

    if a.shape_file:
        import membrane_shape as msh
        shp_ = msh.load_shape(a.shape_file)
        meta = json.load(open(a.shape_file))
        core = a.core_thickness if a.core_thickness is not None else meta.get("core_thickness", 30.0)
        d = msh.distance_on_grid(shp_, origin, delta, n)
        data = slab_profile(d.astype(float), 0.0, core, a.head_thickness, a.eps_core, a.eps_head, a.eps_water, a.smooth).astype(np.float32)
        write_map(a.out, EpsMap(origin, delta, data), "curved membrane (%s), core=%gA head=%gA eps core/head/water=%g/%g/%g"
                  % (shp_.kind, core, a.head_thickness, a.eps_core, a.eps_head, a.eps_water))
        dout = a.distance_out or (os.path.splitext(a.out)[0] + "_dist.dx")
        write_map(dout, EpsMap(origin, delta, d), "signed distance to the membrane mid-surface (A), kind=%s" % shp_.kind)
        print("wrote %s and %s  grid %s  spacing %s A" % (a.out, dout, tuple(n), tuple(float(round(x, 4)) for x in delta)))
        print("  %s" % shp_.describe())
        print("  core |d| < %g A (eps %g), head zone %g A each side (eps %g), water eps %g, smoothing %g A; d range %.1f .. %.1f A"
              % (core / 2, a.eps_core, a.head_thickness, a.eps_head, a.eps_water, a.smooth, float(d.min()), float(d.max())))
        return

    if a.core_thickness is None:
        a.core_thickness = 30.0
    ax = "xyz".index(a.axis)
    z = origin[ax] + delta[ax] * np.arange(n[ax])
    prof = slab_profile(z, a.center, a.core_thickness, a.head_thickness,
                        a.eps_core, a.eps_head, a.eps_water, a.smooth).astype(np.float32)
    shp = [1, 1, 1]; shp[ax] = -1
    data = np.broadcast_to(prof.reshape(shp), n).copy()
    m = EpsMap(origin, delta, data)
    write_map(a.out, m, "membrane slab, normal=%s centre=%g core=%gA head=%gA eps core/head/water=%g/%g/%g"
              % (a.axis, a.center, a.core_thickness, a.head_thickness, a.eps_core, a.eps_head, a.eps_water))
    print("wrote %s  grid %s  spacing %s A" % (a.out, tuple(n), tuple(float(round(x, 4)) for x in delta)))
    print("  profile along %s: core |%s-%g| < %g A (eps %g), head zone %g A each side (eps %g), water eps %g, smoothing %g A"
          % (a.axis, a.axis, a.center, a.core_thickness / 2, a.eps_core, a.head_thickness, a.eps_head, a.eps_water, a.smooth))
    if not a.like:
        print("  NOTE: a slab-only map should be fed to 'combine' (it extends its edge values), or used with"
              " 'Custom dielectric map outside : EXTEND'.")


# --------------------------------------------------------------------------------------- combine

# ------------------------------------------------------------------------------------ pore / cavity water
def _shift_or(a):
    """6-neighbour dilation of a boolean array."""
    b = a.copy()
    b[1:, :, :] |= a[:-1, :, :]; b[:-1, :, :] |= a[1:, :, :]
    b[:, 1:, :] |= a[:, :-1, :]; b[:, :-1, :] |= a[:, 1:, :]
    b[:, :, 1:] |= a[:, :, :-1]; b[:, :, :-1] |= a[:, :, 1:]
    return b


def _label_components_2d(mask):
    """Connected components (4-neighbour) of a 2-D boolean array; returns (labels, n)."""
    lab = np.zeros(mask.shape, dtype=np.int32)
    n = 0
    for i0, j0 in zip(*np.nonzero(mask)):
        if lab[i0, j0]:
            continue
        n += 1
        stack = [(i0, j0)]
        lab[i0, j0] = n
        while stack:
            i, j = stack.pop()
            for ii, jj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= ii < mask.shape[0] and 0 <= jj < mask.shape[1] and mask[ii, jj] and not lab[ii, jj]:
                    lab[ii, jj] = n
                    stack.append((ii, jj))
    return lab, n


def _grow(lip, wat, region, steps=None):
    """Lock-step 6-neighbour growth of two labels inside `region`; ties go to `lip`. steps=None: until nothing changes."""
    it = 0
    while steps is None or it < steps:
        free = region & ~lip & ~wat
        newl = _shift_or(lip) & free
        neww = _shift_or(wat) & free & ~newl
        if not (newl.any() or neww.any()):
            break
        lip = lip | newl
        wat = wat | neww
        it += 1
    return lip, wat


def _label_components_3d(mask):
    from scipy import ndimage
    return ndimage.label(mask)            # default structure = 6-neighbour


def _open_closed_2d(plane):
    """-> (open, closed): free-space components of a 2-D slice that touch its border (lipid side) / are enclosed by protein (pore)."""
    from scipy import ndimage
    lab, n = ndimage.label(plane)
    if not n: return np.zeros_like(plane), np.zeros_like(plane)
    edge = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])).tolist()) - {0}
    op = np.isin(lab, list(edge)) & (lab > 0)
    return op, (lab > 0) & ~op


def lipid_region(epsP, epsM, bg, tol, probe, delta, axis=2, centre_idx=None, dist=None, seed="all", slice_probe=3.0):
    """Boolean mask of the voxels that really are membrane lipid (the rest of the membrane-map 'non-water' region is
    pore / cavity water).

    1. In the mid-plane of the slab take the free space (not protein grown by the probe). Components that touch the edge of
       the plane are lipid, the others are channel lumens.  seed="all" (default since delphipka-custom 2026-10-22): the closed
       and open components of EVERY other slice of the slab are also seeds (lumen / lipid), judged with a lipid-tail sized probe
       (slice_probe, 3 A: gaps between splayed helices narrower than ~6 A do not let lipid in).  So a pore segment cut off from the
       mid-plane by a closed gate (BM2 below the Trp23 gate) is still pore; the mid-plane slice itself is classified as before.
       seed="mid": the rule before 2026-10-22 (mid-plane only).
    2. Both labels grow through the free space (nearest wins), so the top of a pore that is only partly walled off still
       belongs to the pore. Free space not reached by any label (enclosed cavities away from the mid-plane) is water.
    3. The thin zone the probe skipped next to the protein is labelled by growing both labels a few steps; space that stays
       unlabelled (a pore narrower than the probe) is water. Finally the protein skin takes the label of the nearest region
       (ties: lipid), so lipid-facing surface grooves stay lipid."""
    if dist is not None:
        # curved membrane: seed in a thin shell around the mid-surface (|d| <= 1 voxel) instead of a mid-plane slice.
        # Shell components that reach a face of the box are lipid, the closed ones are channel lumens.
        nonwater = epsM < (bg - tol)
        solid = epsP < (bg - tol)
        nstep = max(1, int(np.ceil(probe / float(min(delta)))))
        blocked = solid
        for _ in range(nstep):
            blocked = _shift_or(blocked)
        cand = nonwater & ~blocked
        # one-voxel "digital surface": half-thickness 0.5 * sum|n_i| * delta keeps it 6-connected for any tilt (n = unit normal),
        # and for a membrane normal to a grid axis it is exactly the mid-plane slice used for flat membranes
        gx, gy, gz = np.gradient(dist.astype(np.float32), float(delta[0]), float(delta[1]), float(delta[2]))
        half = 0.5 * float(delta[0]) * (np.abs(gx) + np.abs(gy) + np.abs(gz))
        del gx, gy, gz
        shell = cand & (np.abs(dist) < half * (1.0 + 1e-4))
        lab, nl = _label_components_3d(shell)
        faces = np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()])
        edge = set(np.unique(faces).tolist()) - {0}
        lip0 = np.isin(lab, list(edge)) if edge else np.zeros_like(shell)
        wat0 = shell & ~lip0
        if seed == "all":
            from scipy import ndimage
            fat = cand & (ndimage.distance_transform_edt(~solid, sampling=tuple(float(d) for d in delta)) > slice_probe)
            dn = dist[nonwater]
            for d in np.arange(np.floor(dn.min() / delta[0]) * delta[0], dn.max() + delta[0], float(delta[0])):
                if abs(d) < 1e-6: continue
                sh = fat & (np.abs(dist - d) < half * (1.0 + 1e-4)) & ~shell
                if not sh.any(): continue
                lb, _ = _label_components_3d(sh)
                fc = np.concatenate([lb[0].ravel(), lb[-1].ravel(), lb[:, 0].ravel(), lb[:, -1].ravel(), lb[:, :, 0].ravel(), lb[:, :, -1].ravel()])
                ed = set(np.unique(fc).tolist()) - {0}
                op = np.isin(lb, list(ed)) & (lb > 0)
                lip0 |= op; wat0 |= (lb > 0) & ~op
        lip, wat = _grow(lip0, wat0, cand)
        space = nonwater & ~solid
        lip, wat = _grow(lip, wat, space, steps=nstep + 2)
        wat |= space & ~lip & ~wat
        lip, wat = _grow(lip, wat, nonwater)
        return lip, int(np.count_nonzero(nonwater & ~lip))
    # work with the membrane normal as axis 0
    eP = np.moveaxis(epsP, axis, 0); eM = np.moveaxis(epsM, axis, 0)
    nonwater = eM < (bg - tol)
    solid = eP < (bg - tol)
    nstep = max(1, int(np.ceil(probe / float(min(delta)))))
    blocked = solid
    for _ in range(nstep):
        blocked = _shift_or(blocked)
    cand = nonwater & ~blocked
    if centre_idx is None:
        col = nonwater[:, 0, 0] | nonwater[:, -1, -1] | nonwater[:, 0, -1] | nonwater[:, -1, 0]
        idx = np.nonzero(col)[0]
        centre_idx = int(round(idx.mean())) if idx.size else nonwater.shape[0] // 2
    plane = cand[centre_idx]
    lab2, n2 = _label_components_2d(plane)
    edge = set(np.unique(np.concatenate([lab2[0, :], lab2[-1, :], lab2[:, 0], lab2[:, -1]])).tolist()) - {0}
    lip0 = np.zeros_like(cand); wat0 = np.zeros_like(cand)
    for k in range(1, n2 + 1):
        (lip0 if k in edge else wat0)[centre_idx][lab2 == k] = True
    if seed == "all":
        from scipy import ndimage
        dl = [float(delta[axis])] + [float(delta[i]) for i in range(3) if i != axis]
        fat = cand & (ndimage.distance_transform_edt(~solid, sampling=dl) > slice_probe)
        for zi in np.nonzero(nonwater.reshape(nonwater.shape[0], -1).any(1))[0]:
            if zi == centre_idx: continue
            op, cl = _open_closed_2d(fat[zi]); lip0[zi] |= op; wat0[zi] |= cl
    lip, wat = _grow(lip0, wat0, cand)                         # step 2
    space = nonwater & ~solid
    lip, wat = _grow(lip, wat, space, steps=nstep + 2)         # step 3, the skipped zone
    wat |= space & ~lip & ~wat                                 # narrow pores / enclosed cavities
    lip, wat = _grow(lip, wat, nonwater)                       # protein skin: nearest label
    lipid = np.moveaxis(lip, 0, axis)
    return lipid, int(np.count_nonzero(np.moveaxis(nonwater, 0, axis) & ~lipid))


def hydrate_exposed(epsM, atoms_file, axes, delta, bg, tol, lum, probe, radius, groups, axis, seed=2.0):
    """Residue-exposure dependent slab edge.  The Gaussian protein map and the 2-D mid-plane seeding of lipid_region can call a narrow
    pore lipid, so an ionisable group that really faces a water-filled pore or funnel sees a dry membrane ('burial' by the model, not by the
    protein).  Here the water-accessible space is found from the ATOMS: probe centres (distance >= probe from every atom surface) inside
    the slab are labelled; components reaching the lateral box faces are lipid, the others (pore, funnel, cavity) are water.  Around the
    selected groups (within `radius` A) the voxels that lie within one probe of such water-space and are still lipid in `epsM` become water."""
    from scipy import ndimage as ndi
    pts = []; meta = []
    for line in open(atoms_file):
        if line.startswith(("ATOM", "HETATM")):
            t = line[54:].split()
            try:
                r = float(t[1]); pts.append((float(line[30:38]), float(line[38:46]), float(line[46:54]), r))
                meta.append((line[17:20].strip(), line[21], line[22:26].strip(), line[12:16].strip()))
            except (ValueError, IndexError):
                pass
    n = epsM.shape; occ = np.zeros(n, dtype=bool)
    for x, y, z, r in pts:
        if r < 0.5:
            continue
        ii = [np.nonzero(np.abs(axes[i] - c) <= r)[0] for i, c in enumerate((x, y, z))]
        if any(q.size == 0 for q in ii):
            continue
        X, Y, Z = np.meshgrid(axes[0][ii[0]], axes[1][ii[1]], axes[2][ii[2]], indexing="ij")
        occ[np.ix_(ii[0], ii[1], ii[2])] |= (X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2 <= r * r
    sam = tuple(delta)
    D = ndi.distance_transform_edt(~occ, sampling=sam)
    slab = epsM < (bg - tol)
    lat = [i for i in range(3) if i != axis]
    def lateral(freem):
        lab, nl = ndi.label(freem & slab)
        touch = set()
        for l_ in lat:
            for sl in (0, -1):
                idx = [slice(None)] * 3; idx[l_] = sl
                touch |= set(np.unique(lab[tuple(idx)]).tolist())
        touch.discard(0)
        isl = np.isin(lab, list(touch))
        return isl, (lab > 0) & ~isl
    # seeds with a lipid-chain sized probe (pores > 2 A radius separate cleanly from the lipid), then both labels grow through the
    # finer water-probe space (lock-step, ties lipid) so a narrow stretch of a pore (His19 constriction) still belongs to the pore
    lip, wat = lateral(D >= seed)
    free = D >= probe
    lip, wat = _grow(lip, wat, free & slab)
    water = wat
    reach = ndi.distance_transform_edt(~water, sampling=sam) <= probe + max(delta)       # skin between the atoms and the probe centres
    tgt = {"his": {("HIS", "ND1"), ("HIS", "NE2")}, "acid": {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")},
           "lys": {("LYS", "NZ")}, "arg": {("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2")}}
    want = set()
    for g in str(groups).split(","):
        want |= tgt.get(g.strip().lower(), set())
    near = np.zeros(n, dtype=bool); who = set()
    for (x, y, z, r), m in zip(pts, meta):
        if (m[0], m[3]) not in want:
            continue
        ii = [np.nonzero(np.abs(axes[i] - c) <= radius)[0] for i, c in enumerate((x, y, z))]
        if any(q.size == 0 for q in ii):
            continue
        X, Y, Z = np.meshgrid(axes[0][ii[0]], axes[1][ii[1]], axes[2][ii[2]], indexing="ij")
        sph = (X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2 <= radius * radius
        sub = reach[np.ix_(ii[0], ii[1], ii[2])] & slab[np.ix_(ii[0], ii[1], ii[2])] & ~occ[np.ix_(ii[0], ii[1], ii[2])] & sph
        if sub.any():
            who.add("%s%s%s" % (m[0], m[2], m[1]))
        near[np.ix_(ii[0], ii[1], ii[2])] |= sph
    new = near & reach & slab & ~occ
    out = np.where(new, lum, epsM)
    print("  hydrate-exposed: pore/funnel/cavity water space %d voxels (%.0f A^3); %d lipid-labelled voxels within %g A of %s set to eps %g; groups reached: %s"
          % (int(water.sum()), water.sum() * float(np.prod(delta)), int(new.sum()), radius, groups, lum, " ".join(sorted(who)) or "none"))
    return out

def cmd_combine(a):
    P = read_map(a.protein)
    M = read_map(a.membrane)

    # output grid = protein grid, optionally padded
    delta = P.delta.copy()
    npad = [int(round(a.pad / delta[i])) for i in range(3)]
    origin = P.origin - np.array(npad) * delta
    n = [P.n[i] + 2 * npad[i] for i in range(3)]
    axes = [origin[i] + delta[i] * np.arange(n[i]) for i in range(3)]

    epsP = resample(P, axes, outside="value", fill=a.bg)
    epsM = resample(M, axes, outside=("edge" if a.membrane_outside == "edge" else "value"), fill=a.bg)

    interior = a.interior if a.interior is not None else float(P.data.min())
    if a.background_only and not a.lumen_water:
        sys.exit("combine: --background-only needs the lumen-water detection (do not use --no-lumen-water)")

    nwat = 0
    if a.lumen_water:
        dist = None
        if a.shape_file:
            import membrane_shape as msh
            dist = msh.distance_on_grid(msh.load_shape(a.shape_file), origin, delta, n)
        lipid, nwat = lipid_region(epsP, epsM, a.bg, a.tol, a.probe, delta, axis="xyz".index(a.axis), dist=dist, seed=a.lumen_seed, slice_probe=a.lumen_slice_probe)
        lum = a.lumen_eps if a.lumen_eps is not None else a.bg
        epsM = np.where(lipid | (epsM >= a.bg - a.tol), epsM, lum)
        if a.hydrate_radius and a.atoms:
            epsM = hydrate_exposed(epsM, a.atoms, axes, delta, a.bg, a.tol, lum, a.hydrate_probe, a.hydrate_radius, a.hydrate_groups, "xyz".index(a.axis), a.hydrate_seed)
    if a.background_only:
        out = epsM
        if a.protein_eps is not None:
            # protein body inside the membrane: lipid-labelled voxels take the protein dielectric weighted by the protein density
            # (voxels labelled water, i.e. facing a pore/bulk, are left alone: DelPhi's own Gaussian builds that surface)
            if a.bg - interior <= 0:
                sys.exit("combine: --bg must be larger than the protein interior epsilon (%g)" % interior)
            fprot = np.clip((a.bg - epsP) / (a.bg - interior), 0.0, 1.0)
            lipmask = epsM < (a.bg - a.tol)
            out = np.where(lipmask, fprot * a.protein_eps + (1.0 - fprot) * epsM, epsM)
            print("  protein body inside the membrane set to eps %g in %d voxels" % (a.protein_eps, int(np.count_nonzero(lipmask & (fprot > 0.5)))))
            if a.atoms:
                # solid interior: atoms (their own radii) closed with a water probe, shrunk by --interior-depth, gets the protein dielectric
                # whatever the lipid/water label was.  Without this the voids between atoms of a pore-side body keep the water value.
                from scipy import ndimage as ndi
                pts = []; meta = []
                for line in open(a.atoms):
                    if line.startswith(("ATOM", "HETATM")):
                        t = line[54:].split()
                        try:
                            r = float(t[1])
                            pts.append((float(line[30:38]), float(line[38:46]), float(line[46:54]), r))
                            meta.append((line[17:20], line[21], line[22:26].strip(), line[12:16].strip()))
                        except (ValueError, IndexError):
                            pass
                occ = np.zeros(tuple(n), dtype=bool)
                for x, y, z, r in pts:
                    if r < 0.5:
                        continue                           # united-atom hydrogens have radius 0
                    ii = [np.nonzero(np.abs(axes[i] - c) <= r)[0] for i, c in enumerate((x, y, z))]
                    if any(q.size == 0 for q in ii):
                        continue
                    X, Y, Z = np.meshgrid(axes[0][ii[0]], axes[1][ii[1]], axes[2][ii[2]], indexing="ij")
                    sph = (X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2 <= r * r
                    occ[np.ix_(ii[0], ii[1], ii[2])] |= sph
                sampling = tuple(delta)
                grown = ndi.distance_transform_edt(~occ, sampling=sampling) <= a.probe_close
                body = ndi.distance_transform_edt(grown, sampling=sampling) > a.probe_close      # closing: fill what a probe cannot enter
                body |= occ
                depth = ndi.distance_transform_edt(body, sampling=sampling)
                interior_mask = depth >= a.interior_depth
                Wzone = None
                if a.solid_graded is not None or a.solid_buried_sasa is not None:
                    from scipy.spatial import cKDTree
                    keep = [i for i, q in enumerate(pts) if q[3] >= 0.5]
                    xyz = np.array([pts[i][:3] for i in keep]); rad = np.array([pts[i][3] for i in keep]) + a.probe_sasa
                    tree = cKDTree(xyz); k = np.arange(240); zz = 1 - 2 * (k + 0.5) / 240; rr = np.sqrt(1 - zz * zz); ph = np.pi * (3 - 5 ** 0.5) * k
                    sph = np.stack([rr * np.cos(ph), rr * np.sin(ph), zz], 1)
                    acid = {}
                    _sg = set(x.strip() for x in str(a.solid_groups).split(","))
                    if "acid" in _sg:
                        acid.update({"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")})
                    if "base" in _sg:
                        acid.update({"LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2"), "HIS": ("ND1", "NE2")})
                    grp = {}
                    for j, i in enumerate(keep):
                        m = meta[i]
                        if m[0] in acid and m[3] in acid[m[0]]:
                            nb = [b for b in tree.query_ball_point(xyz[j], rad[j] + rad.max()) if b != j]
                            pnt = xyz[j] + sph * rad[j]
                            free = (np.linalg.norm(pnt[:, None, :] - xyz[nb][None], axis=2) >= rad[nb][None, :]).all(1) if nb else np.ones(240, bool)
                            g = grp.setdefault((m[0], m[1], m[2]), [0.0, []])
                            g[0] += free.mean() * 4 * np.pi * rad[j] ** 2; g[1].append(xyz[j])
                    if a.solid_graded is not None:
                        S0, S1 = a.solid_graded
                        charged_partner = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("HIS", "ND1"), ("HIS", "NE2")}
                        pidx = [i for i in keep if (meta[i][0], meta[i][3]) in charged_partner]
                        ptree = cKDTree(np.array([pts[i][:3] for i in pidx])) if pidx else None
                        wts = {}
                        for k_, v in grp.items():
                            w_ = float(np.clip((S1 - v[0]) / max(S1 - S0, 1e-6), 0.0, 1.0))
                            if w_ > 0 and ptree is not None and a.solid_partner_factor != 1.0:
                                near_p = [j for x_ in v[1] for j in ptree.query_ball_point(x_, a.partner_dist)
                                          if not (meta[pidx[j]][1] == k_[1] and meta[pidx[j]][2] == k_[2])]
                                if near_p:
                                    w_ *= a.solid_partner_factor
                            if w_ > 0:
                                wts[k_] = w_
                        buried = {k_: grp[k_] for k_ in wts}
                        print("  graded solid zone (SASA %g..%g A^2, partner factor %g): %d of %d Asp/Glu with weight > 0, mean weight %.2f"
                              % (S0, S1, a.solid_partner_factor, len(wts), len(grp), np.mean(list(wts.values())) if wts else 0.0))
                    else:
                        buried = {k_: v for k_, v in grp.items() if v[0] < a.solid_buried_sasa}
                    if a.solid_skip_partnered > 0 and buried:
                        # a buried acid with a cationic partner (Lys NZ incl. a Schiff base, Arg N, His ring N) within R A is stabilised by that
                        # partner and by the water that usually comes with it: do not dry its surroundings
                        _all = {"LYS": [("LYS", "NZ"), ("LYR", "NZ")], "ARG": [("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2")], "HIS": [("HIS", "ND1"), ("HIS", "NE2")]}
                        _cat = {x for g in str(a.solid_skip_cations).upper().split(",") for x in _all.get(g.strip(), [])}
                        _neutral = {t.strip().upper() for t in str(a.solid_neutral_partners or "").split(",") if t.strip()}
                        _ci = [i for i in keep if (meta[i][0], meta[i][3]) in _cat
                               and ("%s%s%s" % (meta[i][0], meta[i][2], meta[i][1])).upper() not in _neutral]   # partners found neutral in a first pass do not count
                        if _ci:
                            _ct = cKDTree(np.array([pts[i][:3] for i in _ci])); _drop = []; _who = {}
                            for k_, v in buried.items():
                                _p = sorted({"%s%s%s" % (meta[_ci[j]][0], meta[_ci[j]][2], meta[_ci[j]][1]) for x_ in v[1] for j in _ct.query_ball_point(x_, a.solid_skip_partnered)
                                             if not (meta[_ci[j]][1] == k_[1] and meta[_ci[j]][2] == k_[2])})
                                if _p:
                                    _drop.append(k_); _who["%s%s%s" % (k_[0], k_[2], k_[1])] = _p
                            if a.partners_out:          # which partner exempted which acid (read by pka-helper's partner check)
                                with open(a.partners_out, "w") as _fo:
                                    for _acid, _p in sorted(_who.items()): _fo.write("%s %s\n" % (_acid, " ".join(_p)))
                            for k_ in _drop:
                                buried.pop(k_)
                                if a.solid_graded is not None:
                                    wts.pop(k_, None)
                            print("  solid zone skipped for %d buried acids with a cationic partner within %g A: %s"
                                  % (len(_drop), a.solid_skip_partnered, " ".join("%s%s%s" % (k_[0], k_[2], k_[1]) for k_ in sorted(_drop))))
                    if a.solid_only_res:                                  # diagnostic: restrict the zone to some residues, e.g. ASP126,ASP127
                        _only = set(x.strip().upper() for x in a.solid_only_res.split(","))
                        buried = {k_: v for k_, v in buried.items() if ("%s%s" % (k_[0], k_[2])) in _only}
                        if a.solid_graded is not None:
                            wts = {k_: w_ for k_, w_ in wts.items() if k_ in buried}
                    print("  buried groups (SASA < %s A^2): %d of %d  %s" % (a.solid_buried_sasa if a.solid_graded is None else "graded", len(buried), len(grp),
                          " ".join("%s%s%s" % (k_[0], k_[2], k_[1]) for k_ in sorted(buried))))
                    near = np.zeros(tuple(n), dtype=bool)
                    if a.solid_graded is not None:
                        Wzone = np.zeros(tuple(n), dtype=np.float32)
                    if buried:
                        R = a.solid_radius
                        for k_, v in buried.items():
                            wg = wts[k_] if a.solid_graded is not None else 1.0
                            for cx, cy, cz in v[1]:
                                ii = [np.nonzero(np.abs(axes[q] - c) <= R)[0] for q, c in enumerate((cx, cy, cz))]
                                if any(w.size == 0 for w in ii):
                                    continue
                                X, Y, Z = np.meshgrid(axes[0][ii[0]], axes[1][ii[1]], axes[2][ii[2]], indexing="ij")
                                sph_ = ((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= R * R)
                                near[np.ix_(ii[0], ii[1], ii[2])] |= sph_
                                if Wzone is not None:
                                    sub_w = Wzone[np.ix_(ii[0], ii[1], ii[2])]
                                    Wzone[np.ix_(ii[0], ii[1], ii[2])] = np.where(sph_, np.maximum(sub_w, wg), sub_w)
                    interior_mask = interior_mask & near
                    if a.solid_spare_partners > 0 and buried:
                        # keep the solvation shell of charged partners (Lys NZ, Arg N, His ring N) that are NOT in direct contact with a flagged group:
                        # the zone models the packing round the buried group, it should not dry out an exposed counter-ion
                        _sp = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("HIS", "ND1"), ("HIS", "NE2")}
                        if a.solid_spare_acids:
                            _sp |= {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
                        ptn = [i for i in keep if (meta[i][0], meta[i][3]) in _sp
                               and (meta[i][0], meta[i][1], meta[i][2]) not in buried]
                        fl = np.array([c for v in buried.values() for c in v[1]])
                        spare = np.zeros(tuple(n), dtype=bool); nsp = 0
                        for i in ptn:
                            cx, cy, cz = pts[i][:3]
                            if np.min(np.linalg.norm(fl - np.array([cx, cy, cz]), axis=1)) < a.spare_contact:
                                continue
                            nsp += 1
                            R2 = a.solid_spare_partners
                            ii = [np.nonzero(np.abs(axes[q] - c) <= R2)[0] for q, c in enumerate((cx, cy, cz))]
                            if any(w.size == 0 for w in ii):
                                continue
                            X, Y, Z = np.meshgrid(axes[0][ii[0]], axes[1][ii[1]], axes[2][ii[2]], indexing="ij")
                            spare[np.ix_(ii[0], ii[1], ii[2])] |= ((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= R2 * R2)
                        before = int(np.count_nonzero(interior_mask))
                        interior_mask = interior_mask & ~spare
                        print("  spared the %g A shell of %d charged partner atoms (> %g A from any flagged group): %d voxels returned to the Gaussian"
                              % (a.solid_spare_partners, nsp, a.spare_contact, before - int(np.count_nonzero(interior_mask))))
                if Wzone is not None:
                    out = np.where(interior_mask, Wzone * a.protein_eps + (1.0 - Wzone) * out, out)
                else:
                    out = np.where(interior_mask, a.protein_eps, out)
                print("  solid interior (atoms closed with a %g A probe, depth >= %g A): %d voxels (%.0f A^3) set to eps %g"
                      % (a.probe_close, a.interior_depth, int(np.count_nonzero(interior_mask)), np.count_nonzero(interior_mask) * float(np.prod(delta)), a.protein_eps))
        info = "BACKGROUND map: membrane + water (+ pore/cavity water), no protein. Use  Custom dielectric map type : BACKGROUND"
    elif a.mode == "blend":
        if a.bg - interior <= 0:
            sys.exit("combine: --bg must be larger than the protein interior epsilon (%g)" % interior)
        f = np.clip((a.bg - epsP) / (a.bg - interior), 0.0, 1.0)
        out = f * interior + (1.0 - f) * epsM
        info = "blend: eps = f*%g + (1-f)*eps_membrane, f = (%g - eps_protein)/(%g - %g) clipped to 0..1" % (interior, a.bg, a.bg, interior)
    elif a.mode == "mask":
        sel = epsP < (a.bg - a.tol)
        out = np.where(sel, epsP, epsM)
        info = "mask: protein map where eps < %g, membrane map elsewhere" % (a.bg - a.tol)
    elif a.mode == "min":
        out = np.minimum(epsP, epsM); info = "min of both maps"
    else:
        out = np.maximum(epsP, epsM); info = "max of both maps"

    if a.waters:
        wx = []
        for line in open(a.waters):
            if line.startswith(("ATOM", "HETATM")):
                if line[17:20].strip() in ("HOH", "WAT", "H2O", "TIP"):
                    wx.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            elif line.strip() and not line.startswith("#"):
                t = line.split()
                if len(t) >= 3:
                    wx.append([float(t[0]), float(t[1]), float(t[2])])
        out = np.array(out, dtype=float)
        gx, gy, gz = axes
        nset = 0
        for w in wx:
            ix = np.nonzero(np.abs(gx - w[0]) <= a.water_radius)[0]
            iy = np.nonzero(np.abs(gy - w[1]) <= a.water_radius)[0]
            iz = np.nonzero(np.abs(gz - w[2]) <= a.water_radius)[0]
            if ix.size == 0 or iy.size == 0 or iz.size == 0:
                continue
            X, Y, Z = np.meshgrid(gx[ix], gy[iy], gz[iz], indexing="ij")
            sel = (X - w[0]) ** 2 + (Y - w[1]) ** 2 + (Z - w[2]) ** 2 <= a.water_radius ** 2
            sub_ = out[np.ix_(ix, iy, iz)]
            nset += int(np.count_nonzero(sel & (sub_ < a.bg - 1e-3)))
            out[np.ix_(ix, iy, iz)] = np.where(sel, a.bg, sub_)
        print("  %d crystal-water sites (radius %g A) set to eps %g in the background (%d voxels changed)" % (len(wx), a.water_radius, a.bg, nset))

    m = EpsMap(origin, delta, out.astype(np.float32))
    write_map(a.out, m, "combined: " + info)
    print("wrote %s  grid %s  spacing %s A" % (a.out, tuple(n), tuple(float(round(x, 4)) for x in delta)))
    print("  " + info)
    print("  eps range %.3f .. %.3f" % (out.min(), out.max()))
    if a.lumen_water:
        print("  pore/cavity water: %d membrane-slab voxels (%.0f A^3) enclosed by the protein were set to water (eps %g)"
              % (nwat, nwat * float(np.prod(delta)), a.lumen_eps if a.lumen_eps is not None else a.bg))
    if a.mode == "blend":
        print("  protein interior epsilon used: %g  (set --interior to the 'Internal Dielectric' you used for the protein map)" % interior)
    print("  in run.prm use:  Custom dielectric map : %s   /   Custom dielectric map outside : EXTEND" % os.path.basename(a.out))


# ------------------------------------------------------------------------------------ info / convert
def cmd_info(a):
    m = read_map(a.map)
    up = m.upper()
    d = m.data
    print("%s" % a.map)
    print("  points   : %d x %d x %d  (%.2f M)" % (*m.n, d.size / 1e6))
    print("  spacing  : %g x %g x %g A" % tuple(m.delta))
    print("  box (A)  : x %.3f .. %.3f   y %.3f .. %.3f   z %.3f .. %.3f" % (m.origin[0], up[0], m.origin[1], up[1], m.origin[2], up[2]))
    print("  centre   : %.3f %.3f %.3f" % tuple((m.origin + up) / 2))
    print("  eps      : min %.4g  max %.4g  mean %.4g" % (d.min(), d.max(), d.mean()))
    bins = [(0, 3), (3, 10), (10, 30), (30, 70), (70, 1e9)]
    tot = d.size
    print("  share    : " + "   ".join("%g-%g: %.1f%%" % (lo, min(hi, 999), 100.0 * np.count_nonzero((d >= lo) & (d < hi)) / tot) for lo, hi in bins))
    # profile along each axis through the box centre line of the most 'low-eps' column
    for a_i, name in enumerate("xyz"):
        idx = [slice(None)] * 3
        other = [i for i in range(3) if i != a_i]
        # use the column with the lowest mean eps to show the profile (typically through the protein/membrane)
        mean_other = d.mean(axis=a_i)
        j = np.unravel_index(np.argmin(mean_other), mean_other.shape)
        idx[other[0]] = j[0]; idx[other[1]] = j[1]
        col = d[tuple(idx)]
        pts = np.linspace(0, col.size - 1, 9).astype(int)
        print("  along %s : " % name + "  ".join("%.0f:%.3g" % (m.origin[a_i] + m.delta[a_i] * p, col[p]) for p in pts))


def cmd_convert(a):
    m = read_map(a.inp)
    write_map(a.out, m, "converted from %s" % os.path.basename(a.inp))
    print("wrote %s" % a.out)


# ---------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("info", help="print grid, range and a few profiles"); s.add_argument("map"); s.set_defaults(f=cmd_info)
    s = sp.add_parser("convert", help="cube <-> dx"); s.add_argument("inp"); s.add_argument("out"); s.set_defaults(f=cmd_convert)

    s = sp.add_parser("membrane", help="write a membrane slab map")
    s.add_argument("out")
    s.add_argument("--like", help="copy the grid of this map (recommended)")
    s.add_argument("--box", type=float, nargs=6, metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"))
    s.add_argument("--spacing", type=float, help="grid spacing for --box (A)")
    s.add_argument("--axis", choices="xyz", default="z", help="membrane normal (default z)")
    s.add_argument("--center", type=float, default=0.0, help="coordinate of the membrane mid-plane along the normal (default 0)")
    s.add_argument("--core-thickness", type=float, default=None, help="hydrocarbon core thickness, A (default 30; with --shape-file: the value in the JSON)")
    s.add_argument("--head-thickness", type=float, default=8.0, help="head-group zone on each side, A (default 8; 0 = none)")
    s.add_argument("--eps-core", type=float, default=2.0, help="core dielectric (default 2)")
    s.add_argument("--eps-head", type=float, default=30.0, help="head-group zone dielectric (default 30)")
    s.add_argument("--eps-water", type=float, default=80.0, help="bulk water dielectric (default 80)")
    s.add_argument("--smooth", type=float, default=1.0, help="Gaussian smoothing width of the layer boundaries, A (default 1)")
    s.add_argument("--shape-file", help="curved membrane: JSON from membrane_shape.py. The layers follow the signed distance to the fitted mid-surface "
                                        "(--axis/--center are ignored); also writes the distance map (--distance-out)")
    s.add_argument("--distance-out", help="with --shape-file: signed-distance map to write (default <out>_dist.dx)")
    s.set_defaults(f=cmd_membrane)

    s = sp.add_parser("combine", help="combine a protein map with a membrane/environment map")
    s.add_argument("protein"); s.add_argument("membrane"); s.add_argument("out")
    s.add_argument("--mode", choices=["blend", "mask", "min", "max"], default="blend")
    s.add_argument("--bg", type=float, default=80.0, help="water dielectric of the protein map's background (default 80)")
    s.add_argument("--interior", type=float, default=None, help="protein interior dielectric for blend (default: minimum of the protein map)")
    s.add_argument("--protein-eps", type=float, default=None,
                   help="with --background-only: dielectric of the protein body where it is embedded in lipid (default: none = lipid eps). Use the same value as 'Internal Dielectric'.")
    s.add_argument("--atoms", default=None, help="PQR of the protein (x y z charge radius). With --protein-eps: the probe-closed solid interior takes the protein dielectric whatever its lipid/water label")
    s.add_argument("--solid-buried-sasa", type=float, default=None, help="with --atoms: restrict the solid interior to the neighbourhood (--solid-radius) of Asp/Glu side chains whose solvent-accessible area (1.4 A probe) is below this many A^2 (e.g. 1.0). Everything else keeps DelPhi's Gaussian")
    s.add_argument("--solid-graded", type=float, nargs=2, metavar=("S0", "S1"), default=None,
                   help="with --atoms: GRADED solid zone instead of the on/off --solid-buried-sasa: each Asp/Glu gets weight w = clip((S1 - SASA)/(S1 - S0), 0, 1) "
                        "(e.g. 0 15) and the zone's eps is w*protein_eps + (1-w)*map value, so partly buried groups get a partly solid environment")
    s.add_argument("--solid-partner-factor", type=float, default=1.0,
                   help="with --solid-graded: multiply w by this for a carboxylate with a charged partner (Lys NZ, Arg N, His ring N) within --partner-dist, whose own pair interaction already stabilises it (default 1 = ignore partners; try 0.5)")
    s.add_argument("--partner-dist", type=float, default=4.0, help="partner distance for --solid-partner-factor, A (default 4)")
    s.add_argument("--probe-sasa", type=float, default=1.4, help=argparse.SUPPRESS)
    s.add_argument("--solid-groups", default="acid", help="with --solid-buried-sasa / --solid-graded: which side chains are tested for burial: 'acid' (Asp/Glu, default), 'base' (Lys NZ, Arg NE/NH1/NH2, His ND1/NE2) or 'acid,base'")
    s.add_argument("--solid-only-res", default="", help="diagnostic: apply the solid zone only to these flagged residues (e.g. ASP126,ASP127; any chain)")
    s.add_argument("--solid-neutral-partners", default="", help="with --solid-skip-partnered: residues (e.g. HIS322A,LYS7B) that do NOT count as partners because a "
                   "first calculation found them neutral")
    s.add_argument("--partners-out", default="", help="with --solid-skip-partnered: write 'ACID partner partner ...' lines for the exempted acids to this file")
    s.add_argument("--solid-skip-cations", default="LYS,ARG,HIS", help="with --solid-skip-partnered: residues whose N atoms count as a salt-bridge partner (default LYS,ARG,HIS)")
    s.add_argument("--solid-skip-partnered", type=float, default=0.0, help="do not apply the solid zone to buried acids that have a cationic partner (Lys NZ / Schiff base, Arg N, His ring N) within this distance (A) (default 0 = off)")
    s.add_argument("--solid-spare-partners", type=float, default=0.0, help="radius (A) round the N atoms of Lys/Arg/His that are not in contact with a flagged group inside which the solid zone is NOT applied (default 0 = off)")
    s.add_argument("--solid-spare-acids", action="store_true", help="with --solid-spare-partners: also spare the shell of Asp/Glu carboxylates that are not flagged as buried (a titrating acid next to a flagged one should not be dried out)")
    s.add_argument("--spare-contact", type=float, default=5.0, help="a partner atom closer than this (A) to a flagged side chain counts as in contact and keeps the zone (default 5)")
    s.add_argument("--solid-radius", type=float, default=6.0, help="radius around the buried groups inside which the solid interior applies, A (default 6)")
    s.add_argument("--hydrate-radius", type=float, default=0.0, help="EXPERIMENTAL residue-exposure dependent slab edge (needs --atoms): within this many A of the selected groups, lipid-labelled voxels next to water-accessible pore / funnel space become water (default 0 = off)")
    s.add_argument("--hydrate-groups", default="his", help="with --hydrate-radius: comma list of his, acid, lys, arg (default his)")
    s.add_argument("--hydrate-seed", type=float, default=2.0, help="with --hydrate-radius: probe used to separate pore from lipid space before growing, A (default 2.0)")
    s.add_argument("--hydrate-probe", type=float, default=1.4, help="probe radius for the atom-based water-accessible space, A (default 1.4)")
    s.add_argument("--probe-close", type=float, default=1.4, help="probe radius for closing the atom spheres, A (default 1.4)")
    s.add_argument("--interior-depth", type=float, default=1.0, help="only voxels at least this deep inside the closed protein are overwritten (the surface stays with DelPhi's Gaussian), A (default 1.0)")
    s.add_argument("--waters", default=None, help="PDB (HOH records) or 'x y z' file of crystal waters to keep: a sphere of --water-radius around each is set to the bulk "
                   "dielectric in the output map (internal / pocket waters; a continuum representation, no explicit water charges)")
    s.add_argument("--water-radius", type=float, default=1.4, help="radius of the water sphere, A (default 1.4)")
    s.add_argument("--tol", type=float, default=0.5, help="mask mode: protein where eps < bg - tol (default 0.5)")
    s.add_argument("--pad", type=float, default=0.0, help="extend the output box by this many A on every side (default 0)")
    s.add_argument("--lumen-eps", type=float, default=None,
                   help="dielectric of the water in channel lumens / enclosed cavities inside the slab (default: --bg, i.e. bulk water). Confined water is less polarisable; try 30-60")
    s.add_argument("--lumen-water", action=argparse.BooleanOptionalAction, default=True,
                   help="keep water in channel lumens / enclosed cavities inside the membrane slab (default on; --no-lumen-water to turn off)")
    s.add_argument("--background-only", action="store_true",
                   help="write only the membrane+water background (protein used just to find pore water). Use it with "
                        "'Custom dielectric map type : BACKGROUND': DelPhi then builds the protein Gaussian itself at full resolution "
                        "(recommended; a protein baked into a 0.5 A map gives wrong desolvation energies)")
    s.add_argument("--axis", choices="xyz", default="z", help="membrane normal, used to find the mid-plane for lumen detection (default z)")
    s.add_argument("--lumen-slice-probe", type=float, default=3.0, help="with --lumen-seed all: probe (A) that decides, slice by slice, whether free space next to the protein is open to the lipid (default 3.0)")
    s.add_argument("--lumen-seed", choices=["all", "mid"], default="all", help="where pore water is seeded: space enclosed by protein in ANY slice of the slab (all, default) or only in the mid-plane (mid, the rule before 2026-10-22; misses a pore segment below a closed gate)")
    s.add_argument("--shape-file", help="curved membrane: JSON from membrane_shape.py; the lumen-water seed is a thin shell around the fitted mid-surface instead of the mid-plane")
    s.add_argument("--probe", type=float, default=2.0, help="probe radius (A, default 2.0) used when seeding lipid vs pore water in the slab mid-plane; protein gaps narrower than 2x this count as closed")
    s.add_argument("--membrane-outside", choices=["edge", "water"], default="edge",
                   help="what the membrane map is outside its own box: continue its edge values (default) or water (--bg)")
    s.set_defaults(f=cmd_combine)

    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
