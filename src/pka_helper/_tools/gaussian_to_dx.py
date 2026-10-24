#!/usr/bin/env python3
"""
Write a DelPhiPKa-compatible OpenDX dielectric map from a PQR file, using the same Gaussian
density the built-in model uses.  Use it as a template: replace the `density` array by your own
field (e.g. from MD, a membrane model, ...) and keep the writer.

    python3 tools/gaussian_to_dx.py sample_1.pqr my_map.dx --type EPS --spacing 0.125

Notes
 * Coordinates are in Angstrom, in the same frame as the PDB given to DelPhiPKa.
 * Hydrogens (radius ~1 A) make narrow peaks: use a spacing of 0.125-0.25 A, otherwise
   trilinear interpolation flattens them.
 * Outside the map DelPhiPKa assumes bulk water, so pad the box by >= 8 A around the molecule.
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("pqr"); ap.add_argument("out")
ap.add_argument("--type", choices=["EPS", "FRACTION"], default="EPS")
ap.add_argument("--spacing", type=float, default=0.25)
ap.add_argument("--padding", type=float, default=10.0)
ap.add_argument("--sigma", type=float, default=0.70)
ap.add_argument("--indi", type=float, default=8.0)
ap.add_argument("--exdi", type=float, default=80.0)
a = ap.parse_args()

atoms = []
for line in open(a.pqr):
    if line.startswith(("ATOM", "HETATM")):
        # fixed columns: coordinates can touch each other (e.g. -30.284-100.454), so never split on whitespace
        t = line[54:].split()                                                  # charge radius
        atoms.append((float(line[30:38]), float(line[38:46]), float(line[46:54]), float(t[1])))
atoms = np.array(atoms)

h = a.spacing
lo = atoms[:, :3].min(0) - a.padding
hi = atoms[:, :3].max(0) + a.padding
n = np.ceil((hi - lo) / h).astype(int) + 1
# ---- replace this block with your own density field (0 = bulk solvent, 1 = protein interior) ----
# Each atom only touches the voxels within 3*sigma*r of it, so work on that small sub-box (fast for whole proteins).
keep = np.ones(tuple(n), dtype=np.float32)
for x, y, z, r in atoms:
    if r < 1e-6:
        continue
    rc = 3 * a.sigma * r
    c = (np.array([x, y, z]) - lo) / h
    i0 = np.maximum(np.floor(c - rc / h).astype(int), 0)
    i1 = np.minimum(np.ceil(c + rc / h).astype(int) + 1, n)
    gx = lo[0] + h * np.arange(i0[0], i1[0]) - x
    gy = lo[1] + h * np.arange(i0[1], i1[1]) - y
    gz = lo[2] + h * np.arange(i0[2], i1[2]) - z
    d2 = gx[:, None, None] ** 2 + gy[None, :, None] ** 2 + gz[None, None, :] ** 2
    den = np.exp(-d2 / (a.sigma ** 4 * r * r))
    den[d2 > rc * rc] = 0
    keep[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] *= (1 - den).astype(np.float32)
density = 1 - keep
# -----------------------------------------------------------------------------------------------


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

values = density if a.type == "FRACTION" else density * a.indi + (1 - density) * a.exdi

with open(a.out, "w") as o:
    o.write("object 1 class gridpositions counts %d %d %d\n" % tuple(n))
    o.write("origin %g %g %g\n" % tuple(lo))
    o.write("delta %g 0 0\ndelta 0 %g 0\ndelta 0 0 %g\n" % (h, h, h))
    o.write("object 2 class gridconnections counts %d %d %d\n" % tuple(n))
    o.write("object 3 class array type double rank 0 items %d data follows\n" % values.size)
    flat = values.ravel()                       # x slowest, z fastest (DX order)
    pad = (-flat.size) % 3
    body = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)]).reshape(-1, 3)
    _write_rows(o, body[:-1] if pad else body, "%.6g", " ")
    if pad:
        o.write(" ".join("%.6g" % v for v in flat[flat.size - (3 - pad):]) + "\n")
print("wrote", a.out, "grid", tuple(n), "origin", lo)
