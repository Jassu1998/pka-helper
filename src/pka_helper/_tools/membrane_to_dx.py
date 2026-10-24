#!/usr/bin/env python3
"""
Build a DelPhiPKa dielectric map for a protein in a membrane slab.

The protein must already be oriented with the membrane normal along z and the bilayer centre at
z = 0 (e.g. an OPM / PPM / CHARMM-GUI-oriented structure).  The map is

    eps(x,y,z) = f_protein(x,y,z) * indi + (1 - f_protein) * eps_env(z)

f_protein is the same Gaussian atom density the built-in model uses; eps_env(z) is water (exdi)
outside the membrane, a hydrocarbon core, and an optional interfacial (headgroup) layer, joined by
smooth error-function steps.

    python3 tools/membrane_to_dx.py oriented.pqr membrane.dx --core 30 --core-eps 2 \
            --head 8 --head-eps 20 --spacing 0.25

Then in run.prm:
    Gaussian surface                   : 1
    Internal Dielectric                : 8.0          # must equal --indi
    External Dielectric                : 80.0         # must equal --exdi
    Custom dielectric map              : membrane.dx
    Custom dielectric map type         : EPS
    Custom dielectric map outside      : EXTEND       # lets the slab continue laterally

The map only needs ~10 A of lateral padding around the protein because EXTEND continues the edge
values.  Defaults are common starting points (30 A hydrocarbon core, eps 2, 8 A interface, eps 20),
not validated numbers: choose them for your lipid and justify them in your methods.
"""
import argparse
import math
import numpy as np

try:
    from scipy.special import erf
except ImportError:                                   # no SciPy: vectorised math.erf
    erf = np.vectorize(math.erf)

ap = argparse.ArgumentParser()
ap.add_argument("pqr"); ap.add_argument("out")
ap.add_argument("--spacing", type=float, default=0.25, help="grid spacing in A (default 0.25)")
ap.add_argument("--pad-xy", type=float, default=10.0, help="lateral padding around the protein (A)")
ap.add_argument("--pad-z", type=float, default=12.0, help="water padding above/below slab and protein (A)")
ap.add_argument("--core", type=float, default=30.0, help="hydrocarbon core thickness (A)")
ap.add_argument("--core-eps", type=float, default=2.0)
ap.add_argument("--head", type=float, default=8.0, help="thickness of each interfacial layer (A), 0 = none")
ap.add_argument("--head-eps", type=float, default=20.0)
ap.add_argument("--width", type=float, default=1.5, help="smoothing width of the steps (A)")
ap.add_argument("--z0", type=float, default=0.0, help="z of the bilayer centre (default 0)")
ap.add_argument("--sigma", type=float, default=0.70, help="Gaussian variance used in run.prm")
ap.add_argument("--indi", type=float, default=8.0, help="Internal Dielectric used in run.prm")
ap.add_argument("--exdi", type=float, default=80.0, help="External Dielectric used in run.prm")
a = ap.parse_args()

# fixed columns: coordinates can touch each other (e.g. -30.284-100.454), so never split on whitespace
atoms = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54]), float(l[54:].split()[1])]
                  for l in open(a.pqr) if l.startswith(("ATOM", "HETATM"))])

# ---- grid ----------------------------------------------------------------------------------
h = a.spacing
half_slab = a.core / 2 + a.head
lo = atoms[:, :3].min(0) - np.array([a.pad_xy, a.pad_xy, a.pad_z])
hi = atoms[:, :3].max(0) + np.array([a.pad_xy, a.pad_xy, a.pad_z])
lo[2] = min(lo[2], a.z0 - half_slab - a.pad_z)
hi[2] = max(hi[2], a.z0 + half_slab + a.pad_z)
n = np.ceil((hi - lo) / h).astype(int) + 1
xs, ys, zs = [lo[i] + h * np.arange(n[i]) for i in range(3)]
print("grid", tuple(n), "=", n.prod() / 1e6, "M points")

# ---- membrane profile eps_env(z) (1-D, smooth) ---------------------------------------------
def step(z, edge):                     # 0 below `edge`, 1 above, smoothed over `width`
    return 0.5 * (1 + erf((z - edge) / (a.width * 0.7071)))

az = np.abs(zs - a.z0)
eps_z = np.full(zs.shape, a.exdi)
if a.head > 0:                         # water -> interface -> core, symmetric in |z|
    w_head = 1 - step(az, a.core / 2 + a.head)          # 1 inside the whole slab
    eps_z = a.exdi + (a.head_eps - a.exdi) * w_head
    w_core = 1 - step(az, a.core / 2)
    eps_z = eps_z + (a.core_eps - eps_z) * w_core
else:
    w_core = 1 - step(az, a.core / 2)
    eps_z = a.exdi + (a.core_eps - a.exdi) * w_core
eps_env = np.broadcast_to(eps_z[None, None, :], tuple(n)).copy()

# ---- protein Gaussian density (identical formula to the built-in model) --------------------
keep = np.ones(tuple(n))
for x, y, z, r in atoms:
    if r < 1e-6:
        continue
    cut = 3 * a.sigma * r
    ix = slice(max(int((x - cut - lo[0]) / h), 0), min(int((x + cut - lo[0]) / h) + 2, n[0]))
    iy = slice(max(int((y - cut - lo[1]) / h), 0), min(int((y + cut - lo[1]) / h) + 2, n[1]))
    iz = slice(max(int((z - cut - lo[2]) / h), 0), min(int((z + cut - lo[2]) / h) + 2, n[2]))
    X, Y, Z = np.meshgrid(xs[ix], ys[iy], zs[iz], indexing="ij")
    d2 = (X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2
    den = np.exp(-d2 / (a.sigma ** 4 * r * r))
    den[d2 > cut ** 2] = 0
    keep[ix, iy, iz] *= 1 - den
f = 1 - keep

eps = f * a.indi + (1 - f) * eps_env

# ---- write OpenDX (x slowest, z fastest) ---------------------------------------------------
with open(a.out, "w") as o:
    o.write("object 1 class gridpositions counts %d %d %d\n" % tuple(n))
    o.write("origin %g %g %g\n" % tuple(lo))
    o.write("delta %g 0 0\ndelta 0 %g 0\ndelta 0 0 %g\n" % (h, h, h))
    o.write("object 2 class gridconnections counts %d %d %d\n" % tuple(n))
    o.write("object 3 class array type double rank 0 items %d data follows\n" % eps.size)
    flat = eps.ravel()
    for i in range(0, flat.size, 3):
        o.write(" ".join("%.5g" % v for v in flat[i:i + 3]) + "\n")
print("wrote", a.out, "| eps range", eps.min(), eps.max())
