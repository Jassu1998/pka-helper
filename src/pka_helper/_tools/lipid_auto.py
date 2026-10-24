"""lipid_auto.py - find Asp/Glu that face the LIPID, from the membrane background map written by membrane_pka.py.

    from lipid_auto import lipid_facing
    LIP = lipid_facing('run_dir')      # -> {(resname, number, chain): fraction}  for acids that qualify

For each Asp/Glu: lip = fraction of the 3-8 A shell around the carboxylate that is membrane core in the background map
(background eps < 10, or core eps + 5 for a higher --eps-core) and not protein (Gaussian protein density < 0.3).  An acid is lipid-facing if lip >= 0.3 AND no Arg/Lys/His
nitrogen lies within 8 A of its carboxylate O (nothing to stabilise the charge).  No slab/z assumption, so it also works with --opm-curved.
Validated only partly: it fixes ATP synthase c-ring Glu58 (6ZBB) but costs ~0.07 pK MAE and 1 pH 7 call on the 91-site benchmark - see CUSTOM_DIELECTRIC.md.
"""
import os, numpy as np
from epsmap import read_dx
SIG = 0.7
def _pqr(p):
    at = []
    for l in open(p):
        if l.startswith(("ATOM", "HETATM")):          # HETATM: groups kept in the calculation (ligands, ions ...) are part of the protein body
            t = l[54:].split(); at.append((l[12:16].strip(), l[17:20].strip(), l[21], int(l[22:26]), float(l[30:38]), float(l[38:46]), float(l[46:54]), float(t[1])))
    return at
def _density(at, m, sigma=SIG):
    n = m.data.shape; org = np.array(m.origin); d = np.array(m.delta); keep = np.ones(n, dtype=np.float32)
    for nm, rn, ch, ri, x, y, z, r in at:
        if r < 1e-6: continue
        rc = 3 * sigma * r; c = (np.array([x, y, z]) - org) / d
        i0 = np.maximum(np.floor(c - rc / d).astype(int), 0); i1 = np.minimum(np.ceil(c + rc / d).astype(int) + 1, n)
        g = [org[k] + d[k] * np.arange(i0[k], i1[k]) - (x, y, z)[k] for k in range(3)]
        d2 = g[0][:, None, None] ** 2 + g[1][None, :, None] ** 2 + g[2][None, None, :] ** 2
        den = np.exp(-d2 / (sigma ** 4 * r * r)); den[d2 > rc * rc] = 0
        keep[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] *= (1 - den).astype(np.float32)
    return 1 - keep
def lipid_eps_threshold(rundir):
    """Background eps below which a voxel counts as membrane core: 10 for the usual eps-2 core; for a higher core dielectric
    (--eps-core) 5 above the core value, read from the undisturbed layer profile membrane_stack.txt (falls back to 10)."""
    try:
        es = [float(l.split()[1]) for l in open(os.path.join(rundir, "membrane_stack.txt")) if l.strip() and not l.startswith("#")]
        return max(10.0, min(es) + 5.0) if es else 10.0
    except (OSError, ValueError, IndexError):
        return 10.0
def lipid_facing(rundir, thr=0.3, cat_cut=8.0, report=None):
    at = _pqr(os.path.join(rundir, "protein_noHET.pqr")); m = read_dx(os.path.join(rundir, "background.dx"))
    rho = _density(at, m); bg = m.data; org = np.array(m.origin); dl = np.array(m.delta); n = bg.shape; thr_eps = lipid_eps_threshold(rundir)
    cat = np.array([(a[4], a[5], a[6]) for a in at if (a[1] in ("ARG", "LYS") and a[0] in ("NZ", "NE", "NH1", "NH2")) or (a[1] in ("HIS", "HSP", "HIP") and a[0] in ("ND1", "NE2"))])
    G = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
    sites = {}
    for a in at:
        if a[1] in G and a[0] in G[a[1]]: sites.setdefault((a[1], a[3], a[2]), []).append((a[4], a[5], a[6]))
    out = {}
    for k, ox in sites.items():
        c = np.mean(ox, 0)
        i0 = np.maximum(np.floor((c - 9 - org) / dl).astype(int), 0); i1 = np.minimum(np.ceil((c + 9 - org) / dl).astype(int) + 1, n)
        g = [org[j] + dl[j] * np.arange(i0[j], i1[j]) for j in range(3)]
        X, Y, Z = np.meshgrid(*g, indexing="ij"); r = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)
        sl = tuple(slice(i0[j], i1[j]) for j in range(3)); sh = (r >= 3) & (r <= 8)
        lip = float(np.mean((bg[sl][sh] < thr_eps) & (rho[sl][sh] < 0.3)))
        ncat = int(np.sum(np.min(np.linalg.norm(cat[:, None] - np.array(ox)[None], axis=2), axis=1) < cat_cut)) if len(cat) else 0
        if report is not None: report[k] = (lip, ncat)
        if lip >= thr and ncat == 0: out[k] = lip
    return out

def lipid_facing_bases(rundir, thr=0.3, an_cut=6.0, report=None):
    """Lys (and, for the report only, Arg) that face the lipid: >= thr of the 3-8 A shell around the side-chain N atom(s) is membrane core
    (background eps < 10, protein density < 0.3) and no Asp/Glu carboxylate O lies within an_cut A (no anion to stabilise the charge).
    -> {(LYS, number, chain): fraction} for Lys only.  Arg is recorded in `report` ((lip, anions)) but never returned: simulations of Arg in a
    bilayer keep it charged (snorkelling), so its DelPhi shift is not trusted to be lifted."""
    at = _pqr(os.path.join(rundir, "protein_noHET.pqr")); m = read_dx(os.path.join(rundir, "background.dx"))
    rho = _density(at, m); bg = m.data; org = np.array(m.origin); dl = np.array(m.delta); n = bg.shape; thr_eps = lipid_eps_threshold(rundir)
    an = np.array([(a[4], a[5], a[6]) for a in at if (a[1] == "ASP" and a[0] in ("OD1", "OD2")) or (a[1] == "GLU" and a[0] in ("OE1", "OE2"))])
    G = {"LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2")}
    sites = {}
    for a in at:
        if a[1] in G and a[0] in G[a[1]]: sites.setdefault((a[1], a[3], a[2]), []).append((a[4], a[5], a[6]))
    out = {}
    for k, ox in sites.items():
        c = np.mean(ox, 0)
        i0 = np.maximum(np.floor((c - 9 - org) / dl).astype(int), 0); i1 = np.minimum(np.ceil((c + 9 - org) / dl).astype(int) + 1, n)
        g = [org[j] + dl[j] * np.arange(i0[j], i1[j]) for j in range(3)]
        X, Y, Z = np.meshgrid(*g, indexing="ij"); r = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)
        sl = tuple(slice(i0[j], i1[j]) for j in range(3)); sh = (r >= 3) & (r <= 8)
        lip = float(np.mean((bg[sl][sh] < thr_eps) & (rho[sl][sh] < 0.3)))
        nan = int(np.sum(np.min(np.linalg.norm(an[:, None] - np.array(ox)[None], axis=2), axis=1) < an_cut)) if len(an) else 0
        if report is not None: report[k] = (lip, nan)
        if k[0] == "LYS" and lip >= thr and nan == 0: out[k] = lip
    return out


KEYATOMS = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"), "LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2"), "HIS": ("ND1", "NE2"), "HSP": ("ND1", "NE2"), "HIP": ("ND1", "NE2")}


def core_depth(rundir, core_eps=40.0):
    """-> {(resname, number, chain): depth in A}: how far the titrating atoms lie INSIDE the membrane core, measured from the background dielectric
    map (core = eps < core_eps) as the straight-line distance to the nearest voxel that is not core.  0 for a group outside the core.
    Direction-free, so it works for a curved or tilted membrane.  The protein is not subtracted: the question is how deep the group is in the lipid
    slab, not how much of it is exposed."""
    from scipy.ndimage import distance_transform_edt
    at = _pqr(os.path.join(rundir, "protein_noHET.pqr")); m = read_dx(os.path.join(rundir, "background.dx"))
    org = np.array(m.origin); dl = np.array(m.delta)
    edt = distance_transform_edt(m.data < core_eps, sampling=tuple(dl))
    sites = {}
    for a in at:
        if a[1] in KEYATOMS and a[0] in KEYATOMS[a[1]]: sites.setdefault((a[1], a[3], a[2]), []).append((a[4], a[5], a[6]))
    out = {}
    for k, xs in sites.items():
        c = np.mean(xs, 0); i = np.round((c - org) / dl).astype(int)
        out[k] = float(edt[tuple(i)]) if np.all(i >= 0) and np.all(i < edt.shape) else 0.0
    return out


def depth_factor(depth, d0=3.0, d1=10.0):
    """0 for a group within d0 A of the core boundary (snorkelling / interface: the continuum map over-penalises it), 1 beyond d1 A."""
    return float(min(max((depth - d0) / (d1 - d0), 0.0), 1.0))
