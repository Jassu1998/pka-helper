"""structure_check.py - warnings about the INPUT STRUCTURE that make a predicted pKa unreliable.

The largest errors in the benchmark came from the structures, not from the electrostatics (NhaA 1ZCD without the Asp163-Lys300 salt
bridge; BM2 2KIX with His27 facing the lipid; OmpF Asp256).  Each left a signature that can be checked:

  protonated-geometry  a buried Asp/Glu (carboxylate O exposed < 10 A^2) whose carboxylate O is <= 3.1 A from a backbone carbonyl O and
                       has no Lys/Arg/His N within 4 A.  Two O atoms that close share a proton, and a backbone carbonyl only accepts one,
                       so the structure shows a NEUTRAL acid.  Continuum methods do not see this H bond: expect a higher pKa than predicted
                       (OmpF Asp127 and Asp256, CpHMD > 8; both flagged, nothing flagged in 160 soluble PKAD-R acids).
  deep-lipid           a titratable group facing the lipid (>= 30 % of its 3-8 A shell is lipid core, i.e. at the core dielectric and outside
                       the protein) at least 5 A inside the core: the
                       result depends strongly on the membrane position and on the orientation of the side chain (BM2 His27).
  saturated            DelPhi value at the edge of its range (marked * by DelPhiPKa): only "very high" / "very low" is known.
  missing-atoms        side-chain heavy atoms missing on a titratable residue, or a chain break within 8 A of its titrating atoms.
  input                several MODELs (only the first is used), alternate locations, resolution worse than 3 A.

    from structure_check import check
    warnings = check(pdb, run_dir)     # -> list of (key or None, code, message); key = (resname, number, chain)
"""
import os, re
import numpy as np

CARB = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
CATION = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("HIS", "ND1"), ("HIS", "NE2"), ("LYR", "NZ")}
KEYATOMS = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"), "LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2"), "HIS": ("ND1", "NE2"),
            "CYS": ("SG",), "TYR": ("OH",)}
SIDECHAIN = {"ASP": ("CB", "CG", "OD1", "OD2"), "GLU": ("CB", "CG", "CD", "OE1", "OE2"), "LYS": ("CB", "CG", "CD", "CE", "NZ"),
             "ARG": ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"), "HIS": ("CB", "CG", "ND1", "CD2", "CE1", "NE2"),
             "CYS": ("CB", "SG"), "TYR": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH")}
RAD = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}


def _atoms(pdb):
    at, models, altloc, resol = [], 0, False, None
    for l in open(pdb, errors="replace"):
        if l.startswith("MODEL"): models += 1
        if l.startswith("ENDMDL") and models >= 1:
            break                                                    # first model only (that is what the calculation uses)
        if l.startswith("REMARK   2 RESOLUTION"):
            m = re.search(r"([\d.]+)\s*ANGSTROM", l)
            if m: resol = float(m.group(1))
        if l.startswith(("ATOM", "HETATM")) and l[17:20] not in ("DUM", "HOH", "WAT"):
            name = l[12:16].strip()
            if name.startswith("H") or (len(l) >= 78 and l[76:78].strip() == "H"): continue
            if l[16] not in (" ", "A"): altloc = True; continue
            if l[16] == "A": altloc = True
            try: xyz = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
            except ValueError: continue
            at.append((l[17:20], l[21], int(l[22:26]), name, xyz))
    return at, models, altloc, resol


IGNORED_OK = {"DUM", "HOH", "WAT", "MSE", "SO4", "PO4", "GOL", "PEG", "EDO", "CL", "NA", "K", "LMU", "LMT", "BOG", "OLC", "OLA", "OLB", "C8E",
              "DMU", "LDA", "UNL", "SQU", "LI1", "XYL", "NH2", "ACE"}


def hetero_groups(pdb):
    """Non-water HETATM residue names in the file (they are not part of the calculation)."""
    n = {}
    for l in open(pdb, errors="replace"):
        if l.startswith("HETATM"):
            rn = l[17:20].strip()
            n.setdefault(rn, set()).add((l[21], l[22:27]))
    return {k: len(v) for k, v in n.items() if k not in ("DUM", "HOH", "WAT", "MSE")}


def _model_count(pdb):
    return sum(1 for l in open(pdb, errors="replace") if l.startswith("MODEL"))


def _sasa_of(idx, X, R, tree, n=200):
    k = np.arange(n); zz = 1 - 2 * (k + 0.5) / n; rr = np.sqrt(1 - zz * zz); ph = np.pi * (3 - 5 ** 0.5) * k
    sph = np.stack([rr * np.cos(ph), rr * np.sin(ph), zz], 1); tot = 0.0
    for i in idx:
        p = X[i] + sph * (R[i] + 1.4); nb = [j for j in tree.query_ball_point(X[i], R[i] + 1.4 + 1.8 + 1.4) if j != i]
        free = (np.linalg.norm(p[:, None] - X[nb][None], axis=2) >= R[nb] + 1.4).all(1) if nb else np.ones(n, bool)
        tot += free.mean() * 4 * np.pi * (R[i] + 1.4) ** 2
    return tot


def protonated_geometry(at, cut=3.1, sasa_max=10.0, sb_cut=4.0):
    from scipy.spatial import cKDTree
    X = np.array([a[4] for a in at]); R = np.array([RAD.get(a[3][0], 1.7) for a in at]); tree = cKDTree(X)
    cat = np.array([X[i] for i, a in enumerate(at) if (a[0], a[3]) in CATION])
    groups = {}
    for i, a in enumerate(at):
        if a[0] in CARB and a[3] in CARB[a[0]]: groups.setdefault((a[0], a[2], a[1]), []).append(i)
    out = []
    for k, ix in groups.items():
        best = None
        for i in ix:
            for j in tree.query_ball_point(X[i], cut):
                b = at[j]
                if b[3] == "O" and not (b[1] == k[2] and b[2] == k[1]):
                    d = float(np.linalg.norm(X[j] - X[i]))
                    if best is None or d < best[0]: best = (d, "%s%d%s" % (b[0], b[2], b[1].strip()))
        if not best: continue
        if len(cat) and min(np.linalg.norm(cat - X[i], axis=1).min() for i in ix) < sb_cut: continue
        if _sasa_of(ix, X, R, tree) >= sasa_max: continue
        out.append((k, "protonated-geometry", "buried, carboxylate O %.1f A from the backbone O of %s and no Lys/Arg/His within %g A: the structure shows a "
                    "NEUTRAL (protonated) acid; continuum methods do not see this H bond, so the real pKa is probably HIGHER than predicted" % (best[0], best[1], sb_cut)))
    return out


def missing_atoms(at, near=8.0):
    from scipy.spatial import cKDTree
    res = {}
    for a in at: res.setdefault((a[1], a[2], a[0]), {})[a[3]] = np.array(a[4])
    out = []
    for (ch, n, rn), d in res.items():
        if rn in SIDECHAIN:
            miss = [x for x in SIDECHAIN[rn] if x not in d]
            if miss: out.append(((rn, n, ch), "missing-atoms", "side-chain atoms missing: %s (the calculation rebuilds or ignores them; the pKa is a guess)" % " ".join(miss)))
    # chain breaks: consecutive numbers with C(i)-N(i+1) > 2.0 A, or a jump in numbering inside a chain
    breaks = []
    keys = sorted((k for k in res if "CA" in res[k]), key=lambda k: (k[0], k[1]))      # amino acids only
    for k0, k1 in zip(keys, keys[1:]):
        if k0[0] != k1[0]: continue
        c, nn = res[k0].get("C"), res[k1].get("N")
        if (k1[1] - k0[1] > 1) or (c is not None and nn is not None and np.linalg.norm(c - nn) > 2.0):
            breaks.append((k0, k1, (res[k0]["CA"] + res[k1]["CA"]) / 2.0))
    if breaks:
        pts = np.array([b[2] for b in breaks]); tr = cKDTree(pts)
        for (ch, n, rn), d in res.items():
            if rn in KEYATOMS:
                ka = [d[x] for x in KEYATOMS[rn] if x in d]
                hit = sorted({i for x in ka for i in tr.query_ball_point(x, near)})
                if hit:
                    b = breaks[hit[0]]
                    out.append(((rn, n, ch), "missing-atoms", "chain break (missing residues) between %s%d and %s%d within %g A: nearby charges or a "
                                "channel wall may be missing" % (b[0][0], b[0][1], b[1][0], b[1][1], near)))
    return out


def deep_lipid(run_dir, depth_min=5.0, lip_min=0.3):
    try:
        import lipid_auto as L
        from epsmap import read_dx
    except ImportError:
        return []
    if not all(os.path.isfile(os.path.join(run_dir, f)) for f in ("background.dx", "protein_noHET.pqr")): return []
    depth = L.core_depth(run_dir)
    if not any(v >= depth_min for v in depth.values()): return []
    at = L._pqr(os.path.join(run_dir, "protein_noHET.pqr")); m = read_dx(os.path.join(run_dir, "background.dx"))
    rho = L._density(at, m); bg = m.data; org = np.array(m.origin); dl = np.array(m.delta); n = bg.shape
    core = L.lipid_eps_threshold(run_dir) - 5.0 if L.lipid_eps_threshold(run_dir) > 10.0 else 2.0
    try: core = min(float(l.split()[1]) for l in open(os.path.join(run_dir, "membrane_stack.txt")) if l.strip() and not l.startswith("#"))
    except (OSError, ValueError, IndexError): pass
    sites = {}
    for a in at:
        if a[1] in L.KEYATOMS and a[0] in L.KEYATOMS[a[1]]: sites.setdefault((a[1], a[3], a[2]), []).append((a[4], a[5], a[6]))
    out = []
    for k, xs in sites.items():
        if depth.get(k, 0.0) < depth_min: continue
        c = np.mean(xs, 0)
        i0 = np.maximum(np.floor((c - 9 - org) / dl).astype(int), 0); i1 = np.minimum(np.ceil((c + 9 - org) / dl).astype(int) + 1, n)
        g = [org[j] + dl[j] * np.arange(i0[j], i1[j]) for j in range(3)]
        X, Y, Z = np.meshgrid(*g, indexing="ij"); r = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)
        sl = tuple(slice(i0[j], i1[j]) for j in range(3)); sh = (r >= 3) & (r <= 8)
        lip = float(np.mean((np.abs(bg[sl][sh] - core) < 0.3) & (rho[sl][sh] < 0.3)))   # lipid = the core value itself, not the eps-4 protein zone
        if lip >= lip_min:
            out.append((k, "deep-lipid", "faces the lipid %.0f A inside the membrane core (%.0f%% of its surroundings is lipid): the result depends strongly on "
                        "the membrane position and the side-chain orientation; check them (e.g. --membrane-scan)" % (depth[k], 100 * lip)))
    return out


def saturated(run_dir):
    p = os.path.join(run_dir, "pKa.csv"); out = []
    if not os.path.isfile(p): return out
    for l in list(open(p))[1:]:
        t = l.split()
        if len(t) >= 2 and t[1].endswith("*"):
            m = re.match(r"([A-Z]{3})(\d{4})(.)", t[0])
            if m: out.append(((m.group(1), int(m.group(2)), m.group(3)), "saturated", "DelPhi value %s is at the edge of its range: only 'very %s' is known"
                                % (t[1], "high" if float(t[1].rstrip("*")) > 7 else "low")))
    return out


def check(pdb, run_dir=None, kept=()):
    at, _, altloc, resol = _atoms(pdb); models = _model_count(pdb)
    w = []
    if models > 1: w.append((None, "input", "%d MODELs in the file: only the first is used (use --ensemble to run them all)" % models))
    if altloc: w.append((None, "input", "alternate locations present: only the first (blank or A) is used"))
    het = hetero_groups(pdb)
    if isinstance(kept, dict):                  # hetero.classify() plan: report what was kept, with which charge, and what was left out
        import hetero as HT
        plan = {k: v for k, v in kept.items() if k in het or k == "MSE"}
        for line in HT.describe({k: v for k, v in plan.items() if v[0] != "parent"}, {k: ("", n, 0) for k, n in het.items()}):
            w.append((None, "input", "HETATM groups " + line))
    else:                                        # old interface: list of --cofactor specs, everything else left out
        kept = {x.split(":")[0].strip().upper() for x in kept}
        het = {k: v for k, v in het.items() if k not in kept}
        if het:
            notable = {k: v for k, v in het.items() if k not in IGNORED_OK}
            w.append((None, "input", "HETATM groups are left out of the calculation: %s%s" % (", ".join("%s x%d" % kv for kv in sorted(het.items())),
                      ". Charged cofactors or ligands among them (e.g. a retinal Schiff base, metal ions, bound substrates) change nearby pKas and are missing"
                      if notable else " (detergent, lipid or buffer molecules; usually harmless)")))
    if resol is not None and resol > 3.0: w.append((None, "input", "resolution %.2f A: side-chain positions are uncertain, so are the pKas of buried groups" % resol))
    if at:
        w += protonated_geometry(at)
        w += missing_atoms(at)
    if run_dir:
        w += deep_lipid(run_dir)
        w += saturated(run_dir)
    return w


def render(warnings):
    if not warnings: return "Structure checks: nothing to report."
    out = ["Structure checks (what each one means: README.md, section 'Structure checks')", ""]
    for code in ("input", "protonated-geometry", "deep-lipid", "missing-atoms", "saturated"):
        ws = [x for x in warnings if x[1] == code]
        if not ws: continue
        out.append("%s (%d)" % (code, len(ws)))
        for k, _, msg in sorted(ws, key=lambda x: (x[0] or ("", 0, ""))[2:] + (x[0] or ("", 0, ""))[1:2] if x[0] else ("", 0)):
            out.append("  %-12s %s" % ("%s%d %s" % (k[0], k[1], k[2]) if k else "", msg))
        out.append("")
    return "\n".join(out)
