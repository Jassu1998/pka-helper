"""tautomer.py - carboxyl proton position for buried Asp/Glu (pka-helper --carboxyl-tautomers).

DelPhiPKa builds every neutral Asp/Glu with its proton on ONE fixed oxygen (OD1 / OE1, the run.prm keys "Hydrogen of ASP/GLU Attached to
Atom").  When a buried carboxyl is H-bonded through the OTHER oxygen (bacteriorhodopsin Asp96: OD2 ... Thr46 OG1 2.7 A, OD1 3.3 A), the
neutral state is built with its proton pointing away from its partner and is penalised instead of stabilised, so the pKa comes out too low.

Fix, after the normal energy stage:
  1. select acids for which the other placement could matter (mode "selected"): carboxylate oxygens buried (SASA of both < 20 A^2) and an
     H-bond acceptor (backbone or Asn/Gln carbonyl O, Ser/Thr/Tyr OH, Asp/Glu O, His N) within 3.3 A of the second oxygen that is closer
     than any acceptor of the first.  Mode "all-buried": every buried acid (to check the selection).
  2. re-run the energy stage for THOSE sites only ("Energy sites only" in run.prm) on a copy of the structure in which their two oxygens
     swap names - the proton then sits on the other oxygen; the charged state is symmetric and unaffected; the dielectric map is reused.
  3. per site keep the placement with the lower neutral-state energy (polar + desolvation), merge its energies and its pair-energy row
     (both directions) into energies.txt / pairwise.txt, and re-titrate ("Do Energy Calculation : F").  Originals: *_fixed_proton.txt.
Nothing is fitted.
"""
import os, re, shutil, subprocess, sys
import numpy as np

SEC = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
ACC = {("ASN", "OD1"), ("GLN", "OE1"), ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"), ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"),
       ("GLU", "OE2"), ("HIS", "ND1"), ("HIS", "NE2")}
RAD = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}


def _sph(n=150):
    k = np.arange(n); zz = 1 - 2 * (k + .5) / n; rr = np.sqrt(1 - zz * zz); ph = np.pi * (3 - 5 ** .5) * k
    return np.stack([rr * np.cos(ph), rr * np.sin(ph), zz], 1)


def hydrogen_oxygen(run_dir):
    """Which oxygen carries the proton in this run (from run.prm; DelPhiPKa's own default is OD2 / OE2 if not given)."""
    h = {"ASP": "OD2", "GLU": "OE2"}
    for l in open(os.path.join(run_dir, "run.prm")):
        m = re.match(r"\s*Hydrogen of (ASP|GLU) Attached to Atom\s*:\s*(\S+)", l)
        if m and m.group(2) in SEC[m.group(1)]: h[m.group(1)] = m.group(2)
    return h


def select(pqr, hox, mode="selected", sasa_max=20.0, cut=3.3):
    """-> {(resname, num, chain)} of acids whose other proton placement is worth computing."""
    from scipy.spatial import cKDTree
    at = [(l[17:20], l[21], int(l[22:26]), l[12:16].strip(), np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])) for l in open(pqr)
          if l.startswith("ATOM") and not l[12:16].strip().startswith("H")]
    X = np.array([a[4] for a in at]); R = np.array([RAD.get(a[3][0], 1.7) for a in at]); T = cKDTree(X); S = _sph()
    idx = {(a[1], a[2], a[3]): i for i, a in enumerate(at)}
    out = set()
    for (ch, num, name), i in idx.items():
        rt = at[i][0]
        if rt not in SEC or name != SEC[rt][0]: continue
        j = idx.get((ch, num, SEC[rt][1]))
        if j is None: continue
        sasa = 0.0
        for q in (i, j):
            p = X[q] + S * (R[q] + 1.4); nb = [m for m in T.query_ball_point(X[q], 6.0) if m != q]
            sasa += ((np.linalg.norm(p[:, None] - X[nb][None], axis=2) >= R[nb] + 1.4).all(1).mean() if nb else 1.0) * 4 * np.pi * (R[q] + 1.4) ** 2
        if sasa >= sasa_max: continue
        if mode == "all-buried": out.add((rt, num, ch)); continue
        hq = idx[(ch, num, hox[rt])]; oq = j if hq == i else i            # oxygen with the proton / the other one
        def best(q):
            d = [np.linalg.norm(X[m] - X[q]) for m in T.query_ball_point(X[q], cut) if not (at[m][1] == ch and at[m][2] == num) and (at[m][3] == "O" or (at[m][0], at[m][3]) in ACC)]
            return min(d) if d else 99.0
        if best(oq) < cut and best(oq) < best(hq): out.add((rt, num, ch))
    return out


def _site(k): return "%s%04d%s" % (k[0], k[1], k[2])


def swapped_pdb(src, dst, sites):
    keys = {(k[2], k[1]) for k in sites}; out = []
    for l in open(src):
        if l.startswith(("ATOM", "HETATM")) and l[17:20] in SEC and (l[21], int(l[22:26])) in keys:
            a, b = SEC[l[17:20]]; nm = l[12:16].strip()
            if nm in (a, b): l = l[:12] + (" %-3s" % (b if nm == a else a)) + l[16:]
        out.append(l)
    open(dst, "w").write("".join(out))


def read_energies(p):
    d = {}
    for l in list(open(p))[1:]:
        t = l.split()
        if len(t) >= 3: d[t[0]] = (float(t[1]), float(t[2]))
    return d


def ename(k, state):    # energies.txt naming: ASP0A0096 (neutral) / ASP-A0096 (charged acid)
    return "%s%s%s%04d" % (k[0], state, k[2], k[1])


def strong_partners(run_dir, sites, kt):
    """Sites coupled to a selected acid by a charged-charged pair energy above kt (kT) in the first run: computing them too makes the
    pair energy with the switched acid the average of both directions, as in a full run."""
    e = [l.split()[0] for l in list(open(os.path.join(run_dir, "energies.txt")))[1:] if l.strip()][0::2]
    names = [(n[:3], int(n[5:9]), n[4]) for n in e]; N = len(names); pos = {k: i for i, k in enumerate(names)}
    p = open(os.path.join(run_dir, "pairwise.txt")).read().splitlines()[1:]
    out = set()
    for k in sites:
        i = pos.get(k)
        if i is None: continue
        for j in range(N):
            if j != i and abs(float(p[(i * N + j) * 4].split()[-1])) > kt: out.add(names[j])
    return sorted(out - set(sites), key=lambda k: (k[2], k[1]))


def run(run_dir, mode, mpi_prefix, exe, log=print, partner_kt=0.0):
    """mpi_prefix: e.g. ['mpirun', '--allow-run-as-root', '-np', '2'] or [].  Returns a list of report lines."""
    prm = open(os.path.join(run_dir, "run.prm")).read()
    hox = hydrogen_oxygen(run_dir)
    sites = sorted(select(os.path.join(run_dir, "protein_noHET.pqr"), hox, mode), key=lambda k: (k[2], k[1]))
    if not sites: return ["carboxyl tautomers: no buried acid needs the other proton placement"]
    log("carboxyl tautomers: other proton placement for %d acid(s): %s" % (len(sites), " ".join("%s%d%s" % (k[0], k[1], k[2]) for k in sites)))
    sub = os.path.join(run_dir, "tautomer")
    shutil.rmtree(sub, ignore_errors=True); os.makedirs(sub)
    pdb = re.search(r"PDB file name\s*:\s*(\S+)", prm).group(1)
    for f in os.listdir(run_dir):
        p = os.path.join(run_dir, f)
        if f == "param": shutil.copytree(p, os.path.join(sub, f))
        elif f.endswith(".dx") and f not in ("protein.dx", "membrane.dx") or f == "membrane_stack.txt":
            os.symlink(os.path.abspath(p), os.path.join(sub, f))
    swapped_pdb(os.path.join(run_dir, pdb), os.path.join(sub, pdb), sites)
    t = re.sub(r"(?m)^Do pKa's  Calculation\s*:.*$", "Do pKa's  Calculation              : F", prm)
    t = re.sub(r"(?m)^Energy checkpoint directory.*\n?", "", t)
    extra = strong_partners(run_dir, sites, partner_kt) if partner_kt > 0 else []
    t += "Energy sites only                  : %s\n" % ",".join(_site(k) for k in sites + extra)
    open(os.path.join(sub, "run.prm"), "w").write(t)
    with open(os.path.join(sub, "run.log"), "w") as lg:
        rc = subprocess.run(mpi_prefix + [exe, "run.prm"], cwd=sub, stdout=lg, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not os.path.isfile(os.path.join(sub, "energies.txt")):
        return ["carboxyl tautomers: the second calculation failed (exit %d, see %s); results use the fixed proton placement" % (rc, sub)]
    E1 = read_energies(os.path.join(run_dir, "energies.txt")); E2 = read_energies(os.path.join(sub, "energies.txt"))
    switch, rep = [], ["carboxyl tautomers (neutral-state energy, kcal/mol: proton on %s / %s):" % ("OD1|OE1" if hox["ASP"] == "OD1" else "default", "other O")]
    for k in sites:
        n0, n1 = ename(k, "0"), ename(k, "0")
        if n0 not in E1 or n0 not in E2: continue
        g1, g2 = sum(E1[n0]), sum(E2[n0]); c1, c2 = sum(E1[ename(k, "-")]), sum(E2[ename(k, "-")])
        use = g2 < g1
        rep.append("  %s%d %s  %7.2f  %7.2f  -> %s%s" % (k[0], k[1], k[2], g1, g2, "other O" if use else "kept",
                   "" if abs(c1 - c2) < 0.05 else "   (charged state differs by %.2f: check)" % (c2 - c1)))
        if use: switch.append(k)
    if not switch:
        rep.append("  no change"); return rep
    for f in ("energies.txt", "pairwise.txt"):
        b = os.path.join(run_dir, f.replace(".txt", "_fixed_proton.txt"))
        if not os.path.isfile(b): shutil.copy(os.path.join(run_dir, f), b)
    # Both runs list the sites in the same order (same atoms).  energies.txt: two lines per site (neutral, charged).  pairwise.txt: for
    # site i, for site j: 4 lines (i+ j+, i+ j0, i0 j+, i0 j0); DelPhiPKa prints the chain of i on both names, so rows are matched by
    # POSITION, never by name.  Switched site s: its own rows come from the second run; the reverse rows (j, s) get the same values.
    e1 = open(os.path.join(run_dir, "energies.txt")).read().splitlines(True); e2 = open(os.path.join(sub, "energies.txt")).read().splitlines(True)
    order = [l.split()[0] for l in e1[1:] if l.strip()][0::2]                   # neutral-state names, one per site, in site order
    def site_of(n): return (n[:3], int(n[5:9]), n[4])
    pos = {site_of(n): i for i, n in enumerate(order)}; N = len(order)
    sw = [pos[k] for k in switch if k in pos]
    for i in sw:
        for q in (1 + 2 * i, 2 + 2 * i): e1[q] = e2[q]
    open(os.path.join(run_dir, "energies.txt"), "w").write("".join(e1))
    p1 = open(os.path.join(run_dir, "pairwise.txt")).read().splitlines(True); p2 = open(os.path.join(sub, "pairwise.txt")).read().splitlines(True)
    if len(p1) != len(p2) or len(p1) != 1 + 4 * N * N:
        return rep + ["carboxyl tautomers: pairwise tables do not match (%d / %d lines, %d sites); results use the fixed proton placement" % (len(p1), len(p2), N)]
    val = lambda l: l.rsplit(None, 1)[1]
    def setv(l, v): return l[:33] + "%10s\n" % v               # DelPhiPKa reads the value from fixed columns (33..42)
    swset = set(sw)
    for i in sw:
        for j in range(N):
            for a in (0, 1):
                for b in (0, 1):
                    fwd = 1 + (i * N + j) * 4 + 2 * a + b; rev = 1 + (j * N + i) * 4 + 2 * b + a
                    p1[fwd] = setv(p1[fwd], val(p2[fwd])); p1[rev] = setv(p1[rev], val(p2[rev]))   # run 2 already holds both directions
    open(os.path.join(run_dir, "pairwise.txt"), "w").write("".join(p1))
    # re-titrate
    shutil.copy(os.path.join(run_dir, "pKa.csv"), os.path.join(run_dir, "pKa_fixed_proton.csv"))
    t = re.sub(r"(?m)^Do Energy Calculation\s*:.*$", "Do Energy Calculation              : F", prm)
    t = re.sub(r"(?m)^Energy checkpoint directory.*\n?", "", t)
    open(os.path.join(run_dir, "retitrate.prm"), "w").write(t)
    with open(os.path.join(run_dir, "retitrate.log"), "w") as lg:
        rc = subprocess.run([exe, "retitrate.prm"], cwd=run_dir, stdout=lg, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        for f in ("energies", "pairwise"): shutil.copy(os.path.join(run_dir, f + "_fixed_proton.txt"), os.path.join(run_dir, f + ".txt"))
        shutil.copy(os.path.join(run_dir, "pKa_fixed_proton.csv"), os.path.join(run_dir, "pKa.csv"))
        return rep + ["carboxyl tautomers: re-titration failed (exit %d, see retitrate.log); results use the fixed proton placement" % rc]
    try:
        import group_pka, io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):      # cluster table for clusters_macro.csv; its printout is not for the screen
            group_pka.main([run_dir])
    except Exception:
        pass
    return rep + ["  %d acid(s) switched; energies re-titrated (originals: *_fixed_proton.*)" % len(switch)]
