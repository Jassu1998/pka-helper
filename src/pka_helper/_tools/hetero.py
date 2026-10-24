"""hetero.py - what pka-helper does with HETATM groups (ligands, cofactors, bound lipids, ions, modified residues).

If HETATM groups were simply left out, the space each filled would become water (bulk water, or pore water inside the membrane):
a removed retinal or substrate became a water pocket next to the groups that bind it.  Now, by default:

  parent   modified residues whose charge is that of the standard residue are read as that residue (atoms the standard residue does not
           have, e.g. the methyl groups of dimethyl-lysine, are left out): MSE, FME -> MET, HYP -> PRO, MLY, MLZ -> LYS, ...
           They are titrated like the standard residue and PROPKA sees them.
  charged  groups with a known fixed charge stay in the calculation as fixed charges on the right atoms plus their volume:
           phospho-Ser/Thr/Tyr (-2), carboxylated Lys (-1), trimethyl-Lys (+1), retinal-Lys Schiff base (LYR, +1), negatively charged
           lipids (cardiolipin -2, phosphatidylglycerol -1, ...).  Not titrated.
  ion      single metal / halide ions are left out by default and listed: crystals often carry ions from the crystallisation buffer that
           are absent where pKas are measured (barnase Zn, RNase T1 Ca: keeping them made those benchmark pKas 1.2 pK worse).  An ion is
           one atom, so leaving it out does not open a pocket.  --keep-ions keeps them all with their usual charge, or keep one kind with
           --cofactor ZN:+2.
  neutral  everything else (retinal, heme, substrates, neutral lipids, unknown ligands) stays in as an uncharged low-dielectric volume.
           Its charge is unknown to pka-helper: give it with --cofactor RES:CHARGE if you know it.
  drop     crystallisation additives, buffer molecules and detergents (sulfate, glycerol, PEG, maltosides, ...): in solution that space is
           water, so they are left out.  Waters and the OPM dummy atoms are always left out.

--hetero drop restores the old behaviour (everything left out except selenomethionine); --drop-het RES,RES leaves out chosen groups;
--cofactor RES:CHARGE keeps a group with that net charge.

    from hetero import classify, map_parents, write_delphi_input
"""
import os, re

BACKBONE = ("N", "CA", "C", "O", "OXT")
HEAVY = {"MET": ("CB", "CG", "SD", "CE"), "PRO": ("CB", "CG", "CD"), "LYS": ("CB", "CG", "CD", "CE", "NZ"), "CYS": ("CB", "SG"),
         "SER": ("CB", "OG"), "ALA": ("CB",), "ARG": ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"), "THR": ("CB", "OG1", "CG2"),
         "TYR": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"), "GLY": ()}

# modified residue -> (standard residue, {atom name in the modified residue: atom name in the standard one}); same charge as the parent
PARENT = {
    "MSE": ("MET", {"SE": "SD"}),        # selenomethionine
    "FME": ("MET", {}),                  # N-formylmethionine (formyl left out)
    "HYP": ("PRO", {}),                  # 4-hydroxyproline
    "MLY": ("LYS", {}),                  # N,N-dimethyl-lysine (titratable)
    "MLZ": ("LYS", {}),                  # N-methyl-lysine
    "CSO": ("CYS", {}),                  # S-hydroxycysteine
    "CME": ("CYS", {}),                  # S,S-(2-hydroxyethyl)thiocysteine
    "SAC": ("SER", {}),                  # N-acetylserine (N-terminal)
    "AME": ("MET", {}),                  # N-acetylmethionine (N-terminal)
    "AYA": ("ALA", {}),                  # N-acetylalanine (N-terminal)
    "2MR": ("ARG", {}),                  # N3,N4-dimethylarginine
    "AGM": ("ARG", {}),                  # 5-methylarginine
}

# fixed charge: resname -> (net charge, atom names that carry it; empty = spread over the heavy atoms)
FIXED = {
    "SEP": (-2.0, ("O1P", "O2P", "O3P", "OP1", "OP2", "OP3")),   # phosphoserine
    "TPO": (-2.0, ("O1P", "O2P", "O3P", "OP1", "OP2", "OP3")),   # phosphothreonine
    "PTR": (-2.0, ("O1P", "O2P", "O3P", "OP1", "OP2", "OP3")),   # phosphotyrosine
    "KCX": (-1.0, ("OQ1", "OQ2")),                               # lysine NZ-carboxylic acid
    "CSD": (-1.0, ("OD1", "OD2")),                               # cysteine sulfinic acid
    "TYS": (-1.0, ("O1", "O2", "O3", "OS1", "OS2", "OS3")),      # sulfotyrosine
    "M3L": (+1.0, ("NZ",)),                                      # trimethyl-lysine (always charged)
    "LYR": (+1.0, ("NZ",)),                                      # retinal-lysine: protonated Schiff base
    # negatively charged lipids: one -1 per phosphate (charge on the P atoms)
    "CDL": (-2.0, "P"), "LHG": (-1.0, "P"), "PGV": (-1.0, "P"), "PGW": (-1.0, "P"), "PGT": (-1.0, "P"), "D39": (-1.0, "P"),
    "PSF": (-1.0, "P"),
}
IONS = {"CA": 2, "MG": 2, "ZN": 2, "MN": 2, "FE2": 2, "FE": 3, "CU": 2, "CU1": 1, "NI": 2, "CO": 2, "CD": 2, "SR": 2, "BA": 2,
        "NA": 1, "K": 1, "LI": 1, "RB": 1, "CS": 1, "CL": -1, "BR": -1, "IOD": -1, "F": -1}
# crystallisation additives, buffers, detergents: left out (in solution this space is water)
ADDITIVES = {"SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "P6G", "2PE", "12P", "15P", "P33", "PE4", "PE5", "MPD", "MRD", "DMS",
             "ACT", "ACY", "FMT", "IPA", "EOH", "MOH", "TRS", "MES", "EPE", "CIT", "FLC", "BME", "DTT", "IMD", "NO3", "SCN", "AZI", "MLI",
             "TAR", "NH4", "IUM", "PEO", "HEZ", "BU3", "PG0", "MXE", "XYL", "SUC", "TRE",
             "LMT", "LMU", "BOG", "BNG", "OLC", "OLA", "OLB", "C8E", "DMU", "LDA", "HTG", "SOG", "CPS", "DDQ", "UMQ", "CXE", "Y01",
             "UNL", "UNX", "UNK"}
SKIP = {"HOH", "WAT", "DOD", "DUM"}
RADIUS = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8, "SE": 1.9, "FE": 1.4, "ZN": 1.39, "MG": 1.18, "CA": 1.7, "MN": 1.4,
          "CU": 1.4, "NI": 1.4, "CO": 1.4, "CD": 1.5, "NA": 1.6, "K": 2.0, "LI": 1.2, "RB": 2.2, "CS": 2.4, "SR": 1.8, "BA": 2.0,
          "CL": 1.75, "BR": 1.85, "I": 1.98, "F": 1.47, "B": 1.8}


def pqr_het(head54, q, r):
    """HETATM line with charge and radius where DelPhiPKa reads them: charge in columns 56-62 (it reads substr(55, 7)), radius from
    column 63.  If the charge starts one column early, a negative charge loses its sign (cardiolipin -2 would be read as +2)."""
    return "%-54s %7.4f%7.4f" % (head54[:54], q, r)


def element(l):
    e = (l[76:78].strip() if len(l) >= 78 else "")
    if not e:
        nm = l[12:16]
        e = nm[:2].strip() if nm[0] != " " and nm[:2].strip().isalpha() and nm[:2].strip().upper() in RADIUS else re.sub(r"[^A-Z]", "", nm.strip())[:1]
    return e.upper()


def classify(pdb, user=(), mode="keep", drop=(), keep_ions=False):
    """-> {resname: (kind, net charge, charge sites)} for every HETATM residue name in the file (DUM / water excluded).
    kind: parent | charged | neutral | ion (left out unless keep_ions) | drop.  user = --cofactor specs (RES or RES:Q), drop = names to leave out."""
    want = {}
    for sp in user:
        name, _, q = sp.partition(":"); want[name.strip().upper()] = float(q) if q else 0.0
    drop = {d.strip().upper() for d in drop if d.strip()}
    names = sorted({l[17:20].strip() for l in open(pdb, errors="replace") if l.startswith("HETATM")} - SKIP)
    plan = {}
    for rn in names:
        if rn in want:
            q = want[rn]; sites = FIXED[rn][1] if rn in FIXED else (() if rn not in IONS else ())
            plan[rn] = ("charged" if q else "neutral", q, sites)
        elif rn in drop: plan[rn] = ("drop", 0.0, ())
        elif mode == "drop": plan[rn] = ("parent", 0.0, ()) if rn == "MSE" else ("drop", 0.0, ())
        elif rn in PARENT: plan[rn] = ("parent", 0.0, ())
        elif rn in FIXED: plan[rn] = ("charged",) + FIXED[rn]
        elif rn in IONS: plan[rn] = ("charged", float(IONS[rn]), ()) if keep_ions else ("ion", float(IONS[rn]), ())
        elif rn in ADDITIVES: plan[rn] = ("drop", 0.0, ())
        else: plan[rn] = ("neutral", 0.0, ())
    return plan


RETINAL = {"RET", "RAL", "RTN"}
WATERS = {"HOH", "WAT", "DOD", "H2O", "TIP"}


def buried_waters(lines, sasa_max=2.0, probe=1.4):
    """Crystal-water oxygens that are buried in the protein: solvent-accessible area (probe 1.4 A) below sasa_max A^2 when only the
    protein (and kept HETATM groups) is present, i.e. waters in internal cavities and channels, not surface waters.
    -> list of PDB lines (HOH O atoms)."""
    import numpy as np
    from scipy.spatial import cKDTree
    wat = [l for l in lines if l.startswith(("ATOM", "HETATM")) and l[17:20].strip() in WATERS and not l[12:16].strip().startswith("H")]
    prot = [l for l in lines if l.startswith(("ATOM", "HETATM")) and l[17:20].strip() not in WATERS | {"DUM"} and element(l) != "H"]
    if not wat or not prot: return []
    X = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in prot]); R = np.array([RADIUS.get(element(l), 1.8) for l in prot])
    T = cKDTree(X); n = 200; k = np.arange(n); z = 1 - 2 * (k + .5) / n; r = np.sqrt(1 - z * z); ph = np.pi * (3 - 5 ** .5) * k
    S = np.stack([r * np.cos(ph), r * np.sin(ph), z], 1); rw = 1.4
    out = []
    for l in wat:
        w = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]); p = w + S * (rw + probe)
        nb = T.query_ball_point(w, rw + probe + 2.0 + probe)
        free = (np.linalg.norm(p[:, None] - X[nb][None], axis=2) >= R[nb] + probe).all(1) if nb else np.ones(n, bool)
        if free.mean() * 4 * np.pi * (rw + probe) ** 2 < sasa_max: out.append(l)
    return out


ACCEPTOR_N = {("HIS", "ND1"), ("HIS", "NE2")}
CARBOXYL_O = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}


def explicit_waters(lines, waters, cut=3.4):
    """Buried crystal waters as explicit TIP3P-charged molecules (O -0.834, H +0.417; O-H 0.9572 A, H-O-H 104.52 deg).  Hydrogens are
    placed to donate H bonds: the first towards the nearest carboxylate oxygen within cut A (else the nearest acceptor: any O, His N),
    the second towards the next acceptor, at the water angle.  -> PQR-format HETATM lines (charge, radius) for the DelPhi input; residue
    name XWT: DelPhiPKa drops HOH and the name fixer moves TIP3 / TP3 / WAT out as solvent."""
    import numpy as np
    acc, cx = [], []
    for l in lines:
        if not l.startswith(("ATOM", "HETATM")) or element(l) == "H" or l[17:20].strip() == "DUM": continue
        rn, an = l[17:20].strip(), l[12:16].strip(); el = element(l)
        if el == "O" or (rn, an) in ACCEPTOR_N:
            acc.append(np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])); cx.append((rn, an) in CARBOXYL_O)
    A = np.array(acc); C = np.array(cx); out = []; theta = np.radians(104.52); b = 0.9572
    for k, l in enumerate(waters):
        o = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        d = np.linalg.norm(A - o, axis=1); ok = (d > 0.5) & (d < cut)
        idx = sorted(np.nonzero(ok)[0], key=lambda i: (not C[i], d[i]))
        u1 = (A[idx[0]] - o) / d[idx[0]] if idx else np.array([0.0, 0.0, 1.0])
        w = None
        for i in idx[1:]:
            v = (A[i] - o) / d[i]; w0 = v - np.dot(v, u1) * u1
            if np.linalg.norm(w0) > 0.2: w = w0 / np.linalg.norm(w0); break
        if w is None:
            t = np.array([1.0, 0.0, 0.0]) if abs(u1[0]) < 0.9 else np.array([0.0, 1.0, 0.0]); w = np.cross(u1, t); w /= np.linalg.norm(w)
        h1 = o + b * u1; h2 = o + b * (np.cos(theta) * u1 + np.sin(theta) * w)
        ch, num = l[21], l[22:26]
        for name, x, q, r in ((" O  ", o, -0.834, 1.52), (" H1 ", h1, 0.417, 1.0), (" H2 ", h2, 0.417, 1.0)):
            out.append(pqr_het("HETATM%5d %4s XWT %s%4s    %8.3f%8.3f%8.3f" % ((len(out) + 1) % 100000, name, ch, num, x[0], x[1], x[2]), q, r))
    return out


def schiff_base_lysines(lines, cut=2.0):
    """Lys residues whose NZ is covalently bound to a separate retinal residue (NZ within cut A of a retinal atom): the protonated
    retinal Schiff base, measured pKa > 12 in bacteriorhodopsin-like pumps.  -> sorted list of (chain, resnum field l[22:27])."""
    import math
    ret = [(float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in lines if l.startswith("HETATM") and l[17:20].strip() in RETINAL]
    if not ret: return []
    out = set()
    for l in lines:
        if l.startswith("ATOM") and l[17:20] == "LYS" and l[12:16].strip() == "NZ":
            x = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
            if min(math.dist(x, r) for r in ret) < cut: out.add((l[21], l[22:27]))
    return sorted(out)


def hold_schiff_base(lines, keys):
    """Rewrite the listed Lys residues as HETATM 'LYR' so that they enter the calculation as a fixed +1 charge on NZ (not titrated)."""
    keys = set(keys); out = []
    for l in lines:
        if l.startswith("ATOM") and l[17:20] == "LYS" and (l[21], l[22:27]) in keys:
            l = "HETATM" + l[6:17] + "LYR" + l[20:]
        out.append(l)
    return out


def map_parents(lines, plan):
    """Rewrite 'parent' residues as ATOM records of the standard residue; atoms the standard residue does not have are left out.
    -> (new lines, {resname: number of residues})."""
    out, n = [], {}
    for l in lines:
        if l.startswith(("ATOM", "HETATM")) and len(l) >= 54 and l[17:20].strip() in PARENT and plan.get(l[17:20].strip(), ("",))[0] == "parent":
            rn = l[17:20].strip(); parent, ren = PARENT[rn]; nm = l[12:16].strip(); nm2 = ren.get(nm, nm)
            if nm2 not in BACKBONE + HEAVY[parent]:
                continue                                   # extra atom (methyl, formyl, hydroxyl ...) or a hydrogen: left out
            el = nm2[0]
            l = "ATOM  " + l[6:12] + (" %-3s" % nm2) + l[16:17] + parent + l[20:]
            if len(l) >= 78: l = l[:76] + ("%2s" % el) + l[78:]
            n.setdefault(rn, set()).add((l[21], l[22:27]))
        out.append(l)
    return out, {k: len(v) for k, v in n.items()}


def write_delphi_input(pdb, outdir, plan):
    """Copy of the structure for DelPhiPKa: ATOM records unchanged; every kept HETATM group (kind charged / neutral) written with its
    per-atom charge and radius in PQR columns (run.prm: 'Remove HETATM : F', 'HETATM in PQR format : T'); dropped groups, waters, dummy
    atoms and hydrogens of HETATM groups left out.  -> (path or None if nothing is kept, {resname: (kind, n residues, net charge)})."""
    lines = open(pdb, errors="replace").read().splitlines()
    res = {}
    for l in lines:
        if l.startswith("HETATM"):
            rn = l[17:20].strip()
            if plan.get(rn, ("drop",))[0] in ("charged", "neutral") and element(l) != "H":
                res.setdefault((rn, l[21], l[22:27]), []).append(l)
    if not res: return None, {}
    q_atom = {}
    for key, ls in res.items():
        kind, q, sites = plan[key[0]]
        if not q: continue
        if sites == "P": idx = [i for i, l in enumerate(ls) if element(l) == "P"]
        else: idx = [i for i, l in enumerate(ls) if l[12:16].strip() in sites]
        if not idx: idx = list(range(len(ls)))                  # sites not found under these names: spread over the heavy atoms
        for i in idx: q_atom[(key, i)] = q / len(idx)
    out, summary = [], {}
    for l in lines:
        if l.startswith("HETATM"):
            rn = l[17:20].strip(); key = (rn, l[21], l[22:27])
            if key not in res or l not in res[key]: continue
            i = res[key].index(l); el = element(l)
            r = RADIUS.get(el, RADIUS.get(el[:1], 1.8)); q = q_atom.get((key, i), 0.0)
            out.append(pqr_het(l[:54], q, r))
        elif l.startswith(("ATOM", "TER", "END", "CRYST1", "MODEL", "ENDMDL")):
            out.append(l)
    for key in res:
        kind, q, _ = plan[key[0]]; s = summary.setdefault(key[0], [kind, 0, q]); s[1] += 1
    path = os.path.join(outdir, os.path.splitext(os.path.basename(pdb))[0] + "_hetero.pdb")
    open(path, "w").write("\n".join(out) + "\n")
    return path, {k: tuple(v) for k, v in summary.items()}


def describe(plan, counts=None, parents=None):
    """One line per kind for the log and the structure check."""
    counts = counts or {}; parents = parents or {}
    by = {}
    for rn, (kind, q, _) in sorted(plan.items()):
        by.setdefault(kind, []).append(rn)
    out = []
    if parents: out.append("read as the standard residue: " + ", ".join("%s x%d" % kv for kv in sorted(parents.items())))
    if by.get("charged"): out.append("kept with a fixed charge (not titrated): " + ", ".join("%s (%+g)%s" % (r, plan[r][1], " x%d" % counts[r][1] if r in counts else "") for r in by["charged"]))
    if by.get("neutral"): out.append("kept as uncharged volume (charge unknown; set it with --cofactor RES:CHARGE): " + ", ".join("%s%s" % (r, " x%d" % counts[r][1] if r in counts else "") for r in by["neutral"]))
    if by.get("ion"): out.append("ions left out (crystal ions are often absent where pKas are measured; keep them with --keep-ions or --cofactor RES:CHARGE): " + ", ".join("%s%s" % (r, " x%d" % counts[r][1] if r in counts else "") for r in by["ion"]))
    if by.get("drop"): out.append("left out (additives / detergents / --drop-het; that space is treated as water or lipid): " + ", ".join(by["drop"]))
    return out
