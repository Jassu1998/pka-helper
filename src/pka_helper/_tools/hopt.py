#!/usr/bin/env python3
"""
hopt.py - choose which carboxylate oxygen carries the proton of the NEUTRAL Asp/Glu state with PDB2PQR's hydrogen-bond optimiser.

DelPhiPKa builds the neutral state of every Asp/Glu with the proton on one fixed oxygen (run.prm: 'Hydrogen of ASP Attached to Atom : OD1',
'Hydrogen of GLU Attached to Atom : OE1').  In a protein the two oxygens are not equivalent: the proton should sit on the one that can
donate a hydrogen bond (to a backbone C=O, another carboxylate, Ser/Thr/Tyr OH, ...) and not point into a clash.

What this does
  1. All Asp/Glu are renamed ASH/GLH (protonated) in a scratch copy, which is run through `pdb2pqr30 --ff AMBER` (default hydrogen-bond
     optimisation and debumping; its 'Carboxylic' optimiser tries the four syn/anti positions on both oxygens and keeps the best H-bond
     network).
  2. For every residue it reads which oxygen ended up carrying the proton (PDB2PQR always calls it OD2/OE2, swapping the heavy-atom labels
     when needed, so the choice is read from the coordinates).
  3. In the PDB that DelPhiPKa will read, the coordinates of OD1/OD2 (OE1/OE2) are swapped where necessary so that the proton-bearing oxygen
     carries the label DelPhiPKa protonates (OD1 / OE1).  The charged state is symmetric in the force field, so nothing else changes.

Limits: every acid is neutral during the optimisation (the optimiser sees an all-neutral carboxylate network); only the oxygen choice is
taken, not the exact H position (DelPhiPKa uses its own geometry); Asn/Gln/His flips and Ser/Thr/Tyr rotations are not transferred.
Usage:  python3 hopt.py in.pdb out.pdb [--atom-d OD1] [--atom-e OE1] [--pdb2pqr pdb2pqr30]
"""
import argparse, os, subprocess, sys, tempfile, shutil
import numpy as np

PAIRS = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
HPAIR = {"ASP": ("ASH", "OD2"), "GLU": ("GLH", "OE2")}

def key(line):
    return (line[21], line[22:26].strip(), line[26].strip())

def xyz(line):
    return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])

def optimise(lines, pdb2pqr="pdb2pqr30", workdir=None):
    """returns {(chain,resseq,icode): xyz of the oxygen that carries the proton}, and the log text"""
    tmp = workdir or tempfile.mkdtemp(prefix="hopt_")
    neu, pqr = os.path.join(tmp, "neutral.pdb"), os.path.join(tmp, "neutral.pqr")
    with open(neu, "w") as fh:
        for l in lines:
            if l.startswith("ATOM") and l[17:20] in PAIRS:
                l = l[:17] + HPAIR[l[17:20]][0] + l[20:]
            if l.startswith("ATOM") or l.startswith("TER"):
                fh.write(l)
        fh.write("END\n")
    r = subprocess.run([pdb2pqr, "--ff", "AMBER", "--drop-water", neu, pqr], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(pqr):
        raise RuntimeError("pdb2pqr failed:\n" + r.stdout[-1500:] + r.stderr[-1500:])
    # PDB2PQR drops the chain ID from the PQR it writes, but keeps the input order: return the proton-bearing oxygens as an ordered list
    res = []
    for l in open(pqr):
        if l.startswith("ATOM"):
            t = l.split()
            if len(t) >= 10 and t[3] in ("ASH", "GLH") and t[2] in ("OD2", "OE2"):
                res.append(("ASP" if t[3] == "ASH" else "GLU", t[-6].rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                            np.array([float(t[-5]), float(t[-4]), float(t[-3])])))
    return res, r.stdout + r.stderr

def apply(lines, carriers, atom_d="OD1", atom_e="OE1"):
    """swap O coordinates so that the proton-bearing oxygen has the label DelPhiPKa protonates; returns new lines and stats"""
    want = {"ASP": atom_d, "GLU": atom_e}
    idx = {}
    for i, l in enumerate(lines):
        if l.startswith("ATOM") and l[17:20] in PAIRS and l[12:16].strip() in PAIRS[l[17:20]]:
            idx.setdefault(key(l), {})[l[12:16].strip()] = i
    out = list(lines); st = {"residues": 0, "swapped": 0, "unmatched": 0}
    swapped = []
    # PDB2PQR drops chain IDs and may reorder chains, so match each residue to the PQR residue of the same name and number that lies nearest
    for k, d in sorted(idx.items(), key=lambda kv: min(kv[1].values())):
        o1, o2 = PAIRS[lines[next(iter(d.values()))][17:20]]
        if o1 not in d or o2 not in d:
            continue
        resn = lines[d[o1]][17:20]
        st["residues"] += 1
        p1, p2 = xyz(lines[d[o1]]), xyz(lines[d[o2]])
        cand = [(np.linalg.norm((p1 + p2) / 2 - c), c) for rn, rs, c in carriers if rn == resn and rs == k[1]]
        if not cand or min(cand)[0] > 3.0:
            st["unmatched"] += 1; continue
        c = min(cand, key=lambda t: t[0])[1]
        d1, d2 = np.linalg.norm(p1 - c), np.linalg.norm(p2 - c)      # debumping can move the atoms a little, so compare, do not threshold
        carrier = o1 if d1 < d2 else o2
        if carrier != want[resn]:
            i1, i2 = d[o1], d[o2]
            a_, b_ = lines[i1], lines[i2]
            out[i1] = a_[:30] + b_[30:54] + a_[54:]
            out[i2] = b_[:30] + a_[30:54] + b_[54:]
            st["swapped"] += 1; swapped.append("%s%s%s" % (resn, k[1], k[0]))
    return out, st, swapped

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb_in"); ap.add_argument("pdb_out")
    ap.add_argument("--atom-d", default="OD1"); ap.add_argument("--atom-e", default="OE1")
    ap.add_argument("--pdb2pqr", default="pdb2pqr30")
    a = ap.parse_args()
    lines = open(a.pdb_in).readlines()
    carriers, log = optimise(lines, a.pdb2pqr)
    new, st, sw = apply(lines, carriers, a.atom_d, a.atom_e)
    open(a.pdb_out, "w").writelines(new)
    print("hopt: %(residues)d Asp/Glu, %(swapped)d oxygen labels swapped, %(unmatched)d unmatched" % st)
    if sw: print("swapped: " + " ".join(sw))

if __name__ == "__main__":
    main()
