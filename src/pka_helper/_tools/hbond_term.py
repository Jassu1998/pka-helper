"""hbond_term.py - optional neutral-state H-bond term for buried Asp/Glu (pka-helper --hbond-term).

A carboxyl group can only H-bond to an acceptor that cannot donate back (a backbone carbonyl O, an Asn/Gln side-chain carbonyl O) - or to
a hydroxyl that accepts (Ser/Thr/Tyr) - when it carries its proton.  The continuum calculation has no term for such H bonds, so it
under-stabilises the neutral acid (OmpF Asp256: CpHMD > 8, DelPhi 3.8; bacteriorhodopsin Asp96 next to Thr46).  With the term on, the
DelPhi pKa of a buried acid (PROPKA burial >= 50 %) is raised by SHIFT pK per such acceptor within 3.1 A of a carboxylate O (at most 2),
before the blend.  SHIFT 1.5 pK (about 2 kcal/mol) was chosen on the development data; it changes only a few sites there (see README).

    from hbond_term import acceptor_counts
    n = acceptor_counts("protein_noHET.pqr")     # {(resname, number, chain): count}
"""
import numpy as np

CARB = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
ACCEPT = {("ASN", "OD1"), ("GLN", "OE1"), ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH")}


def acceptor_counts(pqr, cut=3.1):
    from scipy.spatial import cKDTree
    at = [(l[17:20], l[21], int(l[22:26]), l[12:16].strip(), (float(l[30:38]), float(l[38:46]), float(l[46:54])))
          for l in open(pqr) if l.startswith("ATOM") and not l[12:16].strip().startswith("H")]
    if not at: return {}
    X = np.array([a[4] for a in at]); T = cKDTree(X); out = {}
    for i, a in enumerate(at):
        if a[0] in CARB and a[3] in CARB[a[0]]:
            k = (a[0], a[2], a[1]); s = out.setdefault(k, set())
            for j in T.query_ball_point(X[i], cut):
                b = at[j]
                if b[1] == a[1] and b[2] == a[2]: continue
                if b[3] == "O" or (b[0], b[3]) in ACCEPT: s.add((b[1], b[2], b[3]))
    return {k: len(v) for k, v in out.items()}


def shift_for(k, counts, burial, shift, nmax=2, bur_min=0.5):
    if shift <= 0 or k[0] not in CARB or (burial if burial is not None else 0.0) < bur_min: return 0.0
    return shift * min(counts.get(k, 0), nmax)
