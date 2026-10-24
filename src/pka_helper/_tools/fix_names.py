#!/usr/bin/env python3
"""
fix_names.py - make a CHARMM-style PDB (CHARMM-GUI, VMD psfgen, OPM+CHARMM-GUI membranes) readable by DelPhiPKa.

DelPhiPKa looks up AMBER/PARSE-style names. Names it does not know get radius 0 and charge 0 (a warning is printed, nothing stops),
so the atom is invisible to the dielectric map and a charged group may silently lose its charge.  This script renames:

    HSP / HIP / HID / HIE   -> HIS            (HSD/HSE are already understood; a doubly protonated His is re-protonated by the program)
    ILE CD                  -> CD1
    OT1 / OT2 (C-terminus)  -> O / OXT        (OXT is dropped by the program, as for any C-terminus)
    TIP3 / TIP / SOL / WAT  -> HOH, oxygen only, moved out of the protein into a separate waters file (written as HETATM, radius 0)
    CHARMM residue names HSD/HSE/HSP, CYSH/CYM, ASPP/GLUP, LSN/LYN ... are mapped to the plain residue name

and reports what is left over that the topology does not contain (cofactors etc.).

    python3 tools/fix_names.py in.pdb out.pdb [--waters waters.pdb] [--topology param/topology]
"""
import argparse, collections, os, re, sys

RES = {"HSP": "HIS", "HIP": "HIS", "HID": "HIS", "HIE": "HIS", "HSD": "HIS", "HSE": "HIS",
       "ASPP": "ASP", "GLUP": "GLU", "LSN": "LYS", "LYN": "LYS", "CYM": "CYS", "CYSH": "CYS", "CYX": "CYS",
       "ASH": "ASP", "GLH": "GLU"}
WATER = {"TIP3", "TIP", "TP3", "SOL", "WAT", "HOH", "H2O"}
WATER_O = {"OH2", "O", "OW", "OH"}


def load_topology_atoms(path):
    """{residue: set(atom names)} from the topology file, if it can be parsed; otherwise {}"""
    res = collections.defaultdict(set)
    if not path or not os.path.isfile(path):
        return res
    for line in open(path):
        p = line.split()
        if len(p) >= 3 and p[0] == "*" and p[1] in ("ATOM",) and len(p) >= 4:
            res[p[2]].add(p[3])
    return res


def fix(lines, topo=None):
    out, waters, stats = [], [], collections.Counter()
    last_res = {}
    # last residue of each chain = C-terminus (to decide what OT1/OT2 mean)
    for l in lines:
        if l.startswith(("ATOM", "HETATM")):
            last_res[l[21]] = (l[22:27])
    for l in lines:
        if not l.startswith(("ATOM", "HETATM")):
            out.append(l)
            continue
        l = l.rstrip("\n").ljust(80)
        rn, an = l[17:21].strip(), l[12:16].strip()
        if rn in WATER:
            if an in WATER_O:
                waters.append("HETATM" + l[6:17] + "HOH " + l[21:].rstrip() + "\n")
            stats["water atoms moved out"] += 1
            continue
        new_rn = RES.get(rn, rn)
        if new_rn != rn:
            stats["residue %s -> %s" % (rn, new_rn)] += 1
        new_an = an
        if new_rn == "ILE" and an == "CD":
            new_an = "CD1"
        elif an == "OT1":
            new_an = "O"
        elif an == "OT2":
            new_an = "OXT"
        if new_an != an:
            stats["atom %s %s -> %s" % (new_rn if new_an.startswith("CD") else "C-term", an, new_an)] += 1
        if new_rn != rn or new_an != an:
            # keep the PDB column layout: 4-char atom field, 3-char residue field
            nm = (" " + new_an).ljust(4) if len(new_an) < 4 else new_an
            l = l[:12] + nm + l[16] + new_rn.rjust(3) + l[20:]
            l = l.rstrip() + "\n"
        else:
            l = l.rstrip() + "\n"
        out.append(l)
    return out, waters, stats


def unknown(lines, topo):
    """(residue, atom) pairs that the topology does not know (hydrogens ignored)."""
    bad = collections.Counter()
    for l in lines:
        if l.startswith("ATOM"):
            rn, an = l[17:20].strip(), l[12:16].strip()
            if an.startswith("H") or an == "OXT":
                continue
            if topo and (rn not in topo or an not in topo[rn]):
                bad[(rn, an)] += 1
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb"); ap.add_argument("out")
    ap.add_argument("--waters", default=None, help="write the crystal/bulk waters (oxygens, HETATM HOH) to this file")
    ap.add_argument("--topology", default="param/topology")
    a = ap.parse_args()
    lines = open(a.pdb).readlines()
    new, waters, stats = fix(lines)
    open(a.out, "w").writelines(new)
    for k, v in sorted(stats.items()):
        print("  %-34s %d" % (k, v))
    if waters:
        wf = a.waters or os.path.splitext(a.out)[0] + "_waters.pdb"
        open(wf, "w").writelines(waters)
        print("  %d waters written to %s (use with --waters)" % (len(waters), wf))
    topo = load_topology_atoms(a.topology)
    bad = unknown(new, topo)
    if bad:
        by = collections.Counter()
        for (r, n), c in bad.items():
            by[r] += c
        print("  still unknown to the topology (radius 0, charge 0): " + ", ".join("%s x%d" % (r, c) for r, c in by.most_common()))


if __name__ == "__main__":
    main()
