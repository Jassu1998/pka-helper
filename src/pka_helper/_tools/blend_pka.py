#!/usr/bin/env python3
"""
blend_pka.py - combine a DelphiPKa membrane pKa table with PROPKA, letting the physics move PROPKA only within a bound.

    python3 tools/blend_pka.py --delphi run/pKa.csv --propka protein.pka [--chain A] [--rule class] [--out blended.csv]
    python3 tools/blend_pka.py ... --labels known.csv --calibrate        # adapt the bound / weight to YOUR protein

  pKa = PROPKA + w * clip(DelPhi - PROPKA)       (DelPhi first limited to 0..16; '*' saturated values are flagged)
  --rule burial (default)  the bound grows with how buried PROPKA says the site is:  bound = 1 + 4*burial  pK  (0-100% -> 1..5 pK);
                           Asp/Glu that DelPhi pushes UP get a small constant cap (--acid-up-cap, default 0.5 pK; <0 = older: bound halved);  w = 1.  (Needs the PROPKA burial column; if absent -> class rule.)
  --rule class             Asp/Glu: DelPhi may raise PROPKA by <= 2 pK, lower it by <= 5;  Lys/His/Arg: +-1;  w = 1
  --rule sym               one bound for all (--bound, --w; default +-3, w 0.75)
  --rule fixed             --cu --cd (acids) --bu --bd (bases) --w, all given by you

Why a bound: DelPhi is better than PROPKA where its physics is right (OmpF, NhaA), but pushes buried groups far too high in
places (HV1: Glu192/196/153 at 13-21), and the error is mostly UPWARD for acids.  Defaults were fitted on OmpF + HV1 (63 CpHMD sites)
and checked on NhaA (3 sites) and 3 blind sites; they are provisional - see CUSTOM_DIELECTRIC.md.

His (all rules, switch off with --no-his-rule): DelPhi is ignored for a His that it puts below pK 0 (saturated; wrong in every such
membrane case: HV1 His99, BM2 His27; --old-his-rule: any gap > 3 pK, which also discarded HV1 His140); otherwise it may lower PROPKA by <= 1.5 and raise it by <= 3.
This stops DelPhi from making His worse; it does NOT recover the high pKa of channel His tetrads (BM2 His27 7.9, M2 His37 8.2).

--pqr protein_noHET.pqr: an Asp/Glu with another carboxylate within 4 A (shared proton, e.g. OmpF Asp312-Glu296) is not halved
when DelPhi pushes it up.

--calibrate: give a CSV with columns  residue,pka   (residue like "GLU 196", "GLU196A" or "E196"; pka may be a bound: "<4", ">8").
It searches w and the four bounds on a coarse grid, keeps the default unless leave-one-site-out error improves by >= 0.10 pK, and
prints the chosen parameters, the leave-one-out error, and PROPKA / DelPhi / default for comparison.  With fewer than 8 labels it
refuses to move away from the default (too easy to fit noise).
"""
import math, os, argparse, csv, itertools, re, sys
import numpy as np

AA = {"ASP": "ASP", "GLU": "GLU", "HIS": "HIS", "LYS": "LYS", "ARG": "ARG", "D": "ASP", "E": "GLU", "H": "HIS", "K": "LYS", "R": "ARG"}
ACID = ("ASP", "GLU")
DEFAULT = dict(w=1.0, c0=1.0, k=4.0, upfrac=0.5, acid_up=0.5)   # burial rule; acid_up: DelPhi may raise an Asp/Glu by <= 0.5 pK (None = burial-scaled x upfrac, the older behaviour)
CLASS = dict(w=1.0, cu=2.0, cd=5.0, bu=1.0, bd=1.0)


def read_delphi(path):
    d = {}
    for l in list(open(path))[1:]:
        t = l.split()
        if len(t) >= 2 and t[0][:3] in AA and re.match(r"^\d+$", t[0][3:7]):
            d[(t[0][:3], int(t[0][3:7]), t[0][7:8] or "A")] = (float(t[1].rstrip("*")), t[1].endswith("*"))
    return d


def read_propka(path):
    """-> ({(res, number, chain): pKa}, {(res, number, chain): burial fraction 0..1}).
    pKa comes from the 'SUMMARY OF THIS PREDICTION' table; burial from the detailed table above it ('ASP  7 A  3.18  0 %  ...')."""
    d, bur, sec = {}, {}, False
    for l in open(path):
        if l.startswith("SUMMARY OF THIS PREDICTION"):
            sec = True; continue
        t = l.split()
        if len(t) >= 4 and t[0] in AA and re.match(r"^\d+$", t[1]):
            key = (t[0], int(t[1]), t[2])
            try:
                if sec: d[key] = float(t[3])
                elif len(t) >= 6 and t[5] == "%": bur.setdefault(key, float(t[4]) / 100.0)
            except ValueError: pass
    return d, bur


def parse_label(s):
    s = s.strip(); b = ""
    if s[:1] in "<>~": b, s = s[0], s[1:]
    return b, float(s)


def parse_res(s):
    s = s.strip().upper().replace("_", " ")
    m = re.match(r"^([A-Z]{1,3})\s*(\d+)\s*([A-Z]?)$", s)
    if not m: raise ValueError("cannot read residue '%s'" % s)
    return AA[m.group(1)], int(m.group(2)), m.group(3)


def blend(rt, o, p, par, bur=None, pair=False, lipid=False, depth_f=1.0):
    s = min(max(o, 0.0), 16.0) - p
    if rt == "HIS" and par.get("his", True):               # His rule: DelPhi puts membrane / channel His far too acidic (HV1 His99/193, BM2 His27, M2 His37)
        if o < 0.0 or (par.get("his_old") and s < -3.0): return p   # DelPhi SATURATED below 0 (a pK it cannot resolve) -> keep PROPKA.  (Older rule: any gap > 3 pK; it also
        down = max(par.get("his_down", 1.5), par.get("his_lip", 0.0) * depth_f)   # threw away HV1 His140, where DelPhi 3.0 vs PROPKA 7.6 was closer to the 5.3 measured.)  Otherwise lower by
        return p + float(np.clip(s, -down, par.get("his_up", 3.0)))          # <= his_down (1.5), or his_lip x depth factor deep in the core (lipid eps 10 rules), raise by <= 3
    if par.get("k") is not None and bur is not None:       # burial rule: bound = c0 + k * PROPKA burial; acids pushed UP: x upfrac
        cap = par["c0"] + par["k"] * bur
        if s > 0 and rt in ACID and not pair and not lipid and par.get("acid_up_w"):   # scaled upward push: DelPhi overshoots in the
            s = s * par["acid_up_w"]                                                     # right direction (ref ~40 % of the way on the development data)
        if s > 0 and rt in ACID and not pair:                       # acids pushed UP by DelPhi: 22 of 31 such pushes were wrong (mean 5.2 pK vs 1.9 pK when right) and no burial
            cap = par["acid_up"] if par.get("acid_up") is not None else cap * par["upfrac"]   # measure (PROPKA, group SASA, cation count) tells them apart -> small constant cap
                                                                    # ... except an acid sharing a proton with a carboxylate <= 4 A away (OmpF Asp312-Glu296), which keeps the full bound
        if s > 0 and lipid and rt in ACID and not pair:                # user-declared lipid-facing acid: no partner to stabilise it, so a large continuum push is plausible (ATP synthase c-ring Glu58)
            cap = max(cap, par.get("lipid_cap", 3.0) * (depth_f if par.get("depth_acids") else 1.0))   # PROPKA burial is not trusted for a lipid-facing group (Glu58 chain P: burial 0, yet the lipid-side pK is high)
        if s < 0 and lipid and rt == "LYS":                            # lipid-facing Lys, no Asp/Glu near: PROPKA knows nothing of the membrane (burial 0 for a side chain in the bilayer
            cap = max(cap, par.get("lipid_cap_base", 4.0) * depth_f)   # scaled by depth in the core (depth_f 0 at the interface, 1 deep inside); core), so the burial bound would clip a large, physical downward push (Lys in a bilayer centre: pK ~5-6 in free-energy simulations)
        if s < 0 and rt == "LYS" and not lipid:                        # Lys not facing the lipid: DelPhi's downward shifts made 28 soluble Lys worse (MAE 0.57 vs PROPKA 0.40; PKAD-R subset,
            cap = min(cap, par.get("lys_down_c0", 0.0) + par["k"] * bur)   # 17 families), so a surface Lys is not moved down; a buried one by up to k * burial
        return p + par["w"] * float(np.clip(s, -cap, cap))
    cu, cd = (par["cu"], par["cd"]) if rt in ACID else (par["bu"], par["bd"])
    return p + par["w"] * (min(s, cu) if s > 0 else max(s, -cd))


def read_pairs(pqr, cut=4.0):
    """(residue type, number, chain) of Asp/Glu whose carboxylate O is within `cut` A of a carboxylate O of ANOTHER residue."""
    at = []
    for l in open(pqr):
        if l.startswith("ATOM") and l[17:20].strip() in ACID and l[12:16].strip() in ("OD1", "OD2", "OE1", "OE2"):
            at.append(((l[17:20].strip(), int(l[22:26]), l[21].strip()), np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
    out = set()
    for i, (ka, xa) in enumerate(at):
        for kb, xb in at[i + 1:]:
            if ka != kb and np.linalg.norm(xa - xb) <= cut: out.add(ka); out.add(kb)
    return out


def err(pred, ref, b):
    return max(0.0, pred - ref) if b == "<" else max(0.0, ref - pred) if b == ">" else abs(pred - ref)


def read_alt(path):
    """Another predictor's pKa table -> {(res, number, chain or ''): pKa}.  Reads KaML-CBtree ('ASP-5,4.07'), pKAI ('A 87 ASP 3.56'),
    and a generic CSV with a residue column ('ASP 87', 'D87A', ...) and a pka column (optional chain column)."""
    d = {}
    for l in open(path):
        l = l.strip()
        if not l or l.lower().startswith(("residue", "#")): continue
        m = re.match(r"^([A-Za-z]{3})-(\d+),\s*(-?[\d.]+)$", l)                      # KaML-CBtree
        if m and m.group(1).upper() in AA: d[(AA[m.group(1).upper()], int(m.group(2)), "")] = float(m.group(3)); continue
        t = l.split()
        if len(t) == 4 and t[2].upper() in AA and t[1].isdigit():                        # pKAI: chain number residue pKa
            try: d[(AA[t[2].upper()], int(t[1]), t[0])] = float(t[3]); continue
            except ValueError: pass
        c = [x.strip() for x in l.split(",")]                                           # generic: residue[,chain],pka
        try:
            rt, n, ch = parse_res(c[0]); d[(rt, n, ch or (c[1] if len(c) == 3 else ""))] = float(c[-1])
        except (ValueError, KeyError, IndexError): pass
    return d


def side(x, ph): return x > ph


# Error model (fitted on the benchmark): reference ~ Normal(pKa, sigma), sigma = exp(b0 + b . x), fitted by maximum likelihood
# on 285 development + soluble sites (experiment and CpHMD); leave-one-family-out: 69 % of references within 1 sigma, 95 % within 2.
ERR_BETA = dict(b0=-0.6007, buried=0.7644, his=0.3091, dis=0.2827, mem=0.5611)


def sigma_of(rt, o, p, bur, membrane):
    """Uncertainty (pK units, 1 sigma) of a blended pKa: larger for buried groups (PROPKA burial >= 0.5), His, a membrane run, and when
    DelPhi (clipped to 0..14) and PROPKA disagree."""
    x = ERR_BETA["b0"] + ERR_BETA["buried"] * ((bur or 0.0) >= 0.5) + ERR_BETA["his"] * (rt == "HIS") + ERR_BETA["mem"] * bool(membrane)
    x += ERR_BETA["dis"] * float(np.log1p(abs(min(max(o, 0.0), 14.0) - p)))
    return float(np.exp(x))


def p_protonated(pka, sigma, ph):
    """Probability that the group carries its proton at this pH (true pKa above the pH)."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf((pka - ph) / (sigma * sqrt(2.0))))


def confidence(blended, propka, alts, ph, margin):
    """'confident' when the blend, PROPKA and every other predictor put the group on the SAME side of `ph`, each at least `margin` pK from it.
    Returns (label, number of predictors compared, why).  Tested on 69 membrane sites: confident calls 96% right (50 sites), uncertain calls 68% (19 sites)."""
    vals = [blended, propka] + list(alts)
    sides = {side(v, ph) for v in vals}
    near = [v for v in vals if abs(v - ph) < margin]
    if len(sides) > 1: return "uncertain", len(vals), "predictors disagree on the side of pH %g" % ph
    if near: return "uncertain", len(vals), "within %g pK of pH %g" % (margin, ph)
    return "confident", len(vals), "all %d predictors agree, >= %g pK from pH %g" % (len(vals), margin, ph)


DEPTH = {}  # (res, number, chain) -> Angstrom inside the membrane core (background map); filled by lipid_auto.core_depth
DEPTH_MIN, DEPTH_FULL, DEPTH_ON = 4.0, 10.0, True
ENV = {}    # (res, number, chain) -> (lipid fraction of the 3-8 A shell, cations within 8 A of an acid / anions within 6 A of a Lys); filled by lipid_auto when it runs
GRID = [(au, c0, kk, lc) for au in (0.25, 0.5, 1.0) for c0 in (0.5, 1.0, 2.0) for kk in (2.0, 4.0, 6.0) for lc in (2.0, 3.0, 4.0)]


def depth_f(k):
    """Fraction of the lipid cap that applies at this depth (1 when depth is unknown or the scaling is off)."""
    if not DEPTH_ON or k not in DEPTH: return 1.0
    return float(min(max((DEPTH[k] - DEPTH_MIN) / max(DEPTH_FULL - DEPTH_MIN, 1e-6), 0.0), 1.0))


def environment(k, o, sat, p, bur, pair, lip):
    """-> (description of the site, which prediction is likely more reliable, why).  A HEURISTIC read-out of what the 69-site benchmark showed; not a validated classifier."""
    rt = k[0]; e = []
    if bur is not None: e.append("PROPKA burial %d%%" % round(100 * bur))
    lf = ENV.get(k)
    if lf is not None:
        e.append("%d%% of the 3-8 A shell is membrane lipid" % round(100 * lf[0]))
        if rt in ACID: e.append("%d Arg/Lys/His within 8 A" % lf[1])
        elif rt in ("LYS", "ARG"): e.append("%d Asp/Glu within 6 A" % lf[1])
    if k in DEPTH: e.append("%.1f A inside the membrane core" % DEPTH[k])
    if pair: e.append("shares a proton with a neighbouring carboxylate")
    if sat: e.append("DelPhi value saturated (limit of its range)")
    env = "; ".join(e) if e else "no environment data"
    push = min(max(o, 0.0), 16.0) - p
    if rt == "HIS":
        return env, "PROPKA", "DelPhi shifted membrane/channel His wrongly in every case of the benchmark (Hv1 His99/193, BM2 His27); the tool limits its shift"
    if rt in ACID and lip:
        return env, "pka-helper", "lipid-facing with no cation partner, so nothing stabilises the charge and a large upward shift is plausible (supported by c-ring Glu58, VDAC1 Glu73, Hv1 Glu196 and bR Asp96; wrong for Hv1 Glu153)"
    if rt == "LYS" and lip:
        return env, "pka-helper", "lipid-facing Lys with no Asp/Glu near: PROPKA has no membrane (its burial is ~0 here) and the continuum penalty for a charge in the bilayer core is large; the cap on DelPhi's downward shift grows with depth in the core (none within 4 A of the interface, where side chains snorkel and the map over-penalises; full beyond 10 A). Free-energy simulations put a Lys at the bilayer centre near pK 5-6; not benchmarked here"
    if rt in ACID and pair:
        return env, "pka-helper", "carboxylate pair sharing a proton (OmpF Glu296/Asp312): PROPKA treats these poorly, DelPhi's coupling is physical"
    if rt in ACID and push > 0.5:
        return env, "PROPKA", "DelPhi pushed this acid up, not lipid-facing: on the benchmark such pushes point the right way in 61 of 85 cases but overshoot, so only 30% of the push is used (at most 0.5 pK)"
    if rt in ACID and push < -1.0:
        return env, "unclear", "DelPhi lowered this acid by %.1f pK; on the benchmark such downward shifts improved 24 acids and worsened 8 (mostly acids whose measured pKa is raised by effects the continuum model misses, e.g. the catalytic Glu35 of lysozyme)" % -push
    if rt in ("LYS", "ARG", "TYR", "CYS"):
        return env, "unclear", "too few %s sites in the benchmark to say" % rt
    return env, "unclear", "no rule applies; the two predictions are close or the benchmark has no clear case"


def uncertainty(k, o, b, p, bur, pair, lip, par, ph, dfac=1.0):
    """-> [param_lo, param_hi, param_flips_call]: range of the blended pKa when the blend's own settings are varied over a grid (no fitted error model)."""
    vals = []
    if par.get("k") is not None and bur is not None:                          # sensitivity of the blend to its own settings (burial rule only)
        for au, c0, kk, lc in GRID:
            q = dict(par, acid_up=au, c0=c0, k=kk, lipid_cap=lc, lipid_cap_base=lc + 1.0); vals.append(blend(k[0], o, p, q, bur, pair, lip, dfac))
    lo, hi = (min(vals), max(vals)) if vals else (b, b)
    return ["%.2f" % lo, "%.2f" % hi, int((lo > ph) != (hi > ph))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delphi", required=True); ap.add_argument("--propka", required=True)
    ap.add_argument("--chain", default="", help="only this chain (default: all chains present in both files)")
    ap.add_argument("--rule", default="burial", choices=["burial", "class", "sym", "fixed"])
    ap.add_argument("--bound", type=float, default=3.0); ap.add_argument("--w", type=float, default=None)
    ap.add_argument("--cu", type=float, default=2.0); ap.add_argument("--cd", type=float, default=5.0)
    ap.add_argument("--bu", type=float, default=1.0); ap.add_argument("--bd", type=float, default=1.0)
    ap.add_argument("--no-his-rule", action="store_true", help="treat His like the other residues")
    ap.add_argument("--old-his-rule", action="store_true", help="ignore DelPhi for a His whenever it is > 3 pK below PROPKA (older rule), not only when it is saturated")
    ap.add_argument("--acid-up-cap", type=float, default=0.5, help="largest upward shift DelPhi may give an Asp/Glu (default 0.5 pK); a negative value restores the older burial-scaled bound (1+4*burial, halved)")
    ap.add_argument("--pqr", help="protein_noHET.pqr of the DelPhi run: Asp/Glu with a carboxylate partner within 4 A keep the full upward bound (shared proton)")
    ap.add_argument("--labels"); ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--alt", action="append", default=[], help="another predictor's pKa table (KaML-CBtree csv, pKAI output, or 'residue,pka' csv); repeat for several. Used only for the confidence flag.")
    ap.add_argument("--fixed-charged", default="", help="RES:NUM:CHAIN,... groups held charged in the DelPhi calculation (e.g. a retinal Schiff-base Lys): listed as fixed, no pKa")
    ap.add_argument("--propka-only", default="", help="residue types (e.g. LYS,ARG) that were not titrated by DelPhiPKa (run with 'Skip Lys and Arg'): listed with their PROPKA pKa, no blend")
    ap.add_argument("--ph", type=float, default=7.0, help="pH for the protonation call and confidence flag (default 7)")
    ap.add_argument("--conf-margin", type=float, default=0.5, help="a call is confident only if every predictor is at least this many pK from --ph (default 0.5)")
    ap.add_argument("--lipid-acids", default="", help="comma list of Asp/Glu you know face the lipid (e.g. GLU58 or GLU58:K,GLU58:L); these are exempt from the 0.5 pK upward cap (see --lipid-cap).  Use this to name them yourself; by default they are found automatically (--lipid-auto)")
    ap.add_argument("--lipid-auto", default="", metavar="RUN_DIR", help="folder with the membrane background map (background.dx + protein_noHET.pqr).  DEFAULT: the folder that holds the --delphi table, if it has them.  Finds the lipid-facing acids automatically; same exemption as --lipid-acids. See tools/lipid_auto.py for the rule and its limits")
    ap.add_argument("--lipid-cap-base", type=float, default=4.0, help="largest DOWNWARD shift (pK) for a lipid-facing Lys (no Asp/Glu within 6 A), ignoring the burial bound (default 4.0)")
    ap.add_argument("--lipid-depth-min", type=float, default=4.0, help="a lipid-facing Lys whose side-chain N lies less than this many A inside the membrane core gets no extra downward cap (default 4.0): the continuum map over-penalises a charge at the interface, where real side chains snorkel")
    ap.add_argument("--lipid-depth-full", type=float, default=10.0, help="depth (A inside the core) from which the full --lipid-cap-base applies; the cap grows linearly between --lipid-depth-min and this (default 10.0)")
    ap.add_argument("--no-lipid-depth", action="store_true", help="switch the depth scaling off (the lipid cap is the same at every depth)")
    ap.add_argument("--lipid-depth-acids", action="store_true", help="also scale the upward cap of lipid-facing Asp/Glu by depth (off by default: the evidence for the interface over-penalty is for Lys; c-ring Glu58 and 2XTV Asp243 argue for keeping the acid push)")
    ap.add_argument("--lys-down-c0", type=float, default=0.0, help="constant part of the largest DOWNWARD shift for a Lys that is not lipid-facing (default 0: only burial * k, so a surface Lys keeps PROPKA's value; 1.0 also lowers surface Lys by up to 1 pK)")
    ap.add_argument("--no-lipid-auto", action="store_true", help="switch the automatic lipid-facing exemption off (acids pushed up are capped at 0.5 pK everywhere)")
    ap.add_argument("--lipid-cap", type=float, default=3.0, help="largest upward shift (pK) for a --lipid-acids residue, ignoring the burial bound (default 3.0)")
    ap.add_argument("--his-down", type=float, default=1.5, help="largest DOWNWARD shift DelPhi may give a His (default 1.5; 0 with the lipid eps 10 rules)")
    ap.add_argument("--his-lip-cap", type=float, default=0.0, help="largest downward shift for a His deep in the membrane core, scaled by depth like --lipid-cap-base (default 0 = off; 2 with the lipid eps 10 rules)")
    ap.add_argument("--acid-up-scale", type=float, default=0.0, help="scale DelPhi's upward push on an Asp/Glu by this factor before the upward cap (default 0 = off; "
                    "pka-helper uses 0.3: the pushes mostly point the right way but overshoot)")
    ap.add_argument("--hbond-shift", type=float, default=0.0, help="neutral-state H-bond term: raise the DelPhi pKa of a buried Asp/Glu by this many pK per acceptor "
                    "only the protonated acid can use (backbone/Asn/Gln carbonyl O, Ser/Thr/Tyr hydroxyl; at most 2), before the blend (default 0 = off; needs --pqr)")
    ap.add_argument("--lipid-cat-cut", type=float, default=8.0, help="an acid counts as lipid-facing only if no Arg/Lys/His N lies within this many A (default 8)")
    ap.add_argument("--soluble-run", action="store_true", help="the DelPhi run had no membrane (--no-membrane): used only for the error model")
    ap.add_argument("--unpartnered-acids", default="", help="comma list (e.g. GLU325A) of buried Asp/Glu whose only salt-bridge partner came out neutral "
                    "(pka-helper's partner check): noted with the value the lipid-facing upward cap would give")
    ap.add_argument("--uncap-unpartnered", action="store_true", help="give the --unpartnered-acids the lipid-facing upward cap (up to --lipid-cap) "
                    "instead of the 0.5 pK cap (experimental: one measured case, LacY Glu325)")
    ap.add_argument("--out", default="blended.csv")
    a = ap.parse_args()
    par = dict(DEFAULT) if a.rule == "burial" else dict(CLASS)
    if a.rule == "sym": par = dict(w=0.75 if a.w is None else a.w, cu=a.bound, cd=a.bound, bu=a.bound, bd=a.bound)
    elif a.rule == "fixed": par = dict(w=1.0 if a.w is None else a.w, cu=a.cu, cd=a.cd, bu=a.bu, bd=a.bd)
    elif a.w is not None: par["w"] = a.w
    if a.rule == "burial": par["acid_up"] = None if a.acid_up_cap < 0 else a.acid_up_cap
    par["lipid_cap"] = a.lipid_cap; par["lipid_cap_base"] = a.lipid_cap_base; par["depth_acids"] = a.lipid_depth_acids; par["lys_down_c0"] = a.lys_down_c0; par["his_down"] = a.his_down; par["his_lip"] = a.his_lip_cap; par["acid_up_w"] = a.acid_up_scale or None
    global DEPTH_MIN, DEPTH_FULL, DEPTH_ON
    DEPTH_MIN, DEPTH_FULL, DEPTH_ON = a.lipid_depth_min, a.lipid_depth_full, not a.no_lipid_depth
    LIP = set()
    for t in [x.strip().upper() for x in a.lipid_acids.split(",") if x.strip()]:
        rt, n, ch = parse_res(t.split(":")[0] + (t.split(":")[1] if ":" in t else "")); LIP.add((rt, n, ch))
    ld = a.lipid_auto or os.path.dirname(os.path.abspath(a.delphi))
    if not a.no_lipid_auto and a.rule == "burial" and all(os.path.exists(os.path.join(ld, f)) for f in ("background.dx", "protein_noHET.pqr")):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from lipid_auto import lipid_facing, lipid_facing_bases, core_depth
        try: DEPTH.update(core_depth(ld))
        except Exception as ex: print("lipid-auto: could not measure the depth in the membrane core (%s); the lipid cap is not scaled" % ex)
        auto = lipid_facing(ld, cat_cut=a.lipid_cat_cut, report=ENV); LIP |= set(auto)
        print("lipid-auto: %d lipid-facing acids found: %s" % (len(auto), " ".join("%s%d%s" % (k[0], k[1], k[2]) for k in sorted(auto, key=lambda k: (k[2], k[1])))))
        autob = lipid_facing_bases(ld, report=ENV); LIP |= set(autob)
        print("lipid-auto: %d lipid-facing lysines found: %s" % (len(autob), " ".join("%s%d%s" % (k[0], k[1], k[2]) for k in sorted(autob, key=lambda k: (k[2], k[1])))))
    if a.no_his_rule: par["his"] = False
    if a.old_his_rule: par["his_old"] = True
    D = read_delphi(a.delphi); P, BUR = read_propka(a.propka)
    PAIR = read_pairs(a.pqr) if a.pqr else set()
    HB = {}
    if a.hbond_shift > 0 and a.pqr:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from hbond_term import acceptor_counts, shift_for
        HB = acceptor_counts(a.pqr)
    keys = sorted((k for k in D if k in P and (not a.chain or k[2] == a.chain)), key=lambda k: (k[2], k[1]))
    if not keys: sys.exit("no residues in common: check --chain and that both files list the same numbering")
    note = ""
    if a.calibrate:
        if not a.labels: sys.exit("--calibrate needs --labels")
        lab = []
        for r in csv.DictReader(open(a.labels)):
            rt, n, ch = parse_res(r.get("residue") or r.get("Residue")); b, v = parse_label(r.get("pka") or r.get("pKa"))
            ks = [k for k in keys if k[0] == rt and k[1] == n and (not ch or k[2] == ch)]
            lab += [(k, b, v) for k in ks]
        print("calibration: %d labelled sites matched" % len(lab))
        if len(lab) < 8:
            print("  fewer than 8 labelled sites: keeping the default parameters (a fit to so few points would mostly be noise)")
        else:
            Cs = [1, 2, 3, 5, 8]
            grid = [dict(w=w, c0=c0, k=k, upfrac=uf) for w in (0.75, 1.0) for c0 in (0.5, 1, 2) for k in (0, 2, 4, 6, 8) for uf in (0.5, 1.0)]
            def mae(g, S): return float(np.mean([err(blend(k[0], D[k][0], P[k], g, BUR.get(k), k in PAIR), v, b) for k, b, v in S]))
            loo = []
            for i in range(len(lab)):
                tr = lab[:i] + lab[i + 1:]; best = min(grid, key=lambda g: mae(g, tr) + 0.02 * (abs(g["c0"] - 1) + abs(g["k"] - 4) / 2))
                if mae(best, tr) > mae(DEFAULT, tr) - 0.10: best = DEFAULT
                loo.append(err(blend(lab[i][0][0], D[lab[i][0]][0], P[lab[i][0]], best, BUR.get(lab[i][0]), lab[i][0] in PAIR), lab[i][2], lab[i][1]))
            full = min(grid, key=lambda g: mae(g, lab) + 0.02 * (abs(g["c0"] - 1) + abs(g["k"] - 4) / 2))
            chosen = full if mae(full, lab) <= mae(DEFAULT, lab) - 0.10 else DEFAULT
            print("  PROPKA MAE %.2f | DelPhi MAE %.2f | default rule MAE %.2f | calibrated (in-sample) %.2f | calibrated leave-one-out %.2f" % (
                np.mean([err(P[k], v, b) for k, b, v in lab]), np.mean([err(min(max(D[k][0], 0), 16), v, b) for k, b, v in lab]),
                mae(DEFAULT, lab), mae(chosen, lab), np.mean(loo)))
            print("  chosen:", chosen, "(default kept)" if chosen is DEFAULT else "")
            if np.mean(loo) > mae(DEFAULT, lab): print("  warning: leave-one-out error is worse than the default; use the default")
            par = chosen; note = "calibrated"
    ALT = [read_alt(p) for p in a.alt]
    UNP = set()
    for x in [x.strip() for x in a.unpartnered_acids.split(",") if x.strip()]:
        m = re.match(r"^([A-Z]{3})(\d+)(\S*)$", x)
        if m: UNP.add((m.group(1), int(m.group(2)), m.group(3)))
    ncon = 0
    with open(a.out, "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["residue", "chain", "PROPKA", "DelPhiPKa", "saturated", "blended", "shift_applied", "note",
                                         "state_at_pH%g" % a.ph, "confidence", "n_predictors", "confidence_reason"] + ["alt%d_%s" % (i + 1, re.sub(r"\W", "", p.rsplit("/", 1)[-1])[:12]) for i, p in enumerate(a.alt)]
                       + ["param_lo", "param_hi", "param_flips_call", "environment", "likely_more_reliable", "reliability_reason", "sigma", "P_protonated"])
        for k in keys:
            o, sat = D[k]; lipk = (k[0], k[1], k[2]) in LIP or (k[0], k[1], "") in LIP; dfk = depth_f((k[0], k[1], k[2]))
            hb = shift_for(k, HB, BUR.get(k), a.hbond_shift) if HB else 0.0
            o_raw = o; o = o + hb                                   # optional neutral-state H-bond term (before the blend; the DelPhi column stays raw)
            unp = k in UNP and k[0] in ACID and not lipk
            unp_note = ""; lip0 = lipk
            if unp:
                b_open = blend(k[0], o, P[k], par, BUR.get(k), k in PAIR, True, dfk)
                if a.uncap_unpartnered: lipk = True; unp_note = "salt-bridge partner neutral: upward cap lifted; "
                elif b_open > blend(k[0], o, P[k], par, BUR.get(k), k in PAIR, lipk, dfk) + 0.05:
                    unp_note = "salt-bridge partner neutral: may be higher, up to %.2f if nothing else stabilises it (--lipid-acids applies that); " % b_open
                else: unp_note = "salt-bridge partner neutral; "
            b = blend(k[0], o, P[k], par, BUR.get(k), k in PAIR, lipk, dfk)
            sg = sigma_of(k[0], o_raw, P[k], BUR.get(k), not a.soluble_run)
            av = [A.get(k, A.get((k[0], k[1], ""))) for A in ALT]
            conf, npred, why = confidence(b, P[k], [x for x in av if x is not None], a.ph, a.conf_margin)
            if not ALT: why += " (only blend and PROPKA compared: give --alt tables for the full test)"
            ncon += conf == "confident"
            state = "protonated" if b > a.ph else "deprotonated"     # pKa above pH: the group carries its proton (neutral Asp/Glu/Tyr/Cys, charged Lys/Arg/His)
            w.writerow(["%s%d" % (k[0], k[1]), k[2], "%.2f" % P[k], "%.2f%s" % (o_raw, "*" if sat else ""), int(sat), "%.2f" % b, "%+.2f" % (b - P[k]), unp_note + ("neutral-state H bond: DelPhi +%.1f; " % hb if hb else "") + ("His: DelPhi saturated below 0, PROPKA kept" if (k[0] == "HIS" and par.get("his", True) and o < 0.0) else "carboxylate pair: full upward bound" if k in PAIR and k[0] in ACID else ("lipid-facing: downward cap %s" % ("lifted" if dfk >= 0.999 else "scaled to %d%% (%.1f A inside the core)" % (round(100 * dfk), DEPTH.get((k[0], k[1], k[2]), 0.0))) if k[0] == "LYS" else "lipid-facing: upward cap lifted") if not (unp and a.uncap_unpartnered) and ((k[0], k[1], k[2]) in LIP or (k[0], k[1], "") in LIP) else ""),
                        state, conf, npred, why] + ["" if x is None else "%.2f" % x for x in av] + uncertainty(k, o, b, P[k], BUR.get(k), k in PAIR, lipk, par, a.ph, dfk) + list(environment(k, o, sat, P[k], BUR.get(k), k in PAIR, lip0))
                       + ["%.2f" % sg, "%.2f" % p_protonated(b, sg, a.ph)])
        extra = sorted((q for q in P if q[0] in [x.strip().upper() for x in a.propka_only.split(",") if x.strip()] and q not in D and (not a.chain or q[2] == a.chain)), key=lambda q: (q[2], q[1])) if a.propka_only else []
        for q in extra:
            b = P[q]; av = [A.get(q, A.get((q[0], q[1], ""))) for A in ALT]
            conf, npred, why = confidence(b, P[q], [x for x in av if x is not None], a.ph, a.conf_margin)
            ncon += conf == "confident"
            w.writerow(["%s%d" % (q[0], q[1]), q[2], "%.2f" % P[q], "", 0, "%.2f" % b, "+0.00", "PROPKA only (not titrated by DelPhiPKa)",
                        "protonated" if b > a.ph else "deprotonated", conf, npred, why] + ["" if x is None else "%.2f" % x for x in av] + [""] * 8)
        keys = keys + extra
        for item in [x for x in a.fixed_charged.split(",") if x]:
            rn, num, ch = item.split(":")
            q = (rn, int(num), ch)
            if a.chain and ch != a.chain: continue
            ncon += 1
            w.writerow(["%s%d" % (rn, int(num)), ch, "%.2f" % P[q] if q in P else "", "fixed", 0, "14.00", "", "retinal Schiff base: held charged (+1), not titrated; measured pKa > 12",
                         "protonated", "confident", 0, "fixed charge (protonated retinal Schiff base)"] + [""] * (len(ALT) + 8))
            keys = keys + [q]
    print("%d residues -> %s   parameters: %s %s" % (len(keys), a.out, par, note))
    print("pH %g call: %d confident, %d uncertain%s" % (a.ph, ncon, len(keys) - ncon, "" if ALT else "   (no --alt tables: weaker test, see CUSTOM_DIELECTRIC.md)"))


if __name__ == "__main__":
    main()
