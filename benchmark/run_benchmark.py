#!/usr/bin/env python3
"""Reproduce the pka-helper benchmark numbers end to end: run `pka-helper run` on every structure, then score its results/pka_helper.csv.

    python3 benchmark/run_benchmark.py                       # all three suites, default settings
    python3 benchmark/run_benchmark.py --suite membrane      # membrane | peptides | soluble | all
    python3 benchmark/run_benchmark.py --tag eps10 -- --lipid-eps 10     # anything after -- goes to every `pka-helper run`

Runs go to benchmark/runs/<tag>/<suite>/<name>/ and are reused when results/pka_helper.csv exists (delete a folder to redo it).
The report is printed and written to benchmark/runs/<tag>/report.md.

Suites
  membrane  91 sites, 7 structures (sites_membrane.csv). "69" = all but HV1 5OQK (the same protein as the AlphaFold model).
            References: constant-pH MD (OmpF, HV1, NhaA) and experiment (PR Asp97, EmrE Glu14, BM2 His27).
            The blend rules were developed on these sites: treat them as a development set, not an independent test.
  peptides  15 references, single titratable groups in a TM helix (GWALP23 host); experiment + constant-pH MD. Used to calibrate
            --lipid-eps 10, so also a development set.
  soluble   201 sites, 22 PKAD-R structures (Main, wild type, no warning), run with --no-membrane. Not used to tune anything.
  locked    experimental membrane pKas NOT used to tune any default (bacteriorhodopsin, LacY, M2 His37 stepwise). Report only:
            never change a rule because of it, or it stops being a test.
Scoring: mean over the listed chains; DelPhi values clipped to 0..14; a reference '<x' / '>x' only counts a prediction on the wrong side.
"""
import argparse, csv, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = (("PROPKA", "PROPKA"), ("DelPhi", "DelPhiPKa"), ("pka-helper", "blended"))


def err(v, ref, b):
    if b == "<": return max(0.0, v - ref)
    if b == ">": return max(0.0, ref - v)
    return abs(v - ref)


def pka_helper_cmd():
    exe = os.environ.get("PKA_HELPER")
    if exe: return exe.split()
    src = os.path.join(os.path.dirname(HERE), "src")
    return [sys.executable, "-c", "import sys; sys.path.insert(0, %r); from pka_helper.cli import main; sys.exit(main())" % src] \
        if os.environ.get("PKA_HELPER_FROM_SOURCE") else ["pka-helper"]


def run_one(pdb, out, opts, extra, np_):
    if os.path.isfile(os.path.join(out, "results", "pka_helper.csv")): return "reused"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = pka_helper_cmd() + ["run", pdb, "--out", out, "--show", "none"] + (["--np", str(np_)] if np_ else []) + opts + extra
    t = time.time()
    with open(out + ".log", "w") as lg:
        rc = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT).returncode
    return "exit %d, %.0f s" % (rc, time.time() - t)


def read_result(out):
    d = {}
    for r in csv.DictReader(open(os.path.join(out, "results", "pka_helper.csv"))):
        name = r["residue"]; rt, num = name[:3], int(name[3:])
        try: d[(rt, num, r["chain"])] = {k: float(r[c].rstrip("*")) for k, c in COLS}
        except ValueError: continue          # a group held at a fixed charge (e.g. retinal Schiff base): no pKa
    return d


def value(res, rt, num, chains, col):
    v = [res[(rt, num, c)][col] for c in chains if (rt, num, c) in res]
    if not v: return None
    v = float(np.mean(v))
    return min(max(v, 0.0), 14.0) if col == "DelPhi" else v


def summarize(rows, label, groupkey=None, boot_key=None, seed=0):
    """rows: list of dicts with ref, bound, and one value per column. -> markdown lines."""
    out = []
    def stats(R, col):
        e = np.array([err(r[col], r["ref"], r["bound"]) for r in R]); c = sum((r[col] > 7) == (r["ref"] > 7) for r in R)
        return e.mean(), c, len(R)
    out.append("| %s | n | %s |" % (label, " | ".join("%s MAE / right at pH 7" % k for k, _ in COLS)))
    out.append("|---|---|" + "---|" * len(COLS))
    groups = [("all", rows)]
    if groupkey: groups += [(g, [r for r in rows if r[groupkey] == g]) for g in dict.fromkeys(r[groupkey] for r in rows)]
    for g, R in groups:
        if not R: continue
        out.append("| %s | %d | %s |" % (g, len(R), " | ".join("%.2f / %d" % stats(R, k)[:2] for k, _ in COLS)))
    if boot_key:
        rng = np.random.default_rng(seed); F = {}
        for i, r in enumerate(rows): F.setdefault(r[boot_key], []).append(i)
        F = list(F.values())
        d = np.array([err(r["pka-helper"], r["ref"], r["bound"]) - err(r["PROPKA"], r["ref"], r["bound"]) for r in rows])
        bs = [d[[i for j in rng.integers(0, len(F), len(F)) for i in F[j]]].mean() for _ in range(5000)]
        out.append("")
        out.append("pka-helper - PROPKA: %+.3f pK (95%% bootstrap over %d %s: %+.3f .. %+.3f); better on %d sites, worse on %d (|diff| > 0.25)"
                   % (d.mean(), len(F), boot_key, np.percentile(bs, 2.5), np.percentile(bs, 97.5), int((d < -0.25).sum()), int((d > 0.25).sum())))
    return out


def suite_membrane(a, base):
    sites = list(csv.DictReader(open(os.path.join(HERE, "sites_membrane.csv"))))
    for f in dict.fromkeys(s["structure"] for s in sites):
        out = os.path.join(base, "membrane", os.path.splitext(f)[0])
        print("  membrane %-18s %s" % (f, run_one(os.path.join(HERE, "structures", "membrane", f), out, [], a.extra, a.np)), flush=True)
    rows = []
    for s in sites:
        res = read_result(os.path.join(base, "membrane", os.path.splitext(s["structure"])[0]))
        r = dict(group=s["group"], protein=s["protein"], ref=float(s["ref"]), bound=s["bound"], in69=s["in69"], site=s["site"],
                 reftype="CpHMD" if s["source"] == "CpHMD" else "experiment")
        for k, _ in COLS: r[k] = value(res, s["resname"], int(s["resnum"]), s["chains"], k)
        if any(r[k] is None for k, _ in COLS): print("  missing:", s["site"], s["structure"]); continue
        rows.append(r)
    for r in rows: r["protein_fold"] = "hv1" if r["protein"].startswith("hv1") else r["protein"]
    # like with like: the BM2 His27 reference (7.9) is the FIRST of four stepwise pKas of the tetrad (7.9 / 5.9 / 5.9 / 5.8, mean 6.4)
    extra = []
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "pka_helper", "_tools")); import group_pka as G
        rd = os.path.join(base, "membrane", "bm2kix", "raw", "delphipka"); ph, T = G.load_titra(os.path.join(rd, "titra.csv"))
        st = G.macro_general(ph, T, ["HIS0027%s" % c for c in "ABCD"])
        extra = ["BM2 His27 tetrad, stepwise (DelPhi titration): %s (mean %.2f); measured 7.9 / 5.9 / 5.9 / 5.8 (mean 6.4)"
                 % (" / ".join(G.fmt(v) for v in st), float(np.mean([min(max(v, 0), 14) for v in st]))), ""]
    except Exception as e:
        extra = ["(BM2 His27 tetrad steps not available: %s)" % e, ""]
    md = ["## Membrane set", ""] + summarize([r for r in rows if r["in69"] == "yes"], "69 sites", "group", "protein_fold") + [""] \
        + summarize(rows, "91 sites (HV1 twice)") + [""] + extra
    return md, rows


def suite_peptides(a, base):
    sites = list(csv.DictReader(open(os.path.join(HERE, "sites_peptides.csv"))))
    rows = []
    for s in sites:
        name = "%s_%s" % (s["case"], s["lipid"]); out = os.path.join(base, "peptides", name)
        st = run_one(os.path.join(HERE, "structures", "peptides", s["structure"]), out, ["--center", "0", "--core-thickness", s["core_thickness"]], a.extra, a.np)
        if st != "reused": print("  peptide %-10s %s" % (name, st), flush=True)
        res = read_result(out)
        r = dict(group=s["kind"], ref=float(s["ref"]), bound=s["bound"], site=name, reftype="CpHMD" if s["kind"] == "cphmd" else "experiment", in69="yes")
        for k, _ in COLS: r[k] = value(res, s["resname"], int(s["resnum"]), "A", k)
        rows.append(r)
    return ["## TM peptides", ""] + summarize(rows, "15 references", "group") + [""], rows


def suite_soluble(a, base):
    sites = list(csv.DictReader(open(os.path.join(HERE, "sites_soluble.csv"))))
    for f in dict.fromkeys(s["structure"] for s in sites):
        out = os.path.join(base, "soluble", os.path.splitext(f)[0])
        st = run_one(os.path.join(HERE, "structures", "soluble", f), out, ["--no-membrane"], a.extra, a.np)
        if st != "reused": print("  soluble %-8s %s" % (f, st), flush=True)
    rows = []
    for s in sites:
        res = read_result(os.path.join(base, "soluble", os.path.splitext(s["structure"])[0]))
        r = dict(group=s["resname"], family=s["family"], ref=float(s["ref"]), bound=s["bound"], site="%s %s%s" % (s["pdb"], s["resname"], s["resnum"]),
                 reftype="experiment", in69="yes")
        for k, _ in COLS: r[k] = value(res, s["resname"], int(s["resnum"]), s["chain"], k)
        if any(r[k] is None for k, _ in COLS): print("  missing:", r["site"]); continue
        rows.append(r)
    return ["## Soluble PKAD-R set (independent)", ""] + summarize(rows, "%d sites" % len(rows), "group", "family") + [""], rows


def _members(spec, chains):
    """'GLU194 GLU204' + chains ABC -> [[GLU0194A, GLU0204A], [..B..], ...];  'HIS37:A HIS37:B' -> one explicit group."""
    toks = spec.split()
    if all(":" in t for t in toks):
        return [["%s%04d%s" % (t.split(":")[0][:3], int(t.split(":")[0][3:]), t.split(":")[1]) for t in toks]]
    return [["%s%04d%s" % (t[:3], int(t[3:]), c) for t in toks] for c in chains]


def suite_locked(a, base):
    """Experimental membrane pKas that were NOT used to tune any default: run with the defaults and only ever reported."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "pka_helper", "_tools")); import group_pka as G
    sites = list(csv.DictReader(open(os.path.join(HERE, "sites_locked.csv")))); groups = list(csv.DictReader(open(os.path.join(HERE, "groups_locked.csv"))))
    for f in dict.fromkeys([s["structure"] for s in sites] + [g["structure"] for g in groups]):
        out = os.path.join(base, "locked", os.path.splitext(f)[0])
        print("  locked %-10s %s" % (f, run_one(os.path.join(HERE, "structures", "locked", f), out, [], a.extra, a.np)), flush=True)
    rows, unscored = [], []
    for s in sites:
        res = read_result(os.path.join(base, "locked", os.path.splitext(s["structure"])[0]))
        r = dict(group=s["protein"], ref=float(s["ref"]), bound=s["bound"], site=s["site"], scored=s.get("structure_matches_measurement", "yes") == "yes", note=s.get("note", ""))
        for k, _ in COLS: r[k] = value(res, s["resname"], int(s["resnum"]), s["chains"], k)
        if any(r[k] is None for k, _ in COLS): print("  missing:", s["site"]); continue
        rows.append(r)
    unscored = [r for r in rows if not r["scored"]]; rows = [r for r in rows if r["scored"]]
    md = ["## Locked test set (experimental; never used for tuning)", "",
          "Rule (decided without looking at the predictions): a site is scored only if its structure represents the state in which the pKa was measured.", "",
          "| site | reference | PROPKA | DelPhi | pka-helper |", "|---|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %s%.1f | %.2f | %.2f | %.2f |" % (r["site"], r["bound"], r["ref"], r["PROPKA"], r["DelPhi"], r["pka-helper"]))
    md += [""] + summarize(rows, "%d sites" % len(rows))
    if unscored:
        md += ["", "Listed, not scored (structure does not match the measured state):", ""]
        for r in unscored:
            md.append("- %s: reference %s%.1f, PROPKA %.2f, pka-helper %.2f - %s" % (r["site"], r["bound"], r["ref"], r["PROPKA"], r["pka-helper"], r["note"]))
    md += ["", "Stepwise (macroscopic) pKas of coupled groups, from the DelPhi titration, highest first:", ""]
    for g in groups:
        rd = os.path.join(base, "locked", os.path.splitext(g["structure"])[0], "raw", "delphipka")
        try:
            ph, T = G.load_titra(os.path.join(rd, "titra.csv"))
            per = [G.macro_general(ph, T, m) for m in _members(g["members"], g["chains"]) if all(x in T for x in m)]
            st = [float(np.mean([min(max(p[i], 0.0), 14.0) for p in per])) for i in range(len(per[0]))]
            md.append("- %s: %s (measured %s; %s)" % (g["group"], " / ".join("%.1f" % v for v in st), g["ref_steps"].replace(";", " / "), g["source"]))
        except Exception as e:
            md.append("- %s: not available (%s)" % (g["group"], e))
    return md + [""], rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="all", choices=["all", "membrane", "peptides", "soluble", "locked"])
    ap.add_argument("--tag", default="default", help="name of this configuration (runs/<tag>/)")
    ap.add_argument("--np", type=int, default=None, help="MPI ranks per run (default: pka-helper decides)")
    ap.add_argument("extra", nargs="*", help="after --: extra options for every pka-helper run")
    a = ap.parse_args()
    base = os.path.join(HERE, "runs", a.tag)
    md = ["# pka-helper benchmark: %s" % a.tag, "", "options: %s" % (" ".join(a.extra) or "(defaults)"), ""]
    for name, fn in (("membrane", suite_membrane), ("peptides", suite_peptides), ("soluble", suite_soluble), ("locked", suite_locked)):
        if a.suite in ("all", name):
            print("suite", name, flush=True)
            m, rows = fn(a, base); md += m
            with open(os.path.join(base, "%s_sites.csv" % name), "w") as fo:
                w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if a.suite == "all":      # by reference type, development data only (the locked set stays separate)
        pool = [dict(r, kind=r["reftype"]) for name in ("membrane", "peptides", "soluble") for r in csv.DictReader(open(os.path.join(base, "%s_sites.csv" % name)))
                if r.get("in69", "yes") == "yes"]
        for r in pool:
            for k in ("ref", "PROPKA", "DelPhi", "pka-helper"): r[k] = float(r[k])
        md += ["## By reference type (membrane 69 + peptides + soluble; locked set excluded)", "",
               "Constant-pH MD references are a continuum-solvent simulation and may share DelPhi's biases; experiment is the stricter test.", ""] \
            + summarize(pool, "%d sites" % len(pool), "kind") + [""]
    open(os.path.join(base, "report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
