"""pka-helper - membrane-aware pKa prediction: PROPKA anchor + a curved-membrane DelPhi calculation, blended within bounds.

    pka-helper doctor                       check that PROPKA, MPI and the delphiPKa binary are available
    pka-helper run protein_opm.pdb          PDB (OPM-aligned, with DUM atoms) -> final pKa table
    pka-helper blend --delphi pKa.csv ...   blend an existing DelPhi table with a PROPKA file
    pka-helper membrane protein.pdb ...     only the membrane / DelPhi stages
    pka-helper shape ...                    fit or define a curved membrane
"""
import argparse, csv, os, re, shutil, subprocess, sys, time
from . import __version__

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "_tools")
sys.path.insert(0, TOOLS)

# Settings used for every protein in the benchmark (identical everywhere; see the manuscript, Methods).
UNIFORM = ["--lumen-eps", "40", "--solid-buried-sasa", "3.0", "--solid-spare-partners", "5.0", "--solid-spare-acids",
           "--solid-skip-partnered", "4.0"]      # a buried acid salt-bridged to Lys/Arg/His (N within 4 A) is not put in the eps-4 solid zone


def warn(msg):
    print("pka-helper: WARNING: " + msg, file=sys.stderr, flush=True)


def die(msg, code=2):
    sys.exit("pka-helper: error: " + msg)


def log(msg):
    print("[pka-helper] " + msg, flush=True)


def find_delphi(explicit=None):
    """--delphi-exe, then $PKA_HELPER_DELPHI, then the binary built by 'pip install', then PATH."""
    bundled = os.path.join(HERE, "bin", "delphiPKa")
    return explicit or os.environ.get("PKA_HELPER_DELPHI") or (bundled if os.path.isfile(bundled) else "") or shutil.which("delphiPKa") or ""


def find_propka():
    exe = shutil.which("propka3")
    if exe: return [exe]
    try:
        import propka  # noqa: F401
        return [sys.executable, "-m", "propka"]
    except ImportError:
        return None


def dum_z_values(pdb):
    z = set()
    for l in open(pdb):
        if l.startswith(("ATOM", "HETATM")) and l[17:20].strip() == "DUM":
            z.add(round(float(l[46:54]), 1))
    return z


# Memory model for the DelPhi energy stage, PER MPI RANK (every rank holds its own copy of the grids).
# Peak resident memory was measured on 6 proteins (4.3k-5k atoms ... soluble and membrane, fill 70 and 90, 12 runs) and fits
#     GB = 0.04 + 0.56 per million BASE-grid points + 0.0034 per million map points      (residuals < 0.04 GB)
# where the base grid is a cube of side (largest extent of the protein + 3.4 A) / (fill/100) at 1 grid/A. The fit covers 0.1-0.7 M base points;
# larger proteins are an extrapolation, so we plan with 25% more per point, 5 bytes per map point and a floor of 0.3 GB.
GB_PER_M_BASE = 0.70
GB_PER_M_MAP = 0.005
BASE_GB = 0.3
RESERVE = 0.05


def available_ram_gb():
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable"): return int(l.split()[1]) / 1e6
    except OSError:
        pass
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).split()[0]) / 1e9 * 0.5   # macOS: no 'available' figure; assume half of the installed RAM is free
    except Exception:
        return None


def grid_points(run_dir):
    """Number of points of the dielectric grid written by the maps stage (header of background.dx)."""
    p = os.path.join(run_dir, "background.dx")
    for l in open(p):
        if "gridpositions counts" in l:
            n = [int(x) for x in l.split()[-3:]]
            return n[0] * n[1] * n[2]
    return None


def base_points(run_dir, fill):
    """Points of DelPhi's base grid: a cube of side (largest extent of the protein + 3.4 A) / (fill/100) at 1 grid per A."""
    ext = 0.0
    try:
        xs = [[], [], []]
        for l in open(os.path.join(run_dir, "protein_noHET.pqr")):
            if l.startswith("ATOM"):
                for q, (a0, a1) in enumerate(((30, 38), (38, 46), (46, 54))): xs[q].append(float(l[a0:a1]))
        ext = max(max(c) - min(c) for c in xs if c)
    except (OSError, ValueError):
        return None
    return ((ext + 3.4) * 100.0 / fill) ** 3 if ext > 0 else None


def cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def count_sites(run_dir, skip_arg_lys=False):
    """Number of titratable residues the DelPhi stage works through (matches the rows of pKa.csv).
    Counted as changes of residue along the file, not as distinct (name, chain, number) keys: chains that reuse the same numbering
    (or have a blank chain id) would otherwise be merged and the total underestimated (min.pdb: 139 instead of 196)."""
    names = ("ASP", "GLU", "HIS", "HSP", "HSD", "HSE", "HIP", "HID", "HIE") + (() if skip_arg_lys else ("LYS", "ARG"))
    n = 0; prev = None
    try:
        for l in open(os.path.join(run_dir, "protein_noHET.pqr")):
            if l.startswith("ATOM"):
                key = (l[17:20], l[21], l[22:27])
                if key != prev:
                    prev = key
                    if key[0] in names: n += 1
    except OSError:
        return None
    return n or None


def fmt_time(sec):
    sec = int(max(sec, 0))
    return "%dh %02dmin" % (sec // 3600, sec % 3600 // 60) if sec >= 3600 else "%dmin %02ds" % (sec // 60, sec % 60)


class Progress:
    """Live progress of the DelPhi energy stage: finished sites are the files in energy_ckpt/. Shows done/total and an estimated time left."""

    def __init__(self, run_dir, skip_arg_lys=False):
        import threading, glob
        self.dir = run_dir; self.total = count_sites(run_dir, skip_arg_lys); self.fd = os.dup(2); self.tty = os.isatty(self.fd)
        self.glob = glob; self.stop = threading.Event(); self.t0 = time.time(); self.done0 = None; self.last = 0.0; self.n_lines = 0
        self.th = threading.Thread(target=self.loop, daemon=True)
        self.done0 = self.done()      # sites already finished before this run starts (a resume); counted now, not at the first screen update

    def done(self):
        self.maxdone = max(getattr(self, "maxdone", 0), len(self.glob.glob(os.path.join(self.dir, "energy_ckpt", "site_*.ckpt"))))   # the folder is removed when the stage ends
        return self.maxdone

    def delphi_total(self):
        """The site count DelPhi itself prints at the start of the energy stage ('Titratable sites : N (ASP a, ...)'); None with an older binary."""
        if getattr(self, "_dt", None): return self._dt
        try:
            with open(os.path.join(self.dir, "run.log"), errors="replace") as f: head = f.read(400000)
        except OSError:
            return None
        m = re.search(r"Titratable sites\s*:\s*(\d+)\s*\(([^)]*)\)", head)
        if m:
            self._dt = int(m.group(1)); self._dt_what = m.group(2)
        return getattr(self, "_dt", None)

    def line(self):
        dt = self.delphi_total()
        if dt: self.total = dt
        d = self.done()
        if self.done0 is None: self.done0 = d
        el = time.time() - self.t0; new = d - self.done0
        if getattr(self, "finished", False):
            return "DelPhi energy stage finished: %s" % fmt_time(el)
        if not self.total: return "DelPhi energy stage: %d sites done, %s elapsed" % (d, fmt_time(el))
        if d >= self.total:      # all sites have an energy file: what follows (pair energies, titration, writing results) has no site counter
            try: txt = open(os.path.join(self.dir, "run.log"), errors="replace").read()
            except OSError: txt = ""
            what = "titration" if "Start titration" in txt else "pair energies / clustering"
            return "DelPhi energy stage done (%d sites%s), now %s, %s elapsed (no time estimate for this part)" % (d, "" if d == self.total else "; the site count expected was %d" % self.total, what, fmt_time(el))
        eta = ""
        if new >= 3 and el >= 20:
            eta = ", about %s left" % fmt_time(el / new * (self.total - d))
        elif d < self.total:
            rate = self.file_rate()      # after a resume nothing has finished yet in THIS run: use the time stamps of the checkpoint files
            if rate: eta = ", about %s left (from the earlier run's speed)" % fmt_time((self.total - d) / rate)
            else: eta = ", estimating time left ..."
        restored = " (%d restored from an earlier run)" % self.done0 if self.done0 else ""
        return "DelPhi energy stage: %d/%d sites (%d%%)%s, %s elapsed%s" % (d, self.total, 100 * d // self.total, restored, fmt_time(el), eta)

    def file_rate(self):
        """Sites per second from the modification times of the checkpoint files (None if too few or too close together)."""
        try:
            t = sorted(os.path.getmtime(f) for f in self.glob.glob(os.path.join(self.dir, "energy_ckpt", "site_*.ckpt")))
        except OSError:
            return None
        if len(t) < 5 or t[-1] - t[0] < 20: return None
        return (len(t) - 1) / (t[-1] - t[0])

    def emit(self, final=False):
        msg = "[pka-helper] " + self.line()
        if self.tty: os.write(self.fd, ("\r" + msg + "\033[K" + ("\n" if final else "")).encode())
        elif final or time.time() - self.last >= 60:
            os.write(self.fd, (msg + "\n").encode()); self.last = time.time()

    def loop(self):
        while not self.stop.wait(3 if self.tty else 10): self.emit()

    def __enter__(self): self.th.start(); return self

    def __exit__(self, et, *e):
        self.stop.set(); self.th.join(timeout=5); self.finished = et is None; self.emit(final=True); os.close(self.fd)


def choose_np(requested, run_dir, fill, force=False):
    """-> number of MPI ranks. Estimates peak RAM per rank from the base-grid and map sizes and caps the ranks by free RAM and by cores."""
    nb, nm = base_points(run_dir, fill), grid_points(run_dir)
    per = (BASE_GB + nb / 1e6 * GB_PER_M_BASE + (nm or 0) / 1e6 * GB_PER_M_MAP) if nb else None
    npts = nb
    free = available_ram_gb()
    if per is None or free is None:
        warn("cannot estimate memory use (%s); using %s rank(s)" % ("no grid size" if per is None else "free RAM unknown", requested or 2))
        return requested or 2
    fit = int((free * (1 - RESERVE)) // per)
    log("memory: about %.1f GB per MPI rank (%.1f M base-grid points, %.1f M map points, fill %g%%); %.1f GB free, so %d rank(s) fit; %d core(s)" % (per, nb / 1e6, (nm or 0) / 1e6, fill, free, fit, cores()))
    if fit < 1:
        msg = "one rank needs about %.1f GB but only %.1f GB is free (keeping %d%% in reserve)" % (per, free, RESERVE * 100)
        if not force: die(msg + ". Free memory, use a larger machine, or override with --force-memory.")
        warn(msg + " - continuing because of --force-memory; the run may be killed")
        fit = 1
    if requested:
        if requested > fit and force:
            log("using %d MPI rank(s) as requested; the estimate says only %d fit, so memory may run short (--force-memory)" % (requested, fit))
            return requested
        if requested > fit and not force:
            warn("--np %d would need about %.1f GB, more than the %.1f GB it is safe to use here; using %d instead (override with --force-memory)" % (requested, per * requested, free * (1 - RESERVE), max(fit, 1)))
            return max(fit, 1)
        log("using %d MPI rank(s) as requested" % requested)
        return requested
    n = max(1, min(cores(), fit))
    log("using %d MPI rank(s)" % n)
    return n


LOGFILE = None      # set by 'run': the detailed output of every stage goes here unless -v is given


def run_module(mod, argv, quiet=True):
    """Call tools/<mod>.py main() with the given argv. With a log file open and quiet=True, all output (also from child processes) goes to the log."""
    old = sys.argv
    sys.argv = [mod + ".py"] + argv
    saved = None
    try:
        if quiet and LOGFILE:
            sys.stdout.flush(); sys.stderr.flush()
            saved = (os.dup(1), os.dup(2)); lf = open(LOGFILE, "a"); lf.write("\n$ %s %s\n" % (mod, " ".join(argv))); lf.flush()
            os.dup2(lf.fileno(), 1); os.dup2(lf.fileno(), 2)
        m = __import__(mod)
        m.main()
    except SystemExit as e:
        if saved and e.code not in (None, 0):
            os.dup2(saved[0], 1); os.dup2(saved[1], 2); saved = None
            tail = open(LOGFILE).read()[-1500:]
            print("pka-helper: the '%s' step failed. Last lines of %s:\n%s" % (mod, LOGFILE, tail), file=sys.stderr)
            try:      # the reason is usually in the program's own log (run.log), not in this one
                rl = os.path.join(os.path.dirname(LOGFILE), "delphipka", "run.log")
                rt = open(rl).read()[-1200:] if os.path.isfile(rl) else ""
                if rt: print("Last lines of %s:\n%s" % (rl, rt), file=sys.stderr)
                if "alloc-error" in rt or "not enough slots" in rt:
                    print("Hint: Open MPI refused to start that many ranks (it counts physical cores as slots). Re-run with fewer ranks (--np N) or add --oversubscribe.", file=sys.stderr)
            except OSError:
                pass
        raise
    finally:
        if saved:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(saved[0], 1); os.dup2(saved[1], 2); os.close(saved[0]); os.close(saved[1])
        sys.argv = old


def cmd_doctor(a):
    ok = True
    def chk(name, good, detail):
        nonlocal ok
        print("  %-22s %s  %s" % (name, "ok " if good else "MISSING", detail))
        ok = ok and good
    print("pka-helper %s" % __version__)
    try:
        import numpy
        chk("numpy", True, numpy.__version__)
    except ImportError:
        chk("numpy", False, "pip install numpy")
    try:
        import scipy
        chk("scipy", True, scipy.__version__)
    except ImportError:
        chk("scipy", False, "pip install scipy")
    pp = find_propka()
    chk("PROPKA 3", bool(pp), " ".join(pp) if pp else "pip install propka")
    mp = shutil.which("mpirun")
    chk("mpirun", bool(mp), mp or "install an MPI (OpenMPI / MPICH)")
    de = find_delphi(a.delphi_exe)
    chk("delphiPKa binary", bool(de) and os.path.isfile(de), de or "not found: build it (BUILDING.md) and set PKA_HELPER_DELPHI or put it on PATH")
    _par = os.environ.get("PKA_HELPER_PARAM") or os.path.join(HERE, "param")
    chk("force-field files", os.path.isdir(_par), _par)
    return 0 if ok else 1


def standard_residues(pdb, outdir, plan=None):
    """Modified residues with the charge of a standard residue (MSE, FME, HYP, MLY ... see hetero.py) -> ATOM records of that residue;
    without this they would be left out with the other HETATM groups, leaving a hole in the chain (4ATV NhaA: 11 Met per chain).
    Alternate conformations: only the first (blank or A) is kept; the B, C ... copies made the chain check see every residue twice and
    split the chain (1LNI).  -> (path to use, message or None). The input file is never changed."""
    import hetero as HT
    if plan is None: plan = HT.classify(pdb)
    lines = open(pdb, errors="replace").read().splitlines(); out = []; alt = set()
    for l in lines:
        if l.startswith(("ATOM", "HETATM")) and len(l) > 16 and l[16] != " ":
            if l[16] != "A": alt.add((l[21], l[22:27])); continue          # alternate conformations: keep the first (blank or A) only
            l = l[:16] + " " + l[17:]
        out.append(l)
    out, n = HT.map_parents(out, plan)
    if not n and not alt: return pdb, None
    path = os.path.join(outdir, os.path.splitext(os.path.basename(pdb))[0] + "_std.pdb")
    open(path, "w").write("\n".join(out) + "\n")
    msg = []
    if n.get("MSE"): msg.append("%d selenomethionine (MSE) residues read as methionine" % n["MSE"])
    other = {k: v for k, v in n.items() if k != "MSE"}
    if other: msg.append("modified residues read as the standard residue: " + ", ".join("%s x%d -> %s" % (k, v, HT.PARENT[k][0]) for k, v in sorted(other.items())))
    if alt: msg.append("alternate conformations: the first (blank or A) kept for %d residues" % len(alt))
    return path, "; ".join(msg) + " (wrote %s)" % path


def neutral_partners(run_dir, ph, blended=None, charged=()):
    """After a first pass: the salt-bridge partners recorded in solid_partners.txt ('ACID partner ...') that are neutral at pH ph (bases:
    pKa < ph).  The partner pKa is pka-helper's blended value from `blended` (a pka_helper.csv of the first pass) when given, else DelPhi's
    raw value from pKa.csv.  Names in `charged` (e.g. a retinal Schiff-base Lys) always count as charged.
    -> (set of neutral partner names, set of acids whose partners are ALL neutral)."""
    import re
    pf = os.path.join(run_dir, "solid_partners.txt"); kf = os.path.join(run_dir, "pKa.csv")
    if not os.path.isfile(pf): return set(), set()
    pk = {}
    if blended and os.path.isfile(blended):
        for r in csv.DictReader(open(blended)):
            try: pk[r["residue"] + r.get("chain", "")] = float(r["blended"])
            except (ValueError, KeyError): pass
    elif os.path.isfile(kf):
        for l in open(kf):
            t = l.split()
            m = re.match(r"^([A-Z]{3})(\d{4})(\S*)$", t[0]) if t else None
            if m and len(t) > 1:
                try: pk["%s%d%s" % (m.group(1), int(m.group(2)), m.group(3))] = float(t[1].rstrip("*"))
                except ValueError: pass
    for c in charged: pk[c] = 99.0
    neutral, lost = set(), set()
    for l in open(pf):
        t = l.split()
        if len(t) < 2: continue
        acid, parts = t[0], t[1:]
        nn = [p for p in parts if p in pk and pk[p] < ph]          # a base below the pH is neutral; unknown partners (e.g. Schiff base) count as charged
        neutral |= set(nn)
        if len(nn) == len(parts): lost.add(acid)
    return neutral, lost


def apply_cofactors(pdb, specs, outdir):
    """Old interface (--cofactor only): keep the listed HETATM residues as fixed charge + volume, leave every other HETATM group out."""
    import hetero as HT
    plan = HT.classify(pdb, specs, mode="drop")
    if not any(plan.get(sp.partition(":")[0].strip().upper(), ("drop",))[0] != "drop" for sp in specs):
        die("--cofactor: no HETATM residue named %s in %s" % (", ".join(sp.partition(":")[0] for sp in specs), pdb))
    path, summ = HT.write_delphi_input(pdb, outdir, plan)
    return path, "cofactors kept in the DelPhi calculation (fixed charge, not titrated): %s" % ", ".join("%s x%d (net %+g each)" % (k, v[1], v[2]) for k, v in sorted(summ.items()))


def unique_chains(pdb, outdir):
    """Give every protein segment its own chain id when chain ids are blank or the same (chain, number) occurs twice.
    DelPhiPKa and PROPKA both identify a residue by chain and number; min.pdb had blank chain ids and the same numbers eight times, which
    merged those residues and made the titration try 2^40 states. A segment ends at TER, a chain change or a number that goes backwards.
    -> (path to use, message or None). The input file is never changed."""
    lines = open(pdb, errors="replace").read().splitlines()
    seen, dup, blank = set(), False, False
    for l in lines:
        if l.startswith("ATOM") and "DUM" not in l[17:20]:
            key = (l[21], l[22:27])
            blank = blank or l[21] == " "
            if l[12:16].strip() == "CA":
                if key in seen: dup = True
                seen.add(key)
    if not (blank or dup): return pdb, None
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    out, seg, prev_chain, prev_num = [], -1, None, None
    start, in_seg, c_prev, n_cur, cur_key = True, set(), None, None, None
    for l in lines:
        if l.startswith("TER"): out.append(l); start = True; continue
        if l.startswith(("ATOM", "HETATM")) and "DUM" not in l[17:20]:
            ch = l[21]
            try: num = int(l[22:26])
            except ValueError: num = prev_num
            key = (ch, l[22:27], l[17:20])
            new_res = key != cur_key
            brk = start or ch != prev_chain or (prev_num is not None and num is not None and num < prev_num)
            if new_res and not brk:
                if key[:2] in in_seg: brk = True                                   # the same number twice in one segment
                elif l[12:16].strip() == "N" and c_prev is not None:               # peptide bond longer than 2.5 A: a different chain
                    try:
                        xyz = [float(l[30 + 8 * i:38 + 8 * i]) for i in range(3)]
                        if sum((xyz[i] - c_prev[i]) ** 2 for i in range(3)) > 6.25: brk = True
                    except ValueError: pass
            if brk: seg += 1; start = False; in_seg = set(); c_prev = None
            if new_res: cur_key = key; in_seg.add(key[:2])
            if l[12:16].strip() == "C":
                try: c_prev = [float(l[30 + 8 * i:38 + 8 * i]) for i in range(3)]
                except ValueError: pass
            prev_chain, prev_num = ch, num
            if seg >= len(letters): return pdb, "more than %d segments; chain ids not changed" % len(letters)
            l = l[:21] + letters[seg] + l[22:]
        out.append(l)
    path = os.path.join(outdir, os.path.splitext(os.path.basename(pdb))[0] + "_chains.pdb")
    open(path, "w").write("\n".join(out) + "\n")
    return path, "chain ids are blank or repeated (same chain and number twice): wrote %s with one chain letter per segment (A, B, ...); results use these letters" % path


# Blend rules per membrane-core dielectric.  eps 2: pka-helper 0.1.5.  eps 10: refitted in 0.1.7 on the TM peptides + benchmark with every
# protein held out in turn (see README "Lipid dielectric"); c0, k, lipid-cap-base, Lys rule unchanged.
RULES = {2: dict(acid_cap=0.5, lipid_cap=3.0, his_down=1.5, his_lip_cap=0.0, acid_up_scale=0.3),
         10: dict(acid_cap=2.0, lipid_cap=5.0, his_down=0.0, his_lip_cap=2.0, acid_up_scale=0.0)}


def rule_set(a):
    if getattr(a, "no_membrane", False): return RULES[2]          # no lipid: the lipid dielectric (and its rules) do not apply
    return RULES[10] if getattr(a, "lipid_eps", 2.0) >= 6.0 else RULES[2]


def rule_value(a, name):
    v = getattr(a, name, None)
    return rule_set(a)[name] if v is None else v


def _sub_argv(a, pdb, out, drop_flags=("--ensemble", "--membrane-scan"), drop_valued=("--out", "--ensemble-max"), extra=()):
    """The user's own `run` arguments for a sub-run: same options, another input/output; the orchestration flags removed."""
    t = list(a._argv); res = []; i = 0; seen_pdb = False
    while i < len(t):
        x = t[i]
        if x == "run" and not res: i += 1; continue
        if x in drop_flags: i += 1; continue
        if x in drop_valued: i += 2; continue
        if any(x.startswith(v + "=") for v in drop_valued): i += 1; continue
        if not seen_pdb and x == a._pdb_arg: seen_pdb = True; i += 1; continue
        if x == "--membrane-arg": res += t[i:]; break
        res.append(x); i += 1
    rest = [x for x in res]
    if "--membrane-arg" in rest:
        k = rest.index("--membrane-arg"); return [pdb, "--out", out] + list(extra) + rest[:k] + rest[k:]
    return [pdb, "--out", out] + list(extra) + rest


def _run_sub(argv, log_path):
    cmd = [sys.executable, "-c", "import sys; sys.path.insert(0, %r); from pka_helper.cli import main; sys.exit(main())" % os.path.dirname(HERE), "run"] + argv
    with open(log_path, "w") as lg:
        return subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT).returncode


def _read_csv(p):
    return list(csv.DictReader(open(p))) if os.path.isfile(p) else []


def _num(x):
    try: return float(str(x).rstrip("*"))
    except ValueError: return None


def cmd_ensemble(a):
    import numpy as np
    t0 = time.time()
    lines = open(a.pdb, errors="replace").read().splitlines()
    models, cur, outside = [], None, []
    for l in lines:
        if l.startswith("MODEL"): cur = []; continue
        if l.startswith("ENDMDL"):
            if cur is not None: models.append(cur)
            cur = None; continue
        (cur if cur is not None else outside).append(l)
    if len(models) < 2: die("--ensemble: %s has %d MODEL record(s); nothing to average" % (a.pdb, len(models)))
    if a.ensemble_max: models = models[:a.ensemble_max]
    out = os.path.abspath(a.out or os.path.splitext(os.path.basename(a.pdb))[0] + "_pka"); res = os.path.join(out, "results"); os.makedirs(res, exist_ok=True)
    head = [l for l in outside if not l.startswith(("END", "MASTER", "CONECT"))]
    tables = []
    for i, m in enumerate(models, 1):
        mdir = os.path.join(out, "models", "model_%02d" % i); os.makedirs(mdir, exist_ok=True)
        mp = os.path.join(mdir, "model_%02d.pdb" % i)
        open(mp, "w").write("\n".join(head + m) + "\nEND\n")
        if not os.path.isfile(os.path.join(mdir, "results", "pka_helper.csv")):
            log("ensemble: model %d of %d" % (i, len(models)))
            rc = _run_sub(_sub_argv(a, mp, mdir), os.path.join(mdir, "run.log"))
            if rc: warn("model %d failed (see %s); left out" % (i, os.path.join(mdir, "run.log"))); continue
        tables.append(_read_csv(os.path.join(mdir, "results", "pka_helper.csv")))
    if not tables: die("--ensemble: no model finished")
    import blend_pka as BP
    agg = {}
    for t in tables:
        for r in t:
            agg.setdefault((r["residue"], r.get("chain", "")), []).append(r)
    rows = []
    for (resn, ch), rs in agg.items():
        P = [_num(r["PROPKA"]) for r in rs]; D = [min(max(v, 0.0), 14.0) for v in (_num(r["DelPhiPKa"]) for r in rs) if v is not None]; Bl = [_num(r["blended"]) for r in rs]
        fp = sum(1 for b in Bl if b > a.ph) / len(Bl); mb = float(np.mean(Bl))
        Sg = [v for v in (_num(r.get("sigma", "")) for r in rs) if v is not None]
        sg = float(np.sqrt(np.mean(np.square(Sg)) + np.var(Bl))) if Sg else None      # method error and the spread over the models
        rows.append({"residue": resn, "chain": ch, "PROPKA": "%.2f" % np.mean(P), "DelPhiPKa": "%.2f" % np.mean(D) if D else rs[0]["DelPhiPKa"], "saturated": str(sum(1 for r in rs if r.get("saturated") == "1")),
                     "blended": "%.2f" % mb, "blended_sd": "%.2f" % np.std(Bl), "blended_min": "%.2f" % min(Bl), "blended_max": "%.2f" % max(Bl),
                     "protonated_fraction": "%.2f" % fp, "n_models": str(len(rs)), "state_at_pH7": "protonated" if mb > a.ph else "deprotonated",
                     "confidence": "uncertain" if 0 < fp < 1 or abs(mb - a.ph) < 0.5 else "confident",
                     "structure_warning": " ".join(sorted({w for r in rs for w in r.get("structure_warning", "").split()})),
                     "sigma": "%.2f" % sg if sg else "", "P_protonated": "%.2f" % BP.p_protonated(mb, sg, a.ph) if sg else ""})
    rows.sort(key=lambda r: (r["chain"], int("".join(c for c in r["residue"] if c.isdigit()) or 0)))
    with open(os.path.join(res, "pka_helper.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    shown = [r for r in rows if float(r["blended_sd"]) >= 0.5 or r["confidence"] == "uncertain"]
    txt = ["Ensemble of %d models (pka-helper %s, pH %.1f): mean and spread of the final pKa over the models." % (len(tables), __version__, a.ph),
           "The spread is the part of the uncertainty that comes from the structure; it does not include the method's own error.", "",
           "%-9s %-5s %7s %7s %7s %13s  %s" % ("residue", "chain", "PROPKA", "pKa", "sd", "min - max", "protonated in")]
    for r in sorted(shown, key=lambda r: -float(r["blended_sd"])):
        txt.append("%-9s %-5s %7s %7s %7s %13s  %3.0f%% of models" % (r["residue"], r["chain"], r["PROPKA"], r["blended"], r["blended_sd"],
                   "%s - %s" % (r["blended_min"], r["blended_max"]), 100 * float(r["protonated_fraction"])))
    txt.append("\n%d groups; %d listed (spread >= 0.5 pK or the pH %.1f call is not the same in every model). All of them are in results/pka_helper.csv;"
               " each model's full output is in models/." % (len(rows), len(shown), a.ph))
    open(os.path.join(res, "ensemble.txt"), "w").write("\n".join(txt) + "\n")
    el = time.time() - t0
    print("\n".join(txt)); print("\nResults in %s\n(total time %d min %d s, %d model runs)" % (out, el // 60, el % 60, len(tables)))
    return 0


def cmd_membrane_scan(a):
    if a.no_membrane: die("--membrane-scan needs a membrane")
    zs = sorted(dum_z_values(a.pdb))
    t0 = a.core_thickness if a.core_thickness is not None else (zs[-1] - zs[0] if len(zs) == 2 else 30.0)
    if a.core_thickness is None and len(zs) > 2:
        warn("--membrane-scan with a curved membrane: the core thickness of the fitted shape is varied around %.1f A (assumed); give --core-thickness to set it" % t0)
    a._t_start = time.time()
    out = os.path.abspath(a.out or os.path.splitext(os.path.basename(a.pdb))[0] + "_pka")
    side = {}
    for d in (-3.0, 3.0):
        t = t0 + d; sd = os.path.join(out, "raw", "membrane_scan", "core_%.1f" % t)
        if not os.path.isfile(os.path.join(sd, "results", "pka_helper.csv")):
            log("membrane scan: core thickness %.1f A" % t)
            os.makedirs(sd, exist_ok=True)
            argv = [x for x in _sub_argv(a, a.pdb, sd)]
            if "--core-thickness" in argv: k = argv.index("--core-thickness"); argv[k + 1] = "%.2f" % t
            else: argv += ["--core-thickness", "%.2f" % t]
            if _run_sub(argv, sd + ".log"): die("membrane scan run failed: see %s.log" % sd)
        side[t] = {(r["residue"], r.get("chain", "")): _num(r["blended"]) for r in _read_csv(os.path.join(sd, "results", "pka_helper.csv"))}
    a.membrane_scan = False; a._defer_end = True
    if a.core_thickness is None: a.core_thickness = t0 if len(zs) != 2 else None
    rc = cmd_run(a)
    res = os.path.join(out, "results"); p = os.path.join(res, "pka_helper.csv"); rows = _read_csv(p)
    lines = ["Membrane-position sensitivity: final pKa with the hydrophobic core %.1f / %.1f / %.1f A thick (the middle one is the main result)." % (t0 - 3, t0, t0 + 3),
             "Listed: groups that move by >= 1 pK or whose pH %.1f call changes. A large range means the answer depends on where the membrane is." % a.ph, "",
             "%-9s %-5s %8s %8s %8s" % ("residue", "chain", "%.1f A" % (t0 - 3), "%.1f A" % t0, "%.1f A" % (t0 + 3))]
    for r in rows:
        k = (r["residue"], r.get("chain", "")); v = [side[t0 - 3].get(k), _num(r["blended"]), side[t0 + 3].get(k)]
        vv = [x for x in v if x is not None]
        r["membrane_lo"], r["membrane_hi"] = "%.2f" % min(vv), "%.2f" % max(vv)
        r["membrane_flips_call"] = str(int(len({x > a.ph for x in vv}) > 1))
        if max(vv) - min(vv) >= 1.0 or r["membrane_flips_call"] == "1":
            lines.append("%-9s %-5s %8s %8s %8s%s" % (k[0], k[1], *["%.2f" % x if x is not None else "-" for x in v], "   call flips" if r["membrane_flips_call"] == "1" else ""))
    if len(lines) == 4: lines.append("(none: every pKa moves by less than 1 pK and keeps its pH %.1f call)" % a.ph)
    with open(p, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    open(os.path.join(res, "membrane_scan.txt"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    end = getattr(a, "_end_text", "")
    if end:
        el = time.time() - a._t_start
        end = re.sub(r"\(total time \d+ min \d+ s\)", "(total time %d min %d s, including the two membrane-scan runs)" % (el // 60, el % 60), end)
        print(end.replace("  results/pkas.txt", "  results/membrane_scan.txt           how much each pKa depends on the membrane position\n  results/pkas.txt", 1))
    return rc


def cmd_run(a):
    if getattr(a, "ensemble", False): return cmd_ensemble(a)
    if getattr(a, "membrane_scan", False): return cmd_membrane_scan(a)
    if a.lipid_eps not in (2.0, 10.0):
        warn("--lipid-eps %g: the blend rules were fitted at 2 and 10 only; using the rules for %d" % (a.lipid_eps, 10 if a.lipid_eps >= 6 else 2))
    if not os.path.isfile(a.pdb): die("PDB not found: " + a.pdb)
    delphi = find_delphi(a.delphi_exe)
    if not delphi or not os.path.isfile(delphi): die("delphiPKa binary not found. Build it (BUILDING.md), then set PKA_HELPER_DELPHI or pass --delphi-exe. Run 'pka-helper doctor'.")
    pp = find_propka()
    if not pp: die("PROPKA 3 not found (pip install propka). Run 'pka-helper doctor'.")
    global LOGFILE
    stem = os.path.splitext(os.path.basename(a.pdb))[0]
    out = os.path.abspath(a.out or stem + "_pka")
    raw, res = os.path.join(out, "raw"), os.path.join(out, "results")
    for d_ in (os.path.join(raw, "propka"), res): os.makedirs(d_, exist_ok=True)
    import hetero as HT
    het_plan = HT.classify(a.pdb, a.cofactor, a.hetero, [x for d in a.drop_het for x in d.split(",")], a.keep_ions)
    a.pdb, std_msg = standard_residues(a.pdb, raw, het_plan)
    a.pdb, chain_msg = unique_chains(a.pdb, raw)
    LOGFILE = None if a.verbose else os.path.join(raw, "log.txt")
    if LOGFILE: open(LOGFILE, "w").write("pka-helper %s  %s\n" % (__version__, " ".join(sys.argv)))
    if std_msg: log(std_msg)
    if chain_msg: warn(chain_msg)
    try:
        from structure_check import check as _scheck
        for _k, _c, _m in _scheck(a.pdb, kept=het_plan):
            if _c != "input": continue
            if "kept with a fixed charge" in _m or "read as the standard residue" in _m: log(_m)      # informational
            else: warn(_m)
    except Exception as _e:
        log("structure check skipped (%s)" % _e)
    t_start = getattr(a, "_t_start", None) or time.time()      # --membrane-scan: counted from the start of the whole scan
    zs = dum_z_values(a.pdb)
    shape = a.membrane
    if getattr(a, "no_membrane", False):
        # soluble protein (or a membrane protein computed in water): the membrane slab is placed far outside the box, so the maps are
        # water + protein only; every other setting (solid zone, blend rules) stays the same
        ax = "xyz".index(a.axis)
        ca = [float(l[30 + 8 * ax:38 + 8 * ax]) for l in open(a.pdb) if l.startswith("ATOM") and l[12:16].strip() == "CA"]
        a.center = (max(ca) if ca else 0.0) + 1000.0; a.core_thickness = 10.0; shape = "flat"; zs = []
        log("no membrane (--no-membrane): protein in water only")
    elif a.membrane_shape:
        if not os.path.isfile(a.membrane_shape): die("membrane shape file not found: " + a.membrane_shape)
        shape = "file"
        log("membrane geometry taken from %s (the structure must be in the same coordinate frame)" % a.membrane_shape)
    elif not zs and not getattr(a, "no_membrane", False):
        ax = "xyz".index(a.axis)
        ca = [float(l[30 + 8 * ax:38 + 8 * ax]) for l in open(a.pdb) if l.startswith("ATOM") and l[12:16].strip() == "CA"]
        mid = sum(ca) / len(ca) if ca else 0.0
        c = a.center if a.center is not None else 0.0
        warn("no DUM atoms in %s, so the membrane position is NOT known. Assuming a flat membrane with normal along %s, mid-plane at %s = %.1f A and a %.0f A hydrophobic core%s."
             % (a.pdb, a.axis, a.axis, c, a.core_thickness if a.core_thickness is not None else 30.0, "" if a.center is not None and a.core_thickness is not None else " (defaults; set --center / --core-thickness)"))
        warn("the protein's C-alpha centre lies at %s = %.1f A. If the structure is not oriented with its membrane normal along %s and centred on the membrane, the results are not meaningful: orient it first (OPM or PPM) or give --center / --core-thickness / --axis." % (a.axis, mid, a.axis))
        if abs(mid - c) > 8: warn("the C-alpha centre is %.1f A from the assumed mid-plane: this looks wrong for a transmembrane protein." % abs(mid - c))
        if shape == "curved" and not a.lipids: die("--membrane curved needs dummy atoms or --lipids")
        if shape == "auto": shape = "flat"
    elif shape == "auto":
        shape = "flat" if len(zs) <= 4 else "curved"
        log("membrane shape: %s (%d distinct dummy-atom heights)" % (shape, len(zs)))
    # 1. PROPKA on the protein without dummy atoms
    nodum = os.path.join(raw, "propka", stem + "_nodum.pdb")
    with open(nodum, "w") as fo:
        for l in open(a.pdb):
            if l.startswith(("ATOM", "TER", "END")) and "DUM" not in l[17:20]: fo.write(l)
    log("1/3 PROPKA (runs alongside the next step)")
    pka_file = os.path.join(raw, "propka", stem + "_nodum.pka")
    pk_log = open(os.path.join(raw, "propka", "propka_run.log"), "w")
    pk_proc = subprocess.Popen(pp + ["-q", nodum], cwd=os.path.join(raw, "propka"), stdout=pk_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

    def finish_propka():
        pk_proc.wait(); pk_log.close()
        if not os.path.isfile(pka_file):
            try: tail = open(os.path.join(raw, "propka", "propka_run.log"), errors="replace").read()[-800:]
            except OSError: tail = ""
            die("PROPKA produced no .pka file:\n" + tail)
    # 2. membrane DelPhi calculation
    run_dir = os.path.join(raw, "delphipka")
    pdb_copy = os.path.join(raw, os.path.basename(a.pdb)); (None if os.path.abspath(a.pdb) == os.path.abspath(pdb_copy) else shutil.copy(a.pdb, pdb_copy))
    cof_args = []
    schiff_all = HT.schiff_base_lysines(open(a.pdb, errors="replace").read().splitlines())
    schiff = schiff_all if a.hold_schiff_base else []
    het_src = a.pdb
    if schiff:             # retinal Schiff base (Lys bound to a separate RET): held charged, as measured (pKa > 12), not titrated as a plain Lys
        het_src = os.path.join(raw, os.path.splitext(os.path.basename(a.pdb))[0] + "_schiff.pdb")
        open(het_src, "w").write("\n".join(HT.hold_schiff_base(open(a.pdb, errors="replace").read().splitlines(), schiff)) + "\n")
        het_plan = dict(het_plan); het_plan["LYR"] = ("charged",) + HT.FIXED["LYR"]
        log("retinal Schiff base held charged (+1, not titrated): %s" % ", ".join("Lys%s %s" % (k[1].strip(), k[0]) for k in schiff))
    het_path, het_summary = HT.write_delphi_input(het_src, raw, het_plan)
    if a.keep_waters:        # EXPERIMENTAL: buried crystal waters as explicit, H-bond-oriented TIP3P molecules (fixed charges + volume)
        _src_lines = open(a.pdb, errors="replace").read().splitlines()
        _ew = HT.explicit_waters(_src_lines, HT.buried_waters(_src_lines))
        if _ew:
            if not het_path:
                het_path = os.path.join(raw, os.path.splitext(os.path.basename(a.pdb))[0] + "_hetero.pdb")
                open(het_path, "w").write("\n".join(l for l in _src_lines if l.startswith(("ATOM", "TER", "CRYST1"))) + "\n")
            _L = open(het_path).read().splitlines(); _e = [i for i, l in enumerate(_L) if l.startswith("END")]
            _L = _L[:_e[0]] + _ew + _L[_e[0]:] if _e else _L + _ew
            open(het_path, "w").write("\n".join(_L) + "\n")
        log("--keep-waters explicit: %d buried crystal waters kept as explicit TIP3P waters (H placed towards H-bond acceptors)" % (len(_ew) // 3))
    if het_path:          # HETATM groups kept in the calculation (fixed charge or uncharged volume): no artificial water holes
        pdb_copy = het_path
        cof_args = ["--set", "Remove HETATM : F", "--set", "HETATM in PQR format : T"]
        if zs and shape == "flat" and not getattr(a, "no_membrane", False):      # the dummy atoms are not in the DelPhi input: pass the slab explicitly
            if a.center is None: a.center = (max(zs) + min(zs)) / 2.0
            if a.core_thickness is None: a.core_thickness = max(zs) - min(zs)
    argv = [pdb_copy, "--out", run_dir, "--exe", delphi] + UNIFORM + cof_args
    argv += ["--set", "Base grid fill (percent) : %g" % a.base_fill]   # 90: smaller base box, about half the memory, same pKas as 70 in tests (see CUSTOM_DIELECTRIC.md)
    if a.skip_arg_lys: argv += ["--set", "Skip Lys and Arg : T"]
    if shape == "curved":
        if het_path: argv += ["--lipids", os.path.abspath(a.pdb), "--lipid-resnames", "DUM", "--lipid-atoms", "*", "--p-offset", "0"]
        else: argv.append("--opm-curved")
    if shape == "file": argv += ["--membrane-shape", os.path.abspath(a.membrane_shape)]
    if a.lipids: argv += ["--lipids", a.lipids]
    if a.center is not None: argv += ["--center", str(a.center)]
    if a.core_thickness is not None: argv += ["--core-thickness", str(a.core_thickness)]
    if a.axis != "z": argv += ["--axis", a.axis]
    if a.lipid_eps != 2.0: argv += ["--eps-core", "%g" % a.lipid_eps]
    argv += a.membrane_arg
    def blend_args(out_csv):
        bl = ["--delphi", os.path.join(run_dir, "pKa.csv"), "--propka", pka_file, "--pqr", os.path.join(run_dir, "protein_noHET.pqr"),
              "--ph", str(a.ph), "--acid-up-cap", str(rule_value(a, "acid_cap")), "--lipid-cap", str(rule_value(a, "lipid_cap")), "--lipid-auto", run_dir,
              "--his-down", str(rule_value(a, "his_down")), "--his-lip-cap", str(rule_value(a, "his_lip_cap")), "--acid-up-scale", str(rule_value(a, "acid_up_scale")),
              "--out", out_csv]
        _uf = os.path.join(run_dir, "unpartnered_acids.txt")
        if os.path.isfile(_uf) and os.path.abspath(out_csv) != os.path.abspath(os.path.join(raw, "pass1_blend.csv")):
            bl += ["--unpartnered-acids", ",".join(x.strip() for x in open(_uf) if x.strip())]
            if a.uncap_unpartnered: bl += ["--uncap-unpartnered"]
        if a.no_membrane: bl += ["--soluble-run"]
        if a.skip_arg_lys: bl += ["--propka-only", "LYS,ARG"]
        if schiff: bl += ["--fixed-charged", ",".join("LYS:%s:%s" % (k[1].strip(), k[0]) for k in schiff)]
        if a.no_lipid_auto: bl.append("--no-lipid-auto")
        bl += ["--lys-down-c0", str(a.lys_down_c0), "--lipid-cap-base", str(a.lipid_cap_base), "--lipid-depth-min", str(a.lipid_depth_min), "--lipid-depth-full", str(a.lipid_depth_full)]
        if a.no_lipid_depth: bl.append("--no-lipid-depth")
        if a.lipid_depth_acids: bl.append("--lipid-depth-acids")
        if a.lipid_acids: bl += ["--lipid-acids", a.lipid_acids]
        if a.no_his_rule: bl.append("--no-his-rule")
        if a.hbond_term: bl += ["--hbond-shift", "1.5"]
        if a.chain: bl += ["--chain", a.chain]
        for f in a.alt: bl += ["--alt", f]
        return bl

    if a.fast:
        finish_propka()
        return fast_mode(a, out, raw, res, run_dir, argv, pka_file, t_start)
    if a.resume and os.path.isfile(os.path.join(run_dir, "pKa.csv")):
        log("2/3 membrane calculation: reusing %s" % os.path.join(run_dir, "pKa.csv"))
    else:
        log("2/3 membrane calculation (minutes to hours)")
        run_module("membrane_pka", argv + ["--np", "1", "--stage", "prepare"])
        run_module("membrane_pka", argv + ["--np", "1", "--stage", "maps"])
        np_ = choose_np(a.np, run_dir, a.base_fill, a.force_memory)
        fill = a.base_fill
        for _pass in (1, 2):
            while True:
                ec = os.path.join(run_dir, "last_exit_code")
                if os.path.exists(ec): os.remove(ec)
                try:
                    with Progress(run_dir, a.skip_arg_lys):
                        run_module("membrane_pka", argv + ["--np", str(np_), "--stage", "run"] + (["--oversubscribe"] if a.oversubscribe or np_ > cores() else []))
                    break
                except SystemExit:
                    code = open(ec).read().strip() if os.path.exists(ec) else ""
                    if code not in ("137", "-9"): raise                       # not an out-of-memory kill
                    if np_ > 1:
                        np_ = max(1, np_ // 2)
                        warn("the DelPhi stage was killed (out of memory). Retrying with %d MPI rank(s); finished sites are kept (energy checkpoints)" % np_)
                    elif fill < 97:
                        fill = min(97.0, fill + 5.0)
                        shutil.rmtree(os.path.join(run_dir, "energy_ckpt"), ignore_errors=True)       # the grid changes: old energies no longer match
                        argv = [x for x in argv]
                        k = [i for i, x in enumerate(argv) if x == "--set" and argv[i + 1].startswith("Base grid fill")]
                        if k: argv[k[0] + 1] = "Base grid fill (percent) : %g" % fill
                        warn("the DelPhi stage was killed (out of memory) with 1 rank. Retrying with a smaller box (base fill %g%%); the energy stage starts again" % fill)
                    else:
                        die("the DelPhi stage runs out of memory even with 1 rank and base fill %g%%: use a machine with more memory, or --skip-arg-lys to titrate fewer groups" % fill)
            if _pass == 2 or a.no_partner_check: break
            # judge each partner by pka-helper's own first-pass value (PROPKA-anchored blend), not DelPhi's raw value, which is unreliable for
            # buried His and retinal Schiff bases; a Lys carrying the retinal Schiff base always counts as charged
            finish_propka()
            _p1 = os.path.join(raw, "pass1_blend.csv"); run_module("blend_pka", blend_args(_p1))
            _neutral, _lost = neutral_partners(run_dir, a.ph, _p1, {"LYS%d%s" % (int(k[1][:4]), k[0].strip()) for k in schiff_all})
            if not _lost: break
            # a buried acid was spared the low-dielectric zone because of a 'partner' that the calculation itself finds neutral: no salt
            # bridge after all. Rebuild the map without those partners and compute the energies again (pass 2)
            log("partner check: %s spared the buried-acid zone only by %s, which come out neutral at pH %g; recomputing without them"
                % (", ".join(sorted(_lost)), ", ".join(sorted(_neutral)), a.ph))
            open(os.path.join(run_dir, "unpartnered_acids.txt"), "w").write("\n".join(sorted(_lost)) + "\n")
            argv = argv + ["--solid-neutral-partners", ",".join(sorted(_neutral))]
            shutil.copy(os.path.join(run_dir, "pKa.csv"), os.path.join(run_dir, "pKa_pass1.csv"))
            run_module("membrane_pka", argv + ["--np", "1", "--stage", "maps"])
            shutil.rmtree(os.path.join(run_dir, "energy_ckpt"), ignore_errors=True)
    if a.carboxyl_tautomers != "off" and not os.path.isfile(os.path.join(run_dir, "energies_fixed_proton.txt")):
        import tautomer
        mpirun = shutil.which("mpirun") or shutil.which("mpiexec"); nrk = a.np or 1
        try: nrk = np_
        except NameError: pass
        pre = []
        if nrk > 1 and mpirun:
            pre = [mpirun] + (["--allow-run-as-root"] if hasattr(os, "geteuid") and os.geteuid() == 0 else []) + (["--oversubscribe"] if a.oversubscribe or nrk > cores() else []) + ["-np", str(nrk)]
        for line in tautomer.run(run_dir, a.carboxyl_tautomers, pre, delphi, log=log, partner_kt=0.5 if a.tautomer_partners else 0.0):
            log(line)
    if not a.keep_maps:      # intermediates of the map step; nothing after the energy stage reads them (background.dx is kept)
        for _f in ("protein.dx", "membrane.dx"):
            try: os.remove(os.path.join(run_dir, _f))
            except OSError: pass
    # 3. blend
    finish_propka()
    log("3/3 blend")
    bl = blend_args(os.path.join(res, "pka_helper.csv"))
    run_module("blend_pka", bl)
    add_structure_check(a.pdb, run_dir, res, het_plan)
    try:
        import group_pka
        if os.path.isfile(os.path.join(run_dir, "titra.csv")):
            open(os.path.join(res, "clusters.txt"), "w").write(group_pka.render_clusters(run_dir) + "\n")
    except Exception as e:
        log("cluster report skipped (%s)" % e)
    rows = read_result(os.path.join(res, "pka_helper.csv"))
    for _k in [k for k in schiff_all if k not in schiff]:
        _r = [r for r in rows if r["residue"] == "LYS%d" % int(_k[1]) and r.get("chain", "") == _k[0]]
        if _r and _r[0]["_b"] < 10.0:
            warn("Lys%d %s carries the retinal Schiff base but comes out at pKa %.2f (partly neutral); measured Schiff-base pKas are above 12, so "
                 "its counterions see too little positive charge. Rerun with --hold-schiff-base to hold it charged" % (int(_k[1]), _k[0], _r[0]["_b"]))
    with open(os.path.join(res, "pkas.txt"), "w") as fo:
        fo.write("pka-helper %s   input: %s\n\n" % (__version__, os.path.basename(a.pdb)) + render_summary(rows, a.ph, "all") + "\n")
    with open(os.path.join(res, "comparison_with_propka.txt"), "w") as fo: fo.write(render_comparison(rows, a.ph) + "\n")
    with open(os.path.join(res, "uncertain_at_pH7.txt"), "w") as fo: fo.write(render_flips(rows, a.ph) + "\n")
    print("\n" + render_summary(rows, a.ph, a.show))
    el = time.time() - t_start
    end_text = ("\nResults in %s\n  results/pkas.txt                    every group: PROPKA, DelPhi, pKa, state, effect of the blend limits\n  results/comparison_with_propka.txt  side by side with PROPKA, biggest differences first\n  results/uncertain_at_pH7.txt        groups whose pH call depends on PROPKA vs DelPhi\n  results/structure_check.txt         input-structure problems that make a pKa unreliable\n  results/clusters.txt                coupled groups: stepwise (macroscopic) pKas to compare with stepwise measurements\n  results/pka_helper.csv              the same as a table for scripts\n  raw/                                DelPhiPKa calculation (raw/delphipka), PROPKA output, log.txt%s\n(total time %d min %d s)"
          % (out, " (details in log.txt; -v shows them live)" if LOGFILE else "", el // 60, el % 60))
    if getattr(a, "_defer_end", False): a._end_text = end_text      # printed by the caller after its own report
    else: print(end_text)
    return 0


def add_structure_check(pdb, run_dir, res, kept=()):
    """results/structure_check.txt + a 'structure_warning' column in pka_helper.csv (codes; the text file has the explanations)."""
    try:
        from structure_check import check, render
        ws = check(pdb, run_dir, kept)
    except Exception as e:
        log("structure check failed (%s); results are not affected" % e); return
    open(os.path.join(res, "structure_check.txt"), "w").write(render(ws) + "\n")
    per = {}
    for k, code, _ in ws:
        if k: per.setdefault((k[0], k[1], k[2]), []).append(code)
    p = os.path.join(res, "pka_helper.csv"); rows = list(csv.DictReader(open(p)))
    if not rows: return
    import re
    for r in rows:
        m = re.match(r"([A-Z]{3})(\d+)", r["residue"]); key = (m.group(1), int(m.group(2)), r.get("chain", "")) if m else None
        r["structure_warning"] = " ".join(dict.fromkeys(per.get(key, [])))
    with open(p, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n = {c: sum(1 for x in ws if x[1] == c) for c in ("protonated-geometry", "deep-lipid", "missing-atoms")}
    if any(n.values()):
        warn("structure checks: %s - see results/structure_check.txt" % ", ".join("%d %s" % (v, k) for k, v in n.items() if v))


AA3 = {"D": "ASP", "E": "GLU", "H": "HIS", "K": "LYS", "R": "ARG", "C": "CYS", "Y": "TYR"}


def read_result(path):
    rows = []
    for r in csv.DictReader(open(path)):
        try:
            r["_p"] = float(r["PROPKA"]); r["_b"] = float(r["blended"])
        except (KeyError, ValueError):
            continue
        import re
        m = re.match(r"([A-Z]{3})(\d+)", r["residue"]); r["_type"], r["_num"] = (m.group(1), int(m.group(2))) if m else (r["residue"], 0)
        rows.append(r)
    return rows


def sym(r):
    s = "P" if r["state_at_pH7"].startswith("proton") else "D"
    return s + ("?" if r.get("confidence") == "uncertain" else "")


def sig_txt(r):
    s = r.get("sigma", ""); return ("±" + s) if s else ""


def pp_txt(r):
    p = r.get("P_protonated", "")
    if not p: return ""
    v = float(p); return "<0.01" if v < 0.01 else ">0.99" if v > 0.99 else "%.2f" % v


def limits_txt(r):
    """The 'limits varied' cell: 'lo - hi' (+ ' !' if the pH call flips inside), or 'unaffected' when the range has no width."""
    lo, hi = r.get("param_lo", ""), r.get("param_hi", "")
    if lo == hi: return "unaffected"
    return "%s - %s%s" % (lo, hi, " !" if r.get("param_flips_call") == "1" else "")


def render_summary(rows, ph, mode="notable", lipid=None):
    """Plain-text summary: groups that matter, one block per residue number, chains as columns."""
    lipid = lipid or set()
    out = []
    n = len(rows); unc = sum(1 for r in rows if r.get("confidence") == "uncertain")
    out.append("pH %.1f: %d titratable groups, %d confident, %d uncertain (within 0.5 pK of pH or predictors disagree)." % (ph, n, n - unc, unc))
    out.append("'±' = estimated error (1 sigma, pK): about 2 in 3 true values lie within it and 19 in 20 within twice it, on the benchmark "
               "(tested on protein families left out of the fit). It grows for buried groups, His, membrane runs and when PROPKA and DelPhi "
               "disagree. 'P(prot)' = probability that the group is protonated at pH %.1f from that error; near 0.5 the call is open. "
               "'limits varied' = lowest - highest pKa when pka-helper's own limits on the DelPhi shift are varied ('unaffected' = the same "
               "with every setting); '!' = the pH %.1f call flips within that range." % (ph, ph))
    out.append("State: P = protonated, D = deprotonated, ? = uncertain.   shift = pKa - PROPKA.   'check' = a structure warning (results/structure_check.txt).")
    if mode == "none": return "\n".join(out)
    def notable(r):      # PROPKA and pka-helper put the group on opposite sides of the pH (the other uncertain groups are in uncertain_at_pH7.txt)
        return (r["_b"] > ph) != (r["_p"] > ph)
    if mode == "all": sel = rows
    else:
        keys = {(r["_type"], r["_num"]) for r in rows if notable(r)}
        sel = [r for r in rows if (r["_type"], r["_num"]) in keys and (notable(r) or sum(1 for q in rows if (q["_type"], q["_num"]) == (r["_type"], r["_num"])) > 1)]   # a residue number with several chains shows every chain
    groups = {}
    for r in sel: groups.setdefault((r["_type"], r["_num"]), []).append(r)
    if not groups: out.append("\nPROPKA and pka-helper agree on the protonation state of every group at pH %.1f (all values: results/pkas.txt)." % ph)
    else:
        out.append("\n%s" % ("All groups" if mode == "all" else "Groups where PROPKA and pka-helper disagree on the protonation state at pH %.1f" % ph))
        single = [g for g in sorted(groups) if len(groups[g]) == 1]
        multi = [g for g in sorted(groups) if len(groups[g]) > 1]
        if single:
            out.append("\n  %-9s %-5s %7s %7s %7s %6s %7s  %-5s %-14s %s" % ("residue", "chain", "PROPKA", "DelPhi", "pKa", "±", "P(prot)", "state", "limits varied", "note"))
            for g in single:
                r = groups[g][0]
                out.append("  %-9s %-5s %7.2f %7s %7.2f %6s %7s  %-5s %-14s %s" % (r["residue"], r.get("chain", ""), r["_p"], r["DelPhiPKa"], r["_b"], sig_txt(r), pp_txt(r), sym(r), limits_txt(r), (("check: " + r["structure_warning"] + "; ") if r.get("structure_warning") else "") + ("lipid-facing: larger cap" if "lipid" in r.get("note", "") else r.get("note", "")[:40])))
        for g in multi:
            rs = sorted(groups[g], key=lambda r: r.get("chain", ""))
            out.append("\n  %s%d in %d chains" % (g[0], g[1], len(rs)))
            out.append("    chain    " + " ".join("%6s" % r.get("chain", "") for r in rs))
            out.append("    PROPKA   " + " ".join("%6.2f" % r["_p"] for r in rs))
            out.append("    pKa      " + " ".join("%6.2f" % r["_b"] for r in rs))
            if any(r.get("sigma") for r in rs):
                out.append("    ±        " + " ".join("%6s" % sig_txt(r) for r in rs))
                out.append("    P(prot)  " + " ".join("%6s" % pp_txt(r) for r in rs))
            out.append("    state    " + " ".join("%6s" % sym(r) for r in rs))
            if any("lipid" in r.get("note", "") for r in rs):
                out.append("    lipid    " + " ".join("%6s" % ("yes" if "lipid" in r.get("note", "") else "") for r in rs))
            if any(r.get("structure_warning") for r in rs):
                out.append("    check    " + " ".join("%6s" % {"protonated-geometry": "H-geom", "deep-lipid": "deep", "missing-atoms": "atoms", "saturated": "sat"}.get(((r.get("structure_warning") or "").split() or [""])[0], "") for r in rs)
                           + "   (see results/structure_check.txt)")
        if mode != "all":
            nh = len(rows) - len(sel)
            if nh > 0:
                nu = sum(1 for r in rows if r not in sel and r.get("confidence") == "uncertain")
                out.append("\n%d other groups: PROPKA and pka-helper agree at pH %.1f (all values: results/pkas.txt)%s." % (nh, ph,
                           "; %d of them %s close to the pH or %s a structure warning (results/uncertain_at_pH7.txt)" % (nu, "is" if nu == 1 else "are", "has" if nu == 1 else "have") if nu else ""))
    return "\n".join(out)


def render_comparison(rows, ph):
    """PROPKA against the final pKa, biggest differences first."""
    d = [r["_b"] - r["_p"] for r in rows]
    flips = [r for r in rows if (r["_b"] > ph) != (r["_p"] > ph)]
    out = ["Comparison with PROPKA (pH %.1f)" % ph, "",
           "%d groups; mean |pKa - PROPKA| = %.2f pK; %d groups move by >= 1 pK; the pH %.1f call differs for %d." % (len(rows), sum(abs(x) for x in d) / max(len(d), 1), sum(1 for x in d if abs(x) >= 1), ph, len(flips)), "",
           "  %-9s %-5s %7s %7s %7s  %-12s %-12s %s" % ("residue", "chain", "PROPKA", "pKa", "diff", "PROPKA says", "pka-helper", "")]
    for r in sorted(rows, key=lambda r: -abs(r["_b"] - r["_p"])):
        sp = "protonated" if r["_p"] > ph else "deprotonated"; sb = "protonated" if r["_b"] > ph else "deprotonated"
        out.append("  %-9s %-5s %7.2f %7.2f %+7.2f  %-12s %-12s %s" % (r["residue"], r.get("chain", ""), r["_p"], r["_b"], r["_b"] - r["_p"], sp, sb, "<-- call differs" if sp != sb else ""))
    return "\n".join(out)


def fast_mode(a, out, raw, res, run_dir, argv, pka_file, t_start):
    """--fast: PROPKA + the membrane maps (no DelPhi energy calculation). Flags the lipid-facing acids; pKa values are PROPKA's, unchanged."""
    import blend_pka as B
    from lipid_auto import lipid_facing, lipid_facing_bases
    if not (os.path.isfile(os.path.join(run_dir, "background.dx")) and a.resume):
        log("2/3 membrane maps (no DelPhi energy calculation)")
        run_module("membrane_pka", argv + ["--np", "1", "--stage", "prepare"])
        run_module("membrane_pka", argv + ["--np", "1", "--stage", "maps"])
    log("3/3 finding lipid-facing groups")
    env = {}
    lipid_facing(run_dir, report=env)
    flagged = lipid_facing(run_dir)
    flagged_b = lipid_facing_bases(run_dir, report=env)     # lipid-facing lysines (PROPKA may be too HIGH for these)
    flagged = dict(flagged); flagged.update(flagged_b)
    P, BUR = B.read_propka(pka_file)
    ph = a.ph
    rows = []
    for k in sorted(P, key=lambda k: (k[2], k[1])):
        if a.chain and k[2] != a.chain: continue
        lf = env.get(k)
        rows.append(dict(rt=k[0], num=k[1], chain=k[2], pka=P[k], bur=BUR.get(k), lip=(lf[0] if lf else None), ncat=(lf[1] if lf else None), flag=k in flagged))
    fl = [r for r in rows if r["flag"]]
    L = ["pka-helper --fast   (PROPKA values; membrane maps used only to find lipid-facing acids)", "",
         "Nothing here is a DelPhi result: the pKa column is PROPKA's, unchanged.",
         "LIPID-FACING = Asp/Glu with >= 30% of its 3-8 A shell in the membrane lipid and no Arg/Lys/His within 8 A; Lys likewise with no Asp/Glu within 6 A (for a Lys PROPKA may be too HIGH).",
         "PROPKA does not know about the membrane. For such groups the membrane calculation pushes the pKa up, which matched the reference for c-ring Glu58 and",
         "VDAC1 Glu73, so PROPKA's value may be too low: run without --fast to get a number. The detector is not always right: it flagged 5 of the 6 lipid-facing",
         "c-ring Glu58 and VDAC1 Glu73 correctly, but also Hv1 Glu153 and NhaA Asp163 (1ZCD), where the reference stayed low. It is a pointer, not a verdict.", "",
         "%d groups, %d lipid-facing acid(s) or lysine(s)." % (len(rows), len(fl)), ""]
    if fl:
        L.append("  %-9s %-5s %7s %-12s %9s %8s %s" % ("residue", "chain", "PROPKA", "PROPKA says", "burial", "lipid", "partners<6A"))
        for r in fl:
            L.append("  %-9s %-5s %7.2f %-12s %8s%% %7d%% %d" % ("%s%d" % (r["rt"], r["num"]), r["chain"], r["pka"], "protonated" if r["pka"] > ph else "deprotonated",
                     "%d" % round(100 * r["bur"]) if r["bur"] is not None else "?", round(100 * r["lip"]), r["ncat"]))
    else:
        L.append("No lipid-facing acid found: the full calculation would be unlikely to move any acid far from PROPKA.")
    open(os.path.join(res, "lipid_facing.txt"), "w").write("\n".join(L) + "\n")
    T = ["pka-helper --fast: PROPKA pKa values (pH %.1f), lipid-facing acids marked" % ph, "", "  %-9s %-5s %7s %-12s %7s  %s" % ("residue", "chain", "PROPKA", "state", "burial", "")]
    for r in rows:
        T.append("  %-9s %-5s %7.2f %-12s %6s%%  %s" % ("%s%d" % (r["rt"], r["num"]), r["chain"], r["pka"], "protonated" if r["pka"] > ph else "deprotonated",
                 "%d" % round(100 * r["bur"]) if r["bur"] is not None else "?", ("LIPID-FACING: PROPKA may be too high" if r["rt"] == "LYS" else "LIPID-FACING: PROPKA may be too low") if r["flag"] else ""))
    open(os.path.join(res, "pkas.txt"), "w").write("\n".join(T) + "\n")
    with open(os.path.join(res, "pka_helper_fast.csv"), "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["residue", "chain", "PROPKA", "state_at_pH%g" % ph, "propka_burial", "lipid_fraction", "cations_within_6A", "lipid_facing"])
        for r in rows: w.writerow(["%s%d" % (r["rt"], r["num"]), r["chain"], "%.2f" % r["pka"], "protonated" if r["pka"] > ph else "deprotonated", "" if r["bur"] is None else "%.2f" % r["bur"], "" if r["lip"] is None else "%.2f" % r["lip"], "" if r["ncat"] is None else r["ncat"], int(r["flag"])])
    print("\n" + "\n".join(L))
    el = time.time() - t_start
    print("\nResults in %s\n  results/lipid_facing.txt   the flagged acids\n  results/pkas.txt           PROPKA pKa for every group, flagged ones marked\n  results/pka_helper_fast.csv\n(total time %d min %d s)" % (out, el // 60, el % 60))
    return 0


def render_flips(rows, ph):
    """Groups whose pH call depends on whether you trust PROPKA or DelPhi, sorted into three kinds so that each explanation is printed once."""
    import re
    def dv(r):
        try: return float(r["DelPhiPKa"].rstrip("*"))
        except ValueError: return None
    def short_env(e):
        e = re.sub(r"PROPKA burial (\d+)%", r"burial \1%", e)
        e = re.sub(r"(\d+)% of the 3-8 A shell is membrane lipid", r"lipid \1%", e)
        e = re.sub(r"(\d+) Arg/Lys/His within 8 A", r"cations<8A \1", e)
        e = re.sub(r"(\d+) Asp/Glu within 6 A", r"anions<6A \1", e)
        return re.sub(r";? ?DelPhi value saturated \(limit of its range\)", "", e).strip("; ")
    sec = {"lipid": [], "moved": [], "kept": []}
    for r in rows:
        d = dv(r)
        if d is None: continue
        fb, fp, fd = r["_b"] > ph, r["_p"] > ph, d > ph
        if len({fb, fp, fd}) == 1: continue
        r["_d"] = d
        if r.get("likely_more_reliable") == "pka-helper": sec["lipid"].append(r)
        elif fb == fp: sec["kept"].append(r)
        else: sec["moved"].append(r)
    n = sum(len(v) for v in sec.values())
    out = ["Groups whose pH %.1f call differs between PROPKA, the raw DelPhi value and the final pKa" % ph, "",
           "%d of %d groups, in three kinds. * = DelPhi value saturated (its limits are 0 and 16)." % (n, len(rows)),
           "'more reliable' is a rule-of-thumb read-out of the 69-site benchmark (heuristic, not a validated classifier)."]
    if not n: return "\n".join(out + ["", "None."])
    def table(rs):
        o = ["", "  %-9s %-5s %7s %7s %7s  %-6s  %s" % ("residue", "chain", "PROPKA", "DelPhi", "final", "final", "environment")]
        for r in sorted(rs, key=lambda r: (r["_type"], r["_num"], r.get("chain", ""))):
            o.append("  %-9s %-5s %7.2f %7s %7.2f  %-6s  %s" % (r["residue"], r.get("chain", ""), r["_p"], r["DelPhiPKa"], r["_b"],
                     ("P" if r["_b"] > ph else "D") + ("?" if r.get("confidence") == "uncertain" else ""), short_env(r.get("environment", ""))))
        return o
    if sec["lipid"]:
        rs = sec["lipid"]
        out += ["", "1. Lipid-facing acids and lysines: the final call follows DelPhi (%d groups)" % len(rs)]
        for why in dict.fromkeys(r.get("reliability_reason", "") for r in rs):
            out.append("   Reason (%s): %s" % ("Lys" if any(r["_type"] == "LYS" and r.get("reliability_reason", "") == why for r in rs) else "Asp/Glu", why))
        out += ["   pka-helper is probably the more reliable predictor here. The cap on DelPhi's shift (upward for acids, downward for Lys) was lifted for these groups."] + table(rs)
    if sec["moved"]:
        rs = sec["moved"]
        out += ["", "2. DelPhi moved the call and nothing says which is right (%d groups) -- the ones to look at" % len(rs),
                "   The final pKa follows DelPhi. Environment and benchmark give no clear rule (details per group below)."] + table(rs)
        seen = {}
        for r in rs: seen.setdefault(r.get("reliability_reason", ""), []).append("%s:%s" % (r["residue"], r.get("chain", "")))
        out.append("")
        for why, who in seen.items(): out.append("   %s: %s" % (", ".join(who), why))
    if sec["kept"]:
        rs = sec["kept"]
        out += ["", "3. DelPhi disagreed but the bound kept PROPKA's call (%d groups) -- no action needed" % len(rs),
                "   The raw DelPhi value is on the other side of pH %.1f, usually because it failed (saturated or an extreme value);" % ph,
                "   the final pKa stays close to PROPKA and the call is not in doubt."] + table(rs)
    return "\n".join(out) + "\n"


def cmd_pass(mod):
    def f(a):
        run_module(mod, a.rest)
        return 0
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pka-helper", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version="pka-helper " + __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor", help="check the installation"); d.add_argument("--delphi-exe"); d.set_defaults(f=cmd_doctor)
    r = sub.add_parser("run", help="PDB -> final pKa table (PROPKA + membrane DelPhi + bounded blend)")
    r.add_argument("pdb", help="PDB oriented with the membrane normal along z (OPM/PPM); dummy atoms optional")
    r.add_argument("--out", help="output directory (default: <pdb stem>_pka)")
    r.add_argument("--delphi-exe", help="delphiPKa binary (default: $PKA_HELPER_DELPHI, then PATH)")
    r.add_argument("--np", type=int, default=None, help="MPI ranks (default: as many as the cores and the free RAM allow, from an estimate of the memory use)")
    r.add_argument("--force-memory", action="store_true", help="ignore the memory check (the run may be killed if memory runs out)")
    r.add_argument("--oversubscribe", action="store_true", help="allow more ranks than cores")
    r.add_argument("--ensemble", action="store_true", help="run every MODEL of the file (NMR ensembles, MD snapshots saved as models) and report the mean and spread "
                   "of every pKa (results/ensemble.txt); results/pka_helper.csv then holds the mean. Cost: one full run per model")
    r.add_argument("--ensemble-max", type=int, default=0, help="with --ensemble: use at most this many models (default all)")
    r.add_argument("--membrane-scan", action="store_true", help="also run with the hydrophobic core 3 A thinner and 3 A thicker and report how much each pKa "
                   "depends on the membrane position (columns membrane_lo / membrane_hi, results/membrane_scan.txt). Cost: three full runs")
    r.add_argument("--carboxyl-tautomers", choices=["selected", "all-buried", "off"], default="selected",
                   help="where the proton of a neutral Asp/Glu sits: DelPhiPKa puts it on one fixed oxygen. 'selected' (default) recomputes buried acids whose "
                   "H-bond partner is at the other oxygen with the proton there and keeps the lower-energy placement (a few %% extra time); 'all-buried' does "
                   "it for every buried acid; 'off' = fixed placement (the older behaviour)")
    r.add_argument("--tautomer-partners", action="store_true", help="with --carboxyl-tautomers: also recompute groups coupled to a switched acid by > 0.5 kT, "
                   "so their pair energies are averaged over both directions as in a full run (more accurate, about 3x the extra time)")
    r.add_argument("--hetero", choices=["keep", "drop"], default="keep",
                   help="HETATM groups (ligands, cofactors, bound lipids, ions, modified residues). 'keep' (default): modified residues with a "
                   "standard charge are read as that residue, groups with a known charge (ions, phospho-residues, charged lipids, retinal Schiff "
                   "base) are kept as fixed charges, other ligands as uncharged volume; ions (see --keep-ions), crystallisation additives and detergents are left out. "
                   "'drop': leave every HETATM group out (their space becomes water)")
    r.add_argument("--drop-het", action="append", default=[], metavar="RES[,RES]", help="leave these HETATM residue names out (repeatable)")
    r.add_argument("--hold-schiff-base", action="store_true", help="hold a lysine bound to a separate retinal (RET) charged (+1, the protonated "
                   "Schiff base; measured pKas are above 12) instead of titrating it like any lysine. Default: titrated, with a warning if it comes out "
                   "below 10 (in bacteriorhodopsin DelPhi already gives 12-13; holding it changed no counterion pKa by more than 0.4)")
    r.add_argument("--keep-waters", action="store_true", help=argparse.SUPPRESS)
    r.add_argument("--uncap-unpartnered", action="store_true", help=argparse.SUPPRESS)
    r.add_argument("--no-partner-check", action="store_true", help="do not re-check salt-bridge partners: a buried Asp/Glu next to a Lys/Arg/His is "
                   "spared the low-dielectric zone even when the calculation finds that partner neutral")
    r.add_argument("--keep-ions", action="store_true", help="keep metal and halide ions from the structure with their usual charge (default: left out, "
                   "because crystal ions are often absent where pKas are measured; or keep one kind with --cofactor ZN:+2)")
    r.add_argument("--cofactor", action="append", default=[], metavar="RES[:CHARGE]",
                   help="keep this HETATM residue in the DelPhi calculation as a fixed charge and low-dielectric volume (repeat for several), e.g. "
                   "--cofactor RET (retinal, neutral; the Schiff-base Lys is titrated as usual) or --cofactor ZN:+2. Overrides the --hetero choice for that residue name")
    r.add_argument("--hbond-term", action="store_true", help=argparse.SUPPRESS)
    r.add_argument("--keep-maps", action="store_true", help="keep the intermediate protein.dx / membrane.dx maps (deleted after the energy stage by default; they can take hundreds of MB)")
    r.add_argument("--no-membrane", action="store_true", help="compute the protein in water only (soluble proteins); same settings otherwise")
    r.add_argument("--membrane", choices=["auto", "flat", "curved"], default="auto", help="flat slab or fitted curved membrane (default: auto from the dummy atoms)")
    r.add_argument("--axis", choices=list("xyz"), default="z", help="membrane normal when there are no dummy atoms (default z)")
    r.add_argument("--center", type=float, help="membrane mid-plane coordinate along the axis (default: from the dummy atoms, else 0)")
    r.add_argument("--core-thickness", type=float, help="hydrophobic core thickness in A (default: from the dummy atoms, else 30)")
    r.add_argument("--membrane-shape", help="reuse a membrane geometry from an earlier run (its raw/delphipka/membrane_shape.json). The structure must be in the SAME coordinate frame as the one that gave the shape")
    r.add_argument("--lipids", help="PDB with lipids (phosphates) to fit the curved membrane instead of the dummy atoms")
    r.add_argument("--skip-arg-lys", action="store_true", help="do not titrate Lys and Arg: they stay charged and act as fixed charges, and their pKa is taken from PROPKA. About half the groups, so the energy stage takes about half the time. Acids and His close to a Lys/Arg that could itself lose its charge (buried, in the membrane) can come out several pK too low, so use it for a quick look or when Lys/Arg are not of interest")
    r.add_argument("--base-fill", type=float, default=90.0, help="percent of the DelPhi base box filled by the protein (default 90; DelPhiPKa itself uses 70). Larger = smaller box, less memory, same pKas in tests")
    r.add_argument("--lipid-eps", type=float, default=2.0, help="dielectric of the membrane core (default 2). 10 reproduces measured / constant-pH MD pKas of single His/Lys/Asp/Glu in "
                   "transmembrane peptides (0.8 pK vs 4.1 at 2) and removes the saturated DelPhi values of lipid-facing groups, and switches to blend rules "
                   "refitted for it. With every benchmark protein held out in turn the two settings tie (1.18 vs 1.21 pK); 10 is better on lipid-exposed "
                   "peptide groups, 2 on BM2 His27 and EmrE Glu14")
    r.add_argument("--membrane-arg", nargs=argparse.REMAINDER, default=[], help="everything after this goes to the membrane stage unchanged")
    r.add_argument("-v", "--verbose", action="store_true", help="show the full output of every step (default: only progress and the summary; details go to log.txt)")
    r.add_argument("--show", choices=["notable", "all", "none"], default="notable", help="which groups the summary lists (default: those where PROPKA and pka-helper disagree on the protonation state; the files always have all)")
    r.add_argument("--fast", action="store_true", help="no DelPhi energy calculation: PROPKA plus the membrane maps, only to flag the lipid-facing acids (minutes instead of hours; pKa values are PROPKA's)")
    r.add_argument("--resume", action="store_true", help="reuse an existing membrane pKa.csv")
    r.add_argument("--ph", type=float, default=7.0, help="pH for the protonation call (default 7)")
    r.add_argument("--chain", default="", help="only this chain")
    r.add_argument("--acid-cap", type=float, default=None, help="largest upward DelPhi shift for an Asp/Glu, in pK (default 0.5; 2.0 with --lipid-eps 10)")
    r.add_argument("--no-lipid-auto", action="store_true", help="do not exempt lipid-facing acids from the cap")
    r.add_argument("--lipid-cap", type=float, default=None, help="upward cap for lipid-facing acids (default 3.0; 5.0 with --lipid-eps 10)")
    r.add_argument("--acid-up-scale", type=float, default=None, help="fraction of DelPhi's upward push on an Asp/Glu that is used, before the --acid-cap (default 0.3; 0 = the full push)")
    r.add_argument("--his-down", type=float, default=None, help="largest downward DelPhi shift for a His (default 1.5; 0 with --lipid-eps 10)")
    r.add_argument("--his-lip-cap", type=float, default=None, help="largest downward shift for a His deep in the membrane core, scaled by depth (default 0 = off; 2.0 with --lipid-eps 10)")
    r.add_argument("--lys-down-c0", type=float, default=0.0, help="constant part of the largest downward shift for a non-lipid Lys (default 0; 1.0 = the older rule)")
    r.add_argument("--lipid-cap-base", type=float, default=4.0, help="largest downward shift for a lipid-facing Lys deep in the membrane core (default 4.0)")
    r.add_argument("--lipid-depth-min", type=float, default=4.0, help="a lipid-facing Lys less than this many A inside the core gets no extra downward cap (default 4.0)")
    r.add_argument("--lipid-depth-full", type=float, default=10.0, help="depth from which the full --lipid-cap-base applies; linear in between (default 10.0)")
    r.add_argument("--no-lipid-depth", action="store_true", help="same lipid cap at every depth")
    r.add_argument("--lipid-depth-acids", action="store_true", help="also scale the upward cap of lipid-facing acids by depth (off by default)")
    r.add_argument("--lipid-acids", default="", help="extra lipid-facing acids, e.g. GLU58:K,GLU73")
    r.add_argument("--no-his-rule", action="store_true", help="switch the histidine rule off")
    r.add_argument("--alt", action="append", default=[], help="another predictor's table (for the confidence flag); repeatable")
    r.set_defaults(f=cmd_run)
    for name, mod, h in (("blend", "blend_pka", "blend an existing DelPhi table with a PROPKA file"),
                         ("membrane", "membrane_pka", "membrane and DelPhi stages only"),
                         ("shape", "membrane_shape", "fit or define a curved membrane")):
        p = sub.add_parser(name, help=h + " (options as in the underlying tool; add -h)", add_help=False)
        p.add_argument("rest", nargs=argparse.REMAINDER); p.set_defaults(f=cmd_pass(mod))
    a = ap.parse_args(argv)
    a._argv = list(sys.argv[1:] if argv is None else argv)
    a._pdb_arg = getattr(a, "pdb", None)
    sys.exit(a.f(a))


if __name__ == "__main__":
    main()
