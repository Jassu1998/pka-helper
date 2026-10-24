#!/usr/bin/env python3
"""
membrane_pka.py - one command from an oriented PDB to a membrane pKa table with DelPhiPKa.

    python3 tools/membrane_pka.py protein.pdb --np 4

Steps (each one is skipped if --stage says so):
  1. prepare : copy the PDB into <out>/, run DelPhiPKa in "protonation only" mode to get the PQR (hydrogens, radii),
               drop HETATM lines from it (OPM 'DUM' membrane markers, waters, ...).
  2. maps    : protein.dx (Gaussian, only used to find pore/cavity water)  ->  membrane.dx (slab)  ->
               background.dx  (epsmap.py combine --background-only, with lumen water).
  3. run     : write run.prm (Custom dielectric map type BACKGROUND), run mpirun DelPhiPKa, print pKa.csv.

The membrane normal is z and its mid-plane is z = 0 unless the PDB has OPM dummy atoms (residue name DUM): then the slab centre and
the hydrophobic thickness are taken from their z range (override with --center / --core-thickness).

Needs: python3 + numpy, the compiled delphiPKa binary, mpirun (for --np > 1).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_root():
    """The delphipka-custom directory (holds param/ and delphiPKa), wherever this script was copied to."""
    for d in (HERE, os.path.dirname(HERE), os.getcwd()):
        if os.path.isdir(os.path.join(d, "param")):
            return d
    return os.path.dirname(HERE)


def _find_tool(name):
    """epsmap.py / gaussian_to_dx.py: next to this script, in ./tools, or in <root>/tools."""
    for d in (HERE, os.path.join(HERE, "tools"), os.path.join(ROOT, "tools"), os.path.join(os.getcwd(), "tools")):
        if os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    die("cannot find %s (looked next to the script and in tools/); run from the delphipka-custom directory" % name)


ROOT = _find_root()


def log(msg):
    print("[membrane_pka] " + msg, flush=True)


def die(msg):
    print("[membrane_pka] ERROR: " + msg, file=sys.stderr)
    sys.exit(1)


def run(cmd, cwd, logfile=None):
    log("$ " + " ".join(cmd) + ("   (cwd %s)" % cwd if cwd else ""))
    if logfile:
        with open(logfile, "w") as fh:
            r = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        try:      # pka-helper reads this to tell an out-of-memory kill (137 = SIGKILL) from other failures
            open(os.path.join(cwd or ".", "last_exit_code"), "w").write("%d\n" % r.returncode)
        except OSError:
            pass
        die("command failed (exit %d)%s" % (r.returncode, (", see " + logfile) if logfile else ""))


def pdb_info(path, axis):
    """Return (dummy_min, dummy_max or None, n_atoms, atom coord min/max along axis)."""
    col = {"x": (30, 38), "y": (38, 46), "z": (46, 54)}[axis]
    dum, atoms = [], []
    for line in open(path):
        if line.startswith(("ATOM", "HETATM")):
            try:
                v = float(line[col[0]:col[1]])
            except ValueError:
                continue
            if line[17:20].strip() == "DUM":
                dum.append(v)
            elif line.startswith("ATOM"):
                atoms.append(v)
    if not atoms:
        die("no ATOM records with coordinates found in %s" % path)
    return (min(dum), max(dum)) if dum else None, (min(atoms), max(atoms)), len(atoms)


RUN_PRM = """\
PDB file name                      : {pdb}
Charge parameter                   : {param}/amber.crg
Radius parameter                   : {param}/parseres.siz
Topology parameter                 : {param}/topology

Remove HETATM                      : T
Remove water molecule              : T
HETATM in PQR format               : F

Do Protonation                     : T
Do Energy Calculation              : {energy}
Do pKa's  Calculation              : {pka}

Output PQR file (With Topology)    : T
Output PQR file (With pKa result)  : F
At given pH Value                  : 7.0

Gaussian surface                   : 1
Variance of Gaussian distribution  : {sigma}
Internal Dielectric                : {indi}
External Dielectric                : {exdi}
{custom}
Cluster Delimitation Threshold (A) : {cluster}
Kmean++ cluster number             : AUTO

Hydrogen of GLU Attached to Atom   : OE1
Hydrogen of ASP Attached to Atom   : OD1

pH Initial Value                   : {ph0}
pH End Value                       : {ph1}
pH Interval                        : {phs}
"""


def write_prm(path, a, pdbname, energy, pka, custom_map):
    custom = ""
    if custom_map:
        custom = ("Custom dielectric map              : %s\n"
                  "Custom dielectric map type         : BACKGROUND\n"
                  "Custom dielectric map outside      : EXTEND\n" % custom_map)
        if getattr(a, "_dist_map", None):
            custom += "Custom dielectric distance map     : %s\n" % a._dist_map
        if getattr(a, "_stack_file", None):
            custom += "Custom dielectric stack            : %s\n" % a._stack_file
    text = RUN_PRM.format(pdb=pdbname, param="param", energy=energy, pka=pka, sigma=a.sigma, indi=a.indi, exdi=a.exdi,
                          custom=custom, cluster=a.cluster, ph0=a.ph_start, ph1=a.ph_end, phs=a.ph_step)
    _crg, _siz = {"amber": ("amber.crg", "parseres.siz"), "charmm22": ("charmm22.crg", "charmm22.siz"), "gromos": ("gromos.crg", "gromos.siz"), "parse": ("parseres.crg", "parseres.siz")}[a.ff]
    text = text.replace("param/amber.crg", "param/" + _crg).replace("param/parseres.siz", "param/" + _siz)
    if not getattr(a, "no_checkpoint", False) and str(energy).upper().startswith("T"):
        text += "Energy checkpoint directory         : energy_ckpt\n"      # resume an interrupted energy stage by re-running the same command
    if getattr(a, "desolv_cap", 0.0) and a.desolv_cap > 0 and getattr(a, "protein_eps", None) is not None:
        text += "Custom dielectric desolvation cap : %.4f\n" % (a.desolv_cap * 2.302585)      # pK units -> kT
    for kv in a.set or []:
        if ":" not in kv:
            die("--set expects 'Key : value', got %r" % kv)
        key, val = [s.strip() for s in kv.split(":", 1)]
        pat = re.compile(r"^%s\s*:.*$" % re.escape(key), re.M)
        if pat.search(text):
            text = pat.sub("%-35s: %s" % (key, val), text)
        else:
            text += "%-35s: %s\n" % (key, val)
    open(path, "w").write(text)


def is_openmpi(mpirun):
    try:
        out = subprocess.run([mpirun, "--version"], capture_output=True, text=True, timeout=20)
        return "open mpi" in (out.stdout + out.stderr).lower() or "open-mpi" in (out.stdout + out.stderr).lower()
    except Exception:
        return False


def physical_cores():
    """Physical cores (what Open MPI uses as slots); falls back to the logical CPU count."""
    try:
        import re
        t = subprocess.run(["lscpu", "-p=CORE,SOCKET"], capture_output=True, text=True, timeout=20).stdout
        n = len({tuple(l.split(",")) for l in t.splitlines() if l and not l.startswith("#")})
        if n: return n
    except Exception:
        pass
    return os.cpu_count() or 1


def mpi_cmd(a, exe, prm):
    if a.np > 1:
        mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
        if not mpirun:
            die("--np %d needs mpirun/mpiexec on PATH (or use --np 1)" % a.np)
        cmd = [mpirun]
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            cmd.append("--allow-run-as-root")
        if a.oversubscribe or (is_openmpi(mpirun) and a.np > physical_cores()):
            # Open MPI counts PHYSICAL cores as slots by default, so --np 32 on a 16-core / 32-thread machine stops with
            # "not enough slots" (prte-rmaps-base:alloc-error) although 32 hardware threads exist.  --oversubscribe lifts that limit.
            cmd.append("--oversubscribe")
        cmd += ["-np", str(a.np), exe, prm]
        return cmd
    return [exe, prm]


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb", help="oriented PDB (membrane normal along --axis, e.g. from OPM/PPM)")
    ap.add_argument("--out", help="working directory (default: <pdb stem>_membrane)")
    ap.add_argument("--exe", default=os.environ.get("PKA_HELPER_DELPHI") or (os.path.join(os.path.dirname(HERE), "bin", "delphiPKa") if os.path.isfile(os.path.join(os.path.dirname(HERE), "bin", "delphiPKa")) else "") or shutil.which("delphiPKa") or os.path.join(ROOT, "delphiPKa"), help="delphiPKa binary (default: $PKA_HELPER_DELPHI, then delphiPKa on PATH)")
    ap.add_argument("--param-dir", default=os.environ.get("PKA_HELPER_PARAM") or (os.path.join(os.path.dirname(HERE), "param") if os.path.isdir(os.path.join(os.path.dirname(HERE), "param")) else os.path.join(ROOT, "param")), help="force-field parameter directory (default: $PKA_HELPER_PARAM, else the one installed with pka-helper)")
    ap.add_argument("--np", type=int, default=2, help="MPI ranks (default 2)")
    ap.add_argument("--oversubscribe", action="store_true", help="pass --oversubscribe to mpirun (more ranks than cores)")
    ap.add_argument("--no-report-clusters", action="store_true", help="skip the macroscopic pKa report for groups of coupled Asp/Glu (clusters_macro.csv)")
    ap.add_argument("--cluster-cutoff", type=float, default=5.0, help="carboxylate centroids closer than this (A) count as one coupled group (default 5)")
    ap.add_argument("--stage", choices=["all", "prepare", "maps", "run"], default="all",
                    help="'all' (default) = prepare, maps and run; or stop/start at one stage: the others reuse files in --out")
    # model
    g = ap.add_argument_group("dielectric model")
    g.add_argument("--no-fix-names", dest="fix_names", action="store_false", help="do not rename CHARMM-style atoms/residues (ILE CD, OT1/OT2, HSP, TIP3 ...) to the names DelPhiPKa knows")
    g.add_argument("--optimize-h", action="store_true", help="choose the carboxylate oxygen that carries the proton of the neutral Asp/Glu state with PDB2PQR's hydrogen-bond optimiser "
                   "(needs `pip install pdb2pqr`; see tools/hopt.py). Default: the proton is always on OD1/OE1")
    g.add_argument("--sigma", type=float, default=0.7, help="Variance of Gaussian distribution (default 0.7 = the program default; 1.0 buries more of the protein surface)")
    g.add_argument("--indi", type=float, default=None, help="Internal Dielectric (default: same as --protein-eps)")
    g.add_argument("--interior-depth", type=float, default=None, help="with --solid-interior: only protein voxels at least this deep (A) inside the probe-closed body are set to the protein dielectric (default 1.0 = almost the whole body; 2-3 leaves the surface layer, where exposed and pore-lining groups sit, to the Gaussian)")
    g.add_argument("--solid-buried-sasa", type=float, default=None, help="local solid interior: only around Asp/Glu whose solvent-accessible area is below this many A^2 (try 1.0); implies --solid-interior")
    g.add_argument("--solid-neutral-partners", default="", help="residues (HIS322A,...) not counted as salt-bridge partners (found neutral in a first pass)")
    g.add_argument("--solid-skip-cations", default="LYS,ARG,HIS", help="with --solid-skip-partnered: residues counted as salt-bridge partners (default LYS,ARG,HIS)")
    g.add_argument("--solid-skip-partnered", type=float, default=0.0, metavar="R", help="do not make a buried acid solid if a Lys/Arg/His (or Schiff base) nitrogen is within R A of it (bR Asp85/Asp212: try 5)")
    g.add_argument("--solid-spare-partners", type=float, default=0.0, metavar="R", help="return to the Gaussian the R A shell around charged partners (Lys/Arg/His) of buried acids, so the solid zone does not dry their neighbours (OmpF Glu117 fix: 5.0)")
    g.add_argument("--solid-spare-acids", action="store_true", help="with --solid-spare-partners, also spare non-flagged Asp/Glu carboxylates")
    g.add_argument("--solid-groups", default="acid", help="side chains tested for burial by --solid-buried-sasa: acid (Asp/Glu, default), base (Lys/Arg/His) or acid,base")
    g.add_argument("--solid-radius", type=float, default=6.0, help="radius of that neighbourhood, A (default 6)")
    g.add_argument("--solid-interior", action="store_true", help="EXPERIMENTAL: give the probe-closed protein interior the protein dielectric whatever its lipid/water label "
                   "(physically motivated, but on 3EMN it degrades agreement with PROPKA and the known Asp9 value; off by default)")
    g.add_argument("--waters", default=None, help="PDB/xyz file of crystal waters to keep as eps-80 sites in the background map (internal waters)")
    g.add_argument("--ff", default="amber", choices=["amber", "charmm22", "gromos", "parse"], help="charges + radii: amber = amber.crg + parseres.siz (default); charmm22 / gromos = matching .crg and .siz; parse = parseres.crg + parseres.siz")
    g.add_argument("--protein-eps", type=float, default=4.0, help="dielectric of the protein body, also where it is embedded in lipid (default 4.0; 0 = old behaviour: lipid eps in the membrane, Internal Dielectric 2.0). Sets Internal Dielectric too unless --indi is given")
    g.add_argument("--exdi", type=float, default=80.0, help="External Dielectric (default 80)")
    g.add_argument("--axis", choices="xyz", default="z", help="membrane normal (default z)")
    g.add_argument("--center", type=float, default=None, help="membrane mid-plane coordinate (default: from DUM atoms, else 0)")
    g.add_argument("--core-thickness", type=float, default=None, help="hydrophobic core thickness, A (default: from DUM atoms, else 30)")
    g.add_argument("--lipids", default=None, help="CURVED MEMBRANE: PDB with the lipids (phosphate atoms are used; CHARMM-GUI / MD snapshot, same frame as the protein). "
                   "The two leaflets are fitted with a plane / sphere / cylinder / polynomial surface (membrane_shape.py) and the dielectric layers follow the distance to it")
    g.add_argument("--membrane-shape", default=None, help="CURVED MEMBRANE: JSON written by membrane_shape.py (fit or --manual sphere/cylinder/plane), instead of --lipids")
    g.add_argument("--shape", default="auto", help="with --lipids: auto | plane | sphere | cylinder | poly2 | poly3 | poly4 (default auto)")
    g.add_argument("--lipid-atoms", default="P,P31,P8,P1", help="with --lipids: atom names marking a leaflet (default P)")
    g.add_argument("--lipid-resnames", default=None, help="with --lipids: restrict the markers to these residue names (default: every lipid-like residue)")
    g.add_argument("--opm-curved", action="store_true", help="CURVED MEMBRANE from the OPM dummy atoms of the input PDB: shorthand for --lipids <pdb> --lipid-resnames DUM --lipid-atoms '*' "
                   "--p-offset 0 (DUM atoms already sit on the hydrophobic boundary, so the fitted thickness is the core thickness). Use it when the DUM atoms are spread over a curved surface")
    g.add_argument("--lipid-within", type=float, default=None, help="with --lipids: only markers within this many A of the protein centre (e.g. 100)")
    g.add_argument("--p-offset", type=float, default=4.5, help="with --lipids: phosphate-to-core-edge distance, A; core thickness = P-P thickness - 2 x this (default 4.5)")
    g.add_argument("--head-thickness", type=float, default=0.0, help="head-group zone on each side, A (default 0 = none; with the matched-reference fix a zone such as 8 is safe, see CUSTOM_DIELECTRIC.md)")
    g.add_argument("--eps-core", type=float, default=2.0)
    g.add_argument("--eps-head", type=float, default=30.0)
    g.add_argument("--smooth", type=float, default=1.0, help="smoothing of the layer boundaries, A (default 1)")
    g.add_argument("--hydrate-radius", type=float, default=0.0, help="EXPERIMENTAL residue-exposure dependent slab edge: within this many A of the selected groups, lipid-labelled voxels next to atom-based water-accessible pore / funnel space become water (lumen eps). 0 = off. Try 8-10")
    g.add_argument("--hydrate-groups", default="his", help="with --hydrate-radius: comma list of his, acid, lys, arg (default his)")
    g.add_argument("--hydrate-seed", type=float, default=2.0, help="with --hydrate-radius: probe separating pore from lipid space before growing, A (default 2.0)")
    g.add_argument("--hydrate-probe", type=float, default=1.4, help="with --hydrate-radius: probe radius A (default 1.4)")
    g.add_argument("--lumen-seed", choices=["all", "mid"], default="all", help="pore water seeded from space enclosed by protein in any slice of the slab (all, default) or the mid-plane only (mid, the older rule)")
    g.add_argument("--lumen-eps", type=float, default=None, help="dielectric of pore / cavity water inside the membrane slab (default: --exdi = bulk water 80). Confined water is less polarisable; 30-60 is the plausible range")
    g.add_argument("--probe", type=float, default=None, help="probe radius for pore-water detection, A (default: epsmap.py's default)")
    # grid
    g = ap.add_argument_group("grid")
    g.add_argument("--spacing", type=float, default=0.5, help="map spacing, A (default 0.5; smooth background, finer rarely matters)")
    g.add_argument("--pad", type=float, default=20.0, help="map margin around the protein, A (default 20; use >= 15)")
    # titration
    g = ap.add_argument_group("titration")
    g.add_argument("--cluster", type=float, default=12.0, help="Cluster Delimitation Threshold (A), default 12")
    g.add_argument("--ph-start", type=float, default=0.0)
    g.add_argument("--ph-end", type=float, default=14.0)
    g.add_argument("--ph-step", type=float, default=1.0)
    ap.add_argument("--no-checkpoint", action="store_true", help="do not write per-site checkpoints of the energy stage (default: written to <out>/energy_ckpt and used automatically if the run is repeated after an interruption)")
    ap.add_argument("--desolv-cap", type=float, default=0.0, metavar="PK",
                    help="EXPERIMENTAL: cap the reaction-field (desolvation + Born transfer) penalty of every charged side chain at PK pK units (0 = off). "
                         "Continuum models put a charge in an eps 2 slab far too high (a Lys in a model TM peptide shifts ~4 pK, not 20-30)")
    ap.add_argument("--set", action="append", metavar="'Key : value'",
                    help="override/add any other run.prm line, e.g. --set 'Hydrogen of GLU Attached to Atom : OE2' (repeatable)")
    a = ap.parse_args()
    if a.protein_eps is not None and a.protein_eps <= 0:
        a.protein_eps = None                       # old behaviour
    if a.indi is None:
        a.indi = a.protein_eps if a.protein_eps is not None else 2.0

    if not os.path.isfile(a.pdb):
        die("PDB not found: " + a.pdb)
    stem = os.path.splitext(os.path.basename(a.pdb))[0]
    out = os.path.abspath(a.out or (stem + "_membrane"))
    exe = os.path.abspath(a.exe)
    if not os.path.isfile(exe):
        die("delphiPKa binary not found at %s (build it with 'make', or pass --exe)" % exe)
    if not os.path.isdir(a.param_dir):
        die("parameter directory not found: " + a.param_dir)
    try:
        import numpy  # noqa: F401
    except ImportError:
        die("numpy is required (pip install numpy)")
    if a.pad < 12:
        log("WARNING: --pad %g is small; DelPhi's base box reaches beyond it (use >= 15)" % a.pad)

    os.makedirs(out, exist_ok=True)
    dst_param = os.path.join(out, "param")
    if not os.path.exists(dst_param):
        shutil.copytree(a.param_dir, dst_param)
    pdbname = stem + ".pdb"
    pdb_path = os.path.join(out, pdbname)
    if a.fix_names:
        # CHARMM-style names (ILE CD, OT1/OT2, HSP, TIP3 ...) are unknown to the topology: radius 0, charge 0, atom invisible. Rename them.
        import fix_names as _fx
        src = a.pdb if os.path.abspath(a.pdb) != pdb_path else os.path.join(out, stem + "_original.pdb")
        if src != a.pdb:
            shutil.copyfile(a.pdb, src)
        new_lines, wat, st = _fx.fix(open(src).readlines())
        open(pdb_path, "w").writelines(new_lines)
        if st:
            log("name fixes (--no-fix-names to disable): " + "; ".join("%s x%d" % kv for kv in sorted(st.items())))
        if wat:
            wf = os.path.join(out, stem + "_waters.pdb")
            open(wf, "w").writelines(wat)
            log("%d waters were in ATOM records: written to %s (not used unless you pass --waters)" % (len(wat), wf))
        # cofactors without parameters: give metal / sulfide atoms a radius so they are not holes in the protein body
        present = {(l[17:20].strip(), l[12:16].strip()) for l in new_lines if l.startswith("ATOM")}
        siz_path = os.path.join(dst_param, os.path.basename(a.param_siz)) if hasattr(a, "param_siz") else os.path.join(dst_param, {"amber": "parseres.siz", "charmm22": "charmm22.siz", "gromos": "gromos.siz", "parse": "parseres.siz"}[a.ff])
        if os.path.isfile(siz_path):
            known = {tuple(t.split()[:2][::-1]) for t in open(siz_path) if t.strip() and not t.startswith("!") and len(t.split()) >= 3}
            add = []
            for rn, an in sorted(present):
                if (rn, an) in known or an.startswith("H"):
                    continue
                el = re.sub(r"[0-9]", "", an)
                r = {"FE": 1.40, "S": 1.85, "ZN": 1.40, "MG": 1.18, "CA": None, "CU": 1.40, "MN": 1.40}.get(el)
                if r and rn not in ("CYS", "MET") and len(rn) > 0 and rn in ("F4R", "SF4", "FS4", "F3S", "FES", "HEM", "HEC", "CLA", "CHL"):
                    add.append("%-5s %-5s %.2f\n" % (an, rn, r))
            if add:
                with open(siz_path, "a") as fh:
                    fh.write("".join(add))
                log("cofactor atoms without radii got defaults (Fe 1.40, S 1.85 A; charge stays 0): %s" % ", ".join(x.split()[1] + " " + x.split()[0] for x in add))
    elif os.path.abspath(a.pdb) != pdb_path:
        shutil.copyfile(a.pdb, pdb_path)

    if a.optimize_h:
        import hopt as _ho
        if any("Attached to Atom" in x for x in (a.set or [])):
            die("--optimize-h assumes the default proton atoms (OD1 / OE1); do not combine it with --set 'Hydrogen of ... Attached to Atom'")
        try:
            hl = open(pdb_path).readlines()
            carriers, _plog = _ho.optimise(hl, workdir=None)
            hl2, hst, hsw = _ho.apply(hl, carriers)
        except Exception as e:
            die("--optimize-h failed (is pdb2pqr installed? pip install pdb2pqr): %s" % e)
        shutil.copyfile(pdb_path, os.path.join(out, stem + "_before_hopt.pdb"))
        open(pdb_path, "w").writelines(hl2)
        open(os.path.join(out, "hopt.txt"), "w").write("swapped: " + " ".join(hsw) + "\n")
        log("--optimize-h: PDB2PQR H-bond optimiser chose the proton oxygen for %d Asp/Glu; %d differ from the default OD1/OE1 (labels swapped, list in hopt.txt), %d unmatched"
            % (hst["residues"], hst["swapped"], hst["unmatched"]))

    # ---- membrane geometry
    dum, (amin, amax), natoms = pdb_info(pdb_path, a.axis)
    center, core = a.center, a.core_thickness
    shape_json = None
    if a.opm_curved:
        a.lipids = a.pdb; a.lipid_resnames = "DUM"; a.lipid_atoms = "*"; a.p_offset = 0.0
        if not dum:
            die("--opm-curved: no DUM atoms in %s" % a.pdb)
    if a.lipids:
        a.lipids = os.path.abspath(a.lipids)
    if a.membrane_shape:
        a.membrane_shape = os.path.abspath(a.membrane_shape)
    if a.lipids or a.membrane_shape:
        shape_json = os.path.join(out, "membrane_shape.json")
        if a.lipids:
            cmdf = [sys.executable, _find_tool("membrane_shape.py"), "--lipids", a.lipids, "--model", a.shape, "--atom-names", a.lipid_atoms,
                    "--p-offset", str(a.p_offset), "--out", shape_json]
            if a.lipid_resnames:
                cmdf += ["--resnames", a.lipid_resnames]
            if a.lipid_within:
                import numpy as _np
                xyz = _np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in open(pdb_path) if l.startswith("ATOM") and l[17:20].strip() != "DUM"])
                cen = xyz.mean(0)
                cmdf += ["--within", "%.2f" % cen[0], "%.2f" % cen[1], "%.2f" % cen[2], str(a.lipid_within)]
            # the fit is repeated by each stage (prepare / maps / run): reuse it when the inputs are unchanged
            import hashlib as _hl
            _key = _hl.md5((" ".join(cmdf[2:-2]) + "|" + _hl.md5(open(a.lipids, "rb").read()).hexdigest()).encode()).hexdigest()
            _kf = shape_json + ".key"
            if os.path.isfile(shape_json) and os.path.isfile(_kf) and open(_kf).read().strip() == _key:
                log("membrane shape: reusing the fit in %s" % shape_json)
            else:
                run(cmdf, out)
                open(_kf, "w").write(_key + "\n")
        else:
            shutil.copyfile(a.membrane_shape, shape_json)
        import json as _json
        meta = _json.load(open(shape_json))
        core = a.core_thickness if a.core_thickness is not None else meta.get("core_thickness", 30.0)
        log("CURVED membrane (%s): core thickness %.1f A; the layers follow the signed distance to the fitted mid-surface" % (meta["kind"], core))
        a._dist_map = "membrane_dist.dx"
        center = 0.0
    elif dum:
        c0, t0 = (dum[0] + dum[1]) / 2.0, dum[1] - dum[0]
        log("OPM dummy atoms found: %s planes at %.2f and %.2f -> centre %.2f, core thickness %.2f A" % (a.axis, dum[0], dum[1], c0, t0))
        center = c0 if center is None else center
        core = t0 if core is None else core
    else:
        center = 0.0 if center is None else center
        core = 30.0 if core is None else core
        log("no DUM atoms: using centre %.2f and core thickness %.2f A (set --center / --core-thickness)" % (center, core))
    if shape_json:
        log("protein spans %s = %.1f .. %.1f A  (%d atoms)" % (a.axis, amin, amax, natoms))
    else:
        log("protein spans %s = %.1f .. %.1f A  (%d atoms); slab core %.1f .. %.1f A" %
            (a.axis, amin, amax, natoms, center - core / 2, center + core / 2))
    if shape_json is None and not (amin - 1 < center < amax + 1):
        log("WARNING: the membrane centre lies outside the protein: check the orientation of the PDB")

    sys.path.insert(0, os.path.dirname(_find_tool("epsmap.py")))
    from epsmap import slab_profile as _slab
    t_epsmap = _find_tool("epsmap.py")
    t_gauss = _find_tool("gaussian_to_dx.py")
    py = sys.executable
    pqr_file = os.path.join(out, "protein_noHET.pqr")
    t0 = time.time()

    # ---- 1. prepare
    if a.stage in ("all", "prepare"):
        log("step 1/3: protonation (hydrogens, charges, radii)")
        for f in os.listdir(out):
            if f.endswith(".pqr") and f != "protein_noHET.pqr":
                os.remove(os.path.join(out, f))
        write_prm(os.path.join(out, "prepare.prm"), a, pdbname, "F", "F", None)
        run([exe, "prepare.prm"], out, os.path.join(out, "prepare.log"))
        cands = sorted(f for f in os.listdir(out) if f.endswith(".pqr") and f != "protein_noHET.pqr")
        if not cands:
            die("protonation did not produce a PQR file, see prepare.log")
        n = 0
        nh = 0
        with open(os.path.join(out, cands[0])) as fi, open(pqr_file, "w") as fo:
            for line in fi:
                if line.startswith("HETATM"):
                    # groups kept in the calculation (pka-helper --hetero keep / --cofactor) stay: the maps must see them, otherwise their
                    # space is filled with pore water; dummy atoms and waters are always dropped
                    if line[17:20].strip() in ("DUM", "HOH", "WAT", "DOD") or not any(re.sub(r"\s+", " ", x).strip() == "Remove HETATM : F" for x in (a.set or [])):
                        continue
                    nh += 1
                fo.write(line)
                n += line.startswith("ATOM")
        log("PQR %s: %d atoms%s" % (cands[0], n, " + %d HETATM atoms (ligands / cofactors / ions kept)" % nh if nh else " (no HETATM groups)"))

    # ---- 2. maps
    if a.stage in ("all", "maps"):
        if not os.path.isfile(pqr_file):
            die("%s not found: run --stage prepare first" % pqr_file)
        log("step 2/3: dielectric maps (spacing %g A, pad %g A)" % (a.spacing, a.pad))
        run([py, t_gauss, "--type", "EPS", "--spacing", str(a.spacing), "--padding", str(a.pad), "--sigma", str(a.sigma),
             "--indi", str(a.indi), "--exdi", str(a.exdi), pqr_file, os.path.join(out, "protein.dx")], out)
        mcmd = [py, t_epsmap, "membrane", os.path.join(out, "membrane.dx"), "--like", os.path.join(out, "protein.dx"),
                "--axis", a.axis, "--center", str(center), "--core-thickness", str(core), "--head-thickness", str(a.head_thickness),
                "--eps-core", str(a.eps_core), "--eps-head", str(a.eps_head), "--eps-water", str(a.exdi), "--smooth", str(a.smooth)]
        if shape_json:
            mcmd += ["--shape-file", shape_json, "--distance-out", os.path.join(out, "membrane_dist.dx")]
        run(mcmd, out)
        # box-independent layer profile for the Born-transfer term (planar: coordinate along the axis, curved: signed distance)
        import numpy as _np
        _x = _np.arange(-100.0, 100.0 + 1e-9, 0.5)
        _prof = _slab(_x, 0.0, core, a.head_thickness, a.eps_core, a.eps_head, a.exdi, a.smooth)
        _off = 0.0 if shape_json else center
        with open(os.path.join(out, "membrane_stack.txt"), "w") as _f:
            _f.write("# x(A) eps of the undisturbed membrane layers (core %.3f A, head %.3f A, smoothing %.2f A)\n" % (core, a.head_thickness, a.smooth))
            for _xi, _ei in zip(_x, _prof):
                _f.write("%.3f %.4f\n" % (_xi + _off, _ei))
        cmd = [py, t_epsmap, "combine", os.path.join(out, "protein.dx"), os.path.join(out, "membrane.dx"),
               os.path.join(out, "background.dx"), "--background-only", "--interior", str(a.indi), "--bg", str(a.exdi),
               ]
        if a.protein_eps is not None:
            cmd += ["--protein-eps", str(a.protein_eps)]
            if a.solid_interior or a.solid_buried_sasa is not None or a.hydrate_radius:
                cmd += ["--atoms", pqr_file]
            if a.solid_buried_sasa is not None:
                cmd += ["--solid-buried-sasa", str(a.solid_buried_sasa), "--solid-radius", str(a.solid_radius), "--solid-groups", a.solid_groups]
                if a.solid_skip_partnered > 0:
                    cmd += ["--solid-skip-partnered", str(a.solid_skip_partnered)]
                    if a.solid_skip_cations != "LYS,ARG,HIS": cmd += ["--solid-skip-cations", a.solid_skip_cations]
                    cmd += ["--partners-out", os.path.join(out, "solid_partners.txt")]
                    if a.solid_neutral_partners: cmd += ["--solid-neutral-partners", a.solid_neutral_partners]
                if a.solid_spare_partners > 0:
                    cmd += ["--solid-spare-partners", str(a.solid_spare_partners)]
                    if a.solid_spare_acids: cmd += ["--solid-spare-acids"]
            if a.interior_depth is not None:
                cmd += ["--interior-depth", str(a.interior_depth)]
        if a.hydrate_radius:
            if a.protein_eps is None:
                cmd += ["--atoms", pqr_file]
            cmd += ["--hydrate-radius", str(a.hydrate_radius), "--hydrate-groups", a.hydrate_groups, "--hydrate-probe", str(a.hydrate_probe), "--hydrate-seed", str(a.hydrate_seed)]
        if shape_json:
            cmd += ["--shape-file", shape_json]
        if a.waters:
            cmd += ["--waters", a.waters]
        if a.probe is not None:
            cmd += ["--probe", str(a.probe)]
        if a.lumen_eps is not None:
            cmd += ["--lumen-eps", str(a.lumen_eps)]
        if a.lumen_seed != "all":
            cmd += ["--lumen-seed", a.lumen_seed]
        run(cmd, out)
        run([py, t_epsmap, "info", os.path.join(out, "background.dx")], out)

    # ---- 3. run
    if a.stage in ("all", "run"):
        if not os.path.isfile(os.path.join(out, "background.dx")):
            die("background.dx not found: run --stage maps first")
        log("step 3/3: DelPhiPKa (%d rank%s) - this takes minutes to hours" % (a.np, "" if a.np == 1 else "s"))
        if os.path.isfile(os.path.join(out, "membrane_stack.txt")):
            a._stack_file = "membrane_stack.txt"
        _sj = os.path.join(out, "membrane_shape.json")
        if os.path.isfile(_sj):
            import json as _j2
            _kind = _j2.load(open(_sj)).get("kind", "plane")
            if _kind != "plane" and not os.path.isfile(os.path.join(out, "membrane_dist.dx")):
                die("curved membrane (%s) but %s/membrane_dist.dx is missing: without it the Born-transfer term uses flat layers and gives "
                    "wrong energies. Re-run the maps stage (--stage maps) or use --stage all." % (_kind, out))
            if os.path.isfile(os.path.join(out, "membrane_dist.dx")):
                a._dist_map = "membrane_dist.dx"
        write_prm(os.path.join(out, "run.prm"), a, pdbname, "T", "T", "background.dx")
        for f in ("pKa.csv", "titra.csv"):
            if os.path.exists(os.path.join(out, f)):
                os.remove(os.path.join(out, f))
        run(mpi_cmd(a, exe, "run.prm"), out, os.path.join(out, "run.log"))

        text = open(os.path.join(out, "run.log"), errors="replace").read()
        bad = [l for l in text.splitlines() if "[ERROR]" in l]
        warn = [l.strip() for l in text.splitlines() if "[WARNING]" in l]
        for l in text.splitlines():
            if any(k in l for k in ("Custom dielectric map :", "values ", "margins", "epsilon of the map at the atoms")):
                print("   " + l.strip())
        for l in warn:
            print("   " + l)
        if bad:
            die("DelPhi reported errors:\n  " + "\n  ".join(bad) + "\n(see %s/run.log and delphi_run_*.log)" % out)
        pka = os.path.join(out, "pKa.csv")
        if not os.path.isfile(pka):
            die("no pKa.csv produced, see %s/run.log" % out)
        log("pKa table: " + pka)
        rows = open(pka).read().splitlines()
        print("\n".join(rows[:1] + rows[1:16]))
        if len(rows) > 16:
            print("   ... (%d residues, see pKa.csv)" % (len(rows) - 1))
        log("values marked '*' lie outside pH 0-14 or are saturated: they are extrapolations / bounds (e.g. 21.0* means >= 21)")
        if not a.no_report_clusters:
            try:
                import group_pka
                group_pka.main([out, "--cutoff", str(a.cluster_cutoff)])
                log("coupled acids share protons: compare an experiment that follows the LAST proton of a group with pKa_n in clusters_macro.csv, not with the single-site values")
            except SystemExit as e:
                log("cluster report skipped: %s" % e)
            except Exception as e:
                log("cluster report skipped (%s)" % e)
    log("done in %.0f s; files in %s" % (time.time() - t0, out))


if __name__ == "__main__":
    main()
