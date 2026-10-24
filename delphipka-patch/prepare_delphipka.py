"""Get the original DelPhiPKa source, apply pka-helper's changes, and (optionally) compile it.

pka-helper does not include DelPhiPKa. Its changes are kept as a patch against the authors' public release
(github.com/delphi001/DelphiPka, tag v1.0 = commit 2d84023), which you obtain from the authors under their terms.

    python3 prepare_delphipka.py --out build_dir                      # clone v1.0 from GitHub (needs git + internet), patch, compile
    python3 prepare_delphipka.py --src /path/to/DelphiPka --out d     # use a copy you already have (must be v1.0)
    python3 prepare_delphipka.py --out d --no-make                    # patch only

Result: <out>/delphiPKa (the program) and <out>/param (force-field files). `pip install .` runs this automatically.
"""
import argparse, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, "pka-helper-delphipka.patch")
REPO = "https://github.com/delphi001/DelphiPka"
COMMIT = "2d84023b699f9427bbde486c27f1932ccb78f783"          # tag v1.0, 2018-03-05


def fetch(dest):
    """Clone the authors' repository at the pinned commit into dest."""
    if not shutil.which("git"):
        raise RuntimeError("git is not installed: install git, or download DelPhiPKa v1.0 from %s yourself and pass --src" % REPO)
    print("pka-helper: downloading DelPhiPKa v1.0 from %s ..." % REPO, flush=True)
    subprocess.run(["git", "clone", "-q", REPO, dest], check=True)
    subprocess.run(["git", "-C", dest, "checkout", "-q", COMMIT], check=True)


def apply_patch(src):
    """Apply the pka-helper patch in src (tries git apply, then patch)."""
    if shutil.which("git"):
        r = subprocess.run(["git", "apply", "--whitespace=nowarn", PATCH], cwd=src)
        if r.returncode == 0: return
    if shutil.which("patch"):
        r = subprocess.run(["patch", "-p1", "-s", "-i", PATCH], cwd=src)
        if r.returncode == 0: return
    raise RuntimeError("the pka-helper patch does not apply to %s. It is made for DelPhiPKa v1.0 (%s, commit %s); "
                       "use exactly that version (and install git or patch)." % (src, REPO, COMMIT[:7]))


def prepare(out, src=None, make=True, jobs=None):
    """-> out with the patched source, out/delphiPKa (if make) and out/param."""
    if os.path.exists(out): shutil.rmtree(out)
    if src:
        shutil.copytree(src, out, ignore=shutil.ignore_patterns(".git", "*.o"))
    else:
        fetch(out)
    apply_patch(out)
    if make:
        missing = [t for t in ("mpic++", "make") if not shutil.which(t)]
        if missing: raise RuntimeError("cannot compile DelPhiPKa: %s not found (see BUILDING.md)" % ", ".join(missing))
        print("pka-helper: compiling DelPhiPKa (a few minutes) ...", flush=True)
        r = subprocess.run(["make", "-j", str(jobs or os.cpu_count() or 2)], cwd=out)
        if r.returncode != 0 or not os.path.isfile(os.path.join(out, "delphiPKa")):
            raise RuntimeError("compiling DelPhiPKa failed (see the compiler output above). It needs a C++11 MPI compiler (mpic++), "
                               "the GNU Scientific Library with headers and the Boost C++ headers.")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="directory to create (replaced if it exists)")
    ap.add_argument("--src", help="an existing DelPhiPKa v1.0 source directory (default: clone it from GitHub)")
    ap.add_argument("--no-make", action="store_true", help="only apply the patch")
    a = ap.parse_args()
    try:
        d = prepare(os.path.abspath(a.out), a.src, not a.no_make)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        sys.exit("pka-helper: %s" % e)
    print("done: %s%s, force-field files in %s" % (d, "" if a.no_make else ", program " + os.path.join(d, "delphiPKa"), os.path.join(d, "param")))
    if not a.no_make:
        print("use it with:  export PKA_HELPER_DELPHI=%s  PKA_HELPER_PARAM=%s" % (os.path.join(d, "delphiPKa"), os.path.join(d, "param")))
