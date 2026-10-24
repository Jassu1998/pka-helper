"""Build hook: get DelPhiPKa, apply pka-helper's patch, compile it, and ship the program and its force-field files inside the package.

    pip install .                                         downloads DelPhiPKa v1.0 from its authors' GitHub repository (needs git and internet)
    PKA_HELPER_DELPHIPKA_SRC=/path/to/DelphiPka pip install .   use a copy of DelPhiPKa v1.0 you already have
    PKA_HELPER_SKIP_BUILD=1 pip install .                 skip it (see BUILDING.md; then set PKA_HELPER_DELPHI and PKA_HELPER_PARAM)

DelPhiPKa is not part of pka-helper: only the patch (delphipka-patch/) is. A development checkout that has the patched source in
delphipka/ builds from that instead.
"""
import os, shutil, sys
from setuptools import setup, Distribution
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "delphipka-patch"))


class BuildWithDelphi(build_py):
    def run(self):
        super().run()
        if os.environ.get("PKA_HELPER_SKIP_BUILD"):
            print("pka-helper: PKA_HELPER_SKIP_BUILD set, not building DelPhiPKa")
            return
        import subprocess
        work = os.path.join(self.build_lib, "_delphipka_build")
        try:
            dev = os.path.join(HERE, "delphipka")
            if os.path.isfile(os.path.join(dev, "Makefile")):          # development checkout: already patched source
                if os.path.exists(work): shutil.rmtree(work)
                shutil.copytree(dev, work, ignore=lambda d, names: [n for n in names if n.endswith(".o") or (n == "delphiPKa" and os.path.isfile(os.path.join(d, n)))])
                print("pka-helper: compiling DelPhiPKa from delphipka/ (a few minutes) ...", flush=True)
                if subprocess.run(["make", "-j", str(os.cpu_count() or 2)], cwd=work).returncode != 0:
                    raise RuntimeError("compiling DelPhiPKa failed (see the compiler output above)")
            else:
                import prepare_delphipka as P
                P.prepare(work, os.environ.get("PKA_HELPER_DELPHIPKA_SRC") or None, make=True)
        except (RuntimeError, subprocess.CalledProcessError, OSError) as e:
            sys.exit("pka-helper: %s\n  It needs git (to download DelPhiPKa v1.0 from its authors), a C++11 MPI compiler (mpic++), the GNU "
                     "Scientific Library and the Boost C++ headers, e.g.\n    sudo apt-get install git build-essential libopenmpi-dev openmpi-bin libgsl-dev libboost-dev\n"
                     "    conda install -c conda-forge git compilers openmpi gsl libboost-headers make\n"
                     "  Or install without it: PKA_HELPER_SKIP_BUILD=1 pip install .  (see BUILDING.md)" % e)
        pkg = os.path.join(self.build_lib, "pka_helper")
        bindir = os.path.join(pkg, "bin"); os.makedirs(bindir, exist_ok=True)
        shutil.copy(os.path.join(work, "delphiPKa"), os.path.join(bindir, "delphiPKa")); os.chmod(os.path.join(bindir, "delphiPKa"), 0o755)
        par = os.path.join(pkg, "param")
        if not os.path.isdir(par) and os.path.isdir(os.path.join(work, "param")):   # force-field files from the DelPhiPKa download
            shutil.copytree(os.path.join(work, "param"), par)
        shutil.rmtree(work)
        print("pka-helper: DelPhiPKa installed into the package")


class BinaryDistribution(Distribution):
    def has_ext_modules(self):      # the wheel carries a compiled binary: tag it for this platform, not "any"
        return True


setup(cmdclass={"build_py": BuildWithDelphi}, distclass=BinaryDistribution)
