# Building DelPhiPKa by hand

pka-helper does not include DelPhiPKa. `pip install .` normally downloads DelPhiPKa v1.0 from its authors
(github.com/delphi001/DelphiPka, commit 2d84023), applies pka-helper's patch (`delphipka-patch/pka-helper-delphipka.patch`), compiles
it, and installs the program and its force-field files inside the package. You only need this page if that step fails or you want to do
it yourself.

Requirements: git (or a downloaded copy of DelPhiPKa v1.0), a C++11 compiler with MPI (`mpic++`, OpenMPI or MPICH) and the GNU
Scientific Library (GSL) with headers, and the Boost C++ headers.

    # Debian / Ubuntu
    sudo apt-get install git build-essential libopenmpi-dev openmpi-bin libgsl-dev libboost-dev
    # Fedora / RHEL
    sudo dnf install git gcc-c++ make openmpi-devel gsl-devel boost-devel && module load mpi/openmpi-x86_64
    # macOS (Homebrew)
    brew install git open-mpi gsl boost    # then: GSL_PREFIX=$(brew --prefix gsl) CPATH=$(brew --prefix)/include pip install .

**No internet on the machine.** Download DelPhiPKa v1.0 elsewhere (`git clone https://github.com/delphi001/DelphiPka && git -C DelphiPka
checkout v1.0`), copy it over, and point the installer at it:

    PKA_HELPER_DELPHIPKA_SRC=/path/to/DelphiPka pip install .

**Build it yourself** and tell pka-helper where it is:

    python3 delphipka-patch/prepare_delphipka.py --out delphipka-build          # downloads, patches and compiles
    #   (or: --src /path/to/DelphiPka  to use a copy you have; GSL elsewhere: GSL_PREFIX=/path/to/gsl python3 ...)
    PKA_HELPER_SKIP_BUILD=1 pip install .
    export PKA_HELPER_DELPHI=$PWD/delphipka-build/delphiPKa
    export PKA_HELPER_PARAM=$PWD/delphipka-build/param

**Apply the patch by hand**: in a DelPhiPKa v1.0 directory, `git apply path/to/pka-helper-delphipka.patch` (or
`patch -p1 < path/to/pka-helper-delphipka.patch`), then `make`.

Inside an active conda environment the Makefile adds `$CONDA_PREFIX/include` and `$CONDA_PREFIX/lib` automatically.

Check everything with `pka-helper doctor`.
