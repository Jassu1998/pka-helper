# Third-party software and licences

**pka-helper** (the Python package in `src/pka_helper/`, the patch in `delphipka-patch/`, the benchmark and the documentation): MIT
licence, see [LICENSE](LICENSE).

**DelPhiPKa / DelPhi** (Lin Wang, Lin Li, Emil Alexov; Alexov lab, Clemson University) is **not included** in pka-helper. The installer
downloads it from its authors' repository (github.com/delphi001/DelphiPka, v1.0) and applies pka-helper's patch, which holds only
pka-helper's changes (plus, as in any patch, a few lines of surrounding context from the original files). DelPhiPKa, including its
force-field parameter files, is used under its authors' terms: check them before using it, especially for commercial work
(contact: delphi@g.clemson.edu).

**PROPKA 3** is installed from PyPI as a dependency (`propka`, LGPL 2.1). **numpy, scipy**: PyPI dependencies (BSD licences).

**Example and benchmark structures** (`examples/`, `benchmark/structures/`) come from the Protein Data Bank, oriented by OPM where a
membrane is needed, plus one AlphaFold model (Hv1). PDB data are free of all copyright restrictions (CC0); AlphaFold DB models are
CC BY 4.0.
