# pka-helper's changes to DelPhiPKa

pka-helper uses a modified DelPhiPKa (Wang, Li & Alexov; Alexov lab, Clemson University) for its Poisson-Boltzmann calculations.
DelPhiPKa itself is **not included**: it is distributed by its authors under their own terms. This folder holds only pka-helper's changes,
as a patch against the authors' public release:

* base: <https://github.com/delphi001/DelphiPka>, tag **v1.0** (commit `2d84023`, March 2018)
* `pka-helper-delphipka.patch`: the changes (apply in the DelPhiPKa directory with `git apply` or `patch -p1`)
* `prepare_delphipka.py`: downloads v1.0, applies the patch and compiles it. `pip install .` runs it for you.

The force-field parameter files (`param/`) come unchanged from the DelPhiPKa download.

## What the patch changes
1. **Membrane model**: DelPhi can read a dielectric map (lipid core, pore water, low-dielectric zones) instead of, or under, its Gaussian
   dielectric (new `src/delphi/misc/misc_customeps.*`), with an analytic Born term for moving a charge from water into the membrane
   (flat or curved membranes).
2. **Options pka-helper needs**: compute only listed sites ("Energy sites only"), titrate only some chains, keep Lys/Arg fixed,
   salt concentration.
3. **Bug fixes**:
   * pKa interpolation used the pH index instead of the pH value (wrong pKas for any pH step other than 1); saturated values are now
     extrapolated and marked `*`;
   * titration clusters are matched by atom rather than by residue number and chain (repeated or blank chain IDs made clusters grow until
     the titration never finished); clusters are capped at 18 members, with a warning;
   * a failed DelPhi run is reported instead of silently reusing the previous energies;
   * MPI calls use the C interface, so the code compiles with OpenMPI 5 and other current MPI versions.
4. **Speed and robustness**: a faster solver setting with a fallback if it diverges, faster dielectric-boundary and Gaussian-map
   construction (same results), a smaller default box option, checkpoints so an interrupted energy stage resumes, progress output,
   and an experimental multigrid solver (off by default).
5. **Build**: Makefile with the new files, GSL found in conda environments or under `GSL_PREFIX`.
