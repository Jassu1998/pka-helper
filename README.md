# pka-helper

**pKa values for membrane proteins.** pka-helper predicts the pKa of every Asp, Glu, His, Lys and Arg in a protein that sits in a lipid
membrane, and tells you which groups are protonated at your pH.

It combines two methods:

* **PROPKA 3**, a fast empirical predictor. It is reliable for most groups but does not know about the membrane.
* **DelPhiPKa**, a Poisson-Boltzmann calculation, run here with a model of the membrane: low-dielectric lipid core, water-filled pores
  and cavities, and a flat or curved bilayer taken from your OPM/PPM structure.

The final pKa starts from PROPKA's value. The membrane calculation may move it only within limits that were set on a benchmark of
measured and simulated pKas. Every value comes with a confidence call, a note on why it moved, and warnings about the input structure.

> **Version 0.1.1a1 (alpha).** Research software: results and defaults may still change between releases. Read [How far to trust the numbers](#how-far-to-trust-the-numbers) before you use a single pKa
> for a conclusion.

---

## Contents

1. [Install](#install)
2. [Quick start](#quick-start)
3. [Preparing your structure](#preparing-your-structure)
4. [What you get](#what-you-get)
5. [How far to trust the numbers](#how-far-to-trust-the-numbers)
6. [Useful options](#useful-options)
7. [Structure checks](#structure-checks)
8. [How it works](#how-it-works)
9. [Accuracy](#accuracy)
10. [Reproducing the benchmark](#reproducing-the-benchmark)
11. [Credits and citation](#credits-and-citation)

---

## Install

You need **Linux or macOS**, **Python 3.8 or newer**, **git**, a **C++ compiler with MPI** (`mpic++`, from OpenMPI or MPICH) and the
**GNU Scientific Library** (GSL, with headers), and an internet connection during installation.

pka-helper does not include DelPhiPKa. During installation it downloads DelPhiPKa v1.0 from its authors' repository
(github.com/delphi001/DelphiPka), applies pka-helper's changes (`delphipka-patch/`) and compiles it, which takes a few minutes. DelPhiPKa
is used under its authors' terms (see [NOTICE.md](NOTICE.md)). Already have DelPhiPKa v1.0, or no internet on the machine? Use
`PKA_HELPER_DELPHIPKA_SRC=/path/to/DelphiPka pip install .`

**Debian / Ubuntu**

    sudo apt-get install git build-essential libopenmpi-dev openmpi-bin libgsl-dev
    cd pka-helper          # the folder you cloned or unpacked
    pip install .

**conda (any Linux / macOS, no root needed)**

    conda create -n pka -c conda-forge python=3.11 git compilers openmpi gsl make
    conda activate pka
    cd pka-helper          # the folder you cloned or unpacked
    pip install .

(The build finds GSL inside the active conda environment. The conda route has not been tested on this release; if it fails, see
[BUILDING.md](BUILDING.md). GSL installed somewhere else: `GSL_PREFIX=/path/to/gsl pip install .`)

`pip install .` also installs numpy, scipy and PROPKA. Then check the installation:

    pka-helper doctor

Every line should say `ok`. If git, the compiler or GSL is missing, `pip install .` stops with a message saying what to install. To install
without building DelPhiPKa (for example, to use a binary built elsewhere), see [BUILDING.md](BUILDING.md).

---

## Quick start

An example is included: proteorhodopsin (PDB 2L6X, oriented by OPM). It takes about one minute on two cores.

    cd examples/proteorhodopsin
    pka-helper run proteorhodopsin_2l6x.pdb

pka-helper prints its progress, then the groups where PROPKA and pka-helper disagree on the protonation state at pH 7:

    pH 7.0: 25 titratable groups, 22 confident, 3 uncertain (within 0.5 pK of pH or predictors disagree).

      residue   chain  PROPKA  DelPhi     pKa  state limits varied  note
      ASP227    A        8.23    4.41    4.41  D?    4.41 - 5.73
      GLU108    A        4.98   11.98    7.98  P?    6.98 - 8.98 !  lipid-facing: larger cap

    23 other groups: PROPKA and pka-helper agree at pH 7.0 (all values: results/pkas.txt); 1 of them is close to the pH or has a
    structure warning (results/uncertain_at_pH7.txt).

and writes everything to `proteorhodopsin_2l6x_pka/`. The `expected_results/` folder next to the example holds the output of a reference
run, so you can compare.

For your own protein:

    pka-helper run my_protein_opm.pdb                 # membrane protein, OPM/PPM-oriented
    pka-helper run my_protein.pdb --no-membrane       # soluble protein (or a membrane protein in water)
    pka-helper run my_protein_opm.pdb --ph 5.5        # report protonation states at another pH
    pka-helper run my_protein_opm.pdb --fast          # minutes: PROPKA + membrane map only, flags lipid-facing acids

**Run time.** Small proteins take about a minute; a 1,000-residue trimer (OmpF) about 15 minutes on two cores; very large complexes can take
hours. pka-helper picks the number of parallel processes from your cores and free memory. A run that is interrupted can be continued with
`--resume`.

---

## Preparing your structure

* **Orientation.** Download the structure from the **OPM database** (opm.phar.umich.edu) or orient it with the **PPM server**. These put the
  membrane normal along z and add dummy atoms (`DUM`) that mark the membrane boundaries; pka-helper reads the membrane position and shape
  from them. Without dummy atoms it assumes a flat membrane centred at z = 0 with a 30 Å hydrophobic core (change with `--center`,
  `--core-thickness`, `--axis`) and says so.
* **Use the state you want to model.** The structure decides the answer more than anything else: an open or closed gate, a missing salt
  bridge, or a side chain built facing the lipid can move a pKa by several units. If the protein has several conformations, run each one.
* **Complete side chains.** Missing side-chain atoms are reported (see [Structure checks](#structure-checks)). Rebuild them first (for
  example with PDBFixer, Modeller or PyMOL) if they belong to groups you care about.
* **Ligands, cofactors, bound lipids and modified residues (HETATM groups)** stay in the calculation, so their space is not filled with
  water:
  * modified residues with the charge of a standard residue are read as that residue (selenomethionine, formylmethionine,
    hydroxyproline, mono- and dimethyl-lysine, ...);
  * groups with a known charge keep it, on the right atoms: phospho-Ser/Thr/Tyr −2, carboxylated Lys −1, trimethyl-lysine +1, the
    retinal Schiff base when retinal and lysine are one residue (LYR) +1, cardiolipin −2, phosphatidylglycerol −1;
  * everything else (retinal, heme, substrates, neutral lipids, unknown ligands) is kept as an **uncharged** volume. Its charge is not
    known to pka-helper; if you know it, give it: `--cofactor HEM:-1`;
  * **ions are left out** by default, because crystals often contain ions from the crystallisation buffer that are absent where pKas are
    measured. Keep them with `--keep-ions`, or one kind with `--cofactor ZN:+2`;
  * crystallisation additives, buffer molecules and detergents (sulfate, glycerol, PEG, maltosides, ...) are left out: in solution that
    space is water.

  The run log and `results/structure_check.txt` list what was done with each group. `--drop-het RES` leaves a group out;
  `--hetero drop` leaves every HETATM group out. PROPKA does not see kept ligands, only the DelPhi calculation does.
* **Hydrogens, waters, alternate conformations.** Hydrogens and waters are removed and rebuilt by the calculation. For alternate
  conformations only the first (blank or A) is used. Selenomethionine is read as methionine. Chains with blank or repeated IDs are relabelled
  in a copy; your file is never changed.
* **NMR ensembles.** Only the first model is used, unless you give `--ensemble`, which runs every model and reports mean and spread.

---

## What you get

Everything goes into `<name>_pka/` (change with `--out`):

| file | contents |
|---|---|
| `results/pkas.txt` | every group: PROPKA, DelPhi, final pKa, protonation state at the pH, effect of the blend limits, notes |
| `results/pka_helper.csv` | the same as a table, with more columns (below) |
| `results/uncertain_at_pH7.txt` | only the groups whose protonation state is in doubt |
| `results/comparison_with_propka.txt` | side by side with PROPKA, biggest differences first |
| `results/structure_check.txt` | problems in the input structure that make a pKa unreliable |
| `results/clusters.txt` | coupled groups (e.g. a His tetrad, a carboxylate pair) with their stepwise pKas |
| `raw/` | PROPKA output, the DelPhiPKa calculation, and `log.txt` with every step |

Main columns of `pka_helper.csv`:

* `PROPKA`, `DelPhiPKa`: the two methods' values. `saturated` = 1 means DelPhi hit the edge of its range (only "very high/low" is known).
* `blended`: **the pKa to use.** `shift_applied` = blended − PROPKA.
* `state_at_pH7` (or your `--ph`): protonated / deprotonated.
* `confidence`: `confident` when PROPKA and the final value agree on the protonation state and are both at least 0.5 pK from the pH;
  otherwise `uncertain`, with the reason in `confidence_reason`.
* `param_lo`, `param_hi`, `param_flips_call`: the range of the final pKa when pka-helper's own limits on the DelPhi shift are varied
  (81 combinations), and whether the protonation state flips inside that range. On screen this is the `limits varied` column:
  `unaffected` when every combination gives the same value. A wide range means the result depends on a judgement call. It does not
  measure errors in PROPKA or DelPhi themselves: an `unaffected` group can still be wrong.
* `environment`: burial, lipid contact, nearby charges, depth in the membrane.
* `likely_more_reliable` and `reliability_reason`: for this kind of group, which method did better on the benchmark, and why.
* `structure_warning`: see [Structure checks](#structure-checks).
* `sigma`: the estimated error of `blended` (1 standard deviation, pK; `±` on screen). `P_protonated`: the probability that the group is
  protonated at the pH, from `blended` and `sigma` (`P(prot)` on screen). With `--ensemble`, `sigma` also includes the spread over the models.

### Error estimates
`sigma` comes from an error model fitted to the benchmark (285 sites: 201 measured soluble sites, 84 membrane and peptide sites, about
three quarters of them constant-pH simulation references). It grows when the group is buried (PROPKA burial ≥ 50 %), is a histidine, sits
in a membrane run, and when PROPKA and DelPhi disagree:

| group | typical sigma |
|---|---|
| exposed group, soluble protein, methods agree | 0.5–0.6 |
| buried group, soluble protein | 1.2–1.8 |
| exposed group, membrane protein | 1.0–1.4 |
| buried group in a membrane protein, methods disagree by 3–5 pK | 2.5–3.5 |

Tested on protein families left out of the fit: 69 % of the reference values lie within ±1 sigma and 95 % within ±2 sigma (ideal 68 / 95),
in soluble and membrane proteins alike. On the seven measured sites never used for anything (the membrane test set and alpha-lytic protease)
all lie within ±1 sigma, so for membrane proteins the estimate may be on the cautious side. `P_protonated` between about 0.1 and 0.9 is
likewise cautious: on the benchmark such calls were right more often than the probability says. Large sigmas are real: buried acids in
membrane proteins are where both methods can be off by 3–4 pK (LacY Glu325, bacteriorhodopsin Asp96).

---

## How far to trust the numbers

* **Most groups (exposed, or in the water-filled parts of the protein):** expect about 0.5 pK typical error, similar to or a little better
  than PROPKA.
* **Groups buried in the protein or facing the lipid:** errors of 1–3 pK are common, for every method. On the membrane benchmark pka-helper
  averages about 1.0 pK against 1.3 for PROPKA.
* **Known weak spots:**
  * **Buried acids that really are protonated** (e.g. bacteriorhodopsin Asp96, LacY Glu325) tend to come out too low.
  * **Acids whose measured pKa is raised for reasons the continuum model misses** (catalytic acids, unusual packing) can be pulled down
    too far.
  * **Coupled histidine tetrads** (influenza M2 His37, BM2 His27) are not reproduced.
  * **Lipid-facing groups** depend strongly on where the membrane is; check them with `--membrane-scan`.
* **When PROPKA and pka-helper disagree about the protonation state of a buried group, treat the state as unknown.** These groups are
  marked `uncertain`, and they are the ones worth checking with another method (constant-pH MD, free-energy calculations, an experiment).

---

## Useful options

`pka-helper run -h` lists all of them.

| option | what it does |
|---|---|
| `--ph 6.5` | report protonation states at this pH (default 7) |
| `--no-membrane` | soluble protein: no lipid, protein in water |
| `--fast` | no energy calculation: PROPKA values plus the membrane map, only to flag lipid-facing acids (minutes) |
| `--cofactor RES[:Q]` | give a HETATM group a charge (or keep it when it would be left out), e.g. `--cofactor HEM:-1`, `--cofactor ZN:+2` |
| `--keep-ions` | keep the metal and halide ions of the structure with their usual charge |
| `--hold-schiff-base` | hold a lysine bound to a separate retinal (RET) charged, as the protonated Schiff base (measured pKa > 12), instead of titrating it; pka-helper warns when it would be needed |
| `--lipid-acids GLU58` | treat chosen Asp/Glu as uncompensated (no stabilising partner), so DelPhi may raise them by up to 3 pK; e.g. `GLU58:K,GLU73` (residue:chain) |
| `--no-partner-check` | skip the second pass that removes a salt-bridge exemption when the partner is itself neutral |
| `--drop-het RES[,RES]` / `--hetero drop` | leave chosen HETATM groups / all of them out |
| `--membrane-scan` | also run with the core 3 Å thinner and thicker; shows which pKas depend on the membrane position (3 runs) |
| `--ensemble` | run every MODEL of the file and report the mean and spread (one run per model) |
| `--lipid-eps 10` | softer lipid core. Better for groups that face the lipid directly, worse for some buried ones; on the benchmark the two settings tie. Run both and treat a disagreement as uncertain |
| `--chain A` | report only this chain (the calculation still includes all chains) |
| `--show all` | list every group on screen, not only those where PROPKA and pka-helper disagree |
| `--resume` | continue an interrupted run, reusing finished steps |
| `--np 4` | number of parallel processes (default: from cores and free memory) |
| `--skip-arg-lys` | quick look: Lys and Arg are not titrated (about half the time); acids near a buried Lys/Arg may come out too low |
| `--alt table.csv` | pKas from another predictor (e.g. pKAI+, KaML), used only for the confidence call |
| `-v` | show the full output of every step |

The blend limits can be changed (`--acid-cap`, `--acid-up-scale`, `--lipid-cap`, `--his-down`, `--no-his-rule`, `--no-lipid-auto`, ...),
but the defaults are the ones that were tested on the benchmark.

Other commands: `pka-helper blend` (blend an existing DelPhiPKa table with a PROPKA file), `pka-helper membrane` (only the membrane and
DelPhi stages), `pka-helper shape` (fit or define a curved membrane), `pka-helper doctor` (check the installation).

---

## Structure checks

`results/structure_check.txt` and the `structure_warning` column flag the input problems that caused the largest errors on the benchmark.

| warning | meaning | what to do |
|---|---|---|
| `protonated-geometry` | a buried Asp/Glu H-bonded to a backbone carbonyl O (≤ 3.1 Å) with no Lys/Arg/His nearby: the structure itself shows a neutral acid | expect a higher pKa than predicted |
| `deep-lipid` | a group facing the lipid at least 5 Å inside the hydrophobic core | the result depends on the membrane position and side-chain orientation; try `--membrane-scan` |
| `saturated` | DelPhi's value is at the edge of its range (marked `*`) | only "very high" or "very low" is known |
| `missing-atoms` | side-chain atoms missing, or a chain break within 8 Å of the group | rebuild the side chain / loop |
| `input` | several MODELs, alternate conformations, resolution worse than 3 Å; what was done with each HETATM group (kept uncharged, ions left out, ...) | check the model, and give ligand charges with `--cofactor` if you know them |

---

## How it works

1. **PROPKA** gives a first pKa for every group.
2. **The membrane model** is built from the dummy atoms (flat or curved). The hydrophobic core has dielectric 2 (or `--lipid-eps`), the
   protein 4, water 80, and water-filled pores and cavities inside the membrane 40. Pores are found slice by slice; gaps narrower than a lipid
   tail stay water. Buried acids get a small low-dielectric zone around them, except where a salt bridge stabilises them. If that salt-bridge
   partner itself comes out neutral at the chosen pH (judged by pka-helper's own first-pass value; a retinal Schiff-base lysine always
   counts as charged), the bridge is not there, and the acid is recomputed with its zone (`--no-partner-check` turns this off). Ligands,
   cofactors and bound lipids kept in the calculation count as protein, so their pockets are not filled with water.
3. **DelPhiPKa** computes the energy of charging each group in that environment and titrates all groups together. For buried Asp/Glu whose
   H-bond partner sits at the other carboxylate oxygen, the neutral state is also computed with the proton on that oxygen, and the better
   placement is kept.
4. **The blend.** The final pKa = PROPKA + the DelPhi−PROPKA difference, limited by rules that reflect where DelPhi was right on the
   benchmark:
   * Acids pushed **down** by DelPhi: up to 1 + 4 × burial pK.
   * Acids pushed **up**: 30 % of the push, at most 0.5 pK. Upward pushes point the right way but overshoot.
   * **Lipid-facing acids** with no positive partner (no Arg/Lys/His within 8 Å): up to 3 pK.
   * **Histidines**: down by at most 1.5, up by at most 3.
   * **Surface lysines** keep PROPKA's value. **Lipid-facing lysines** may drop by up to 4 pK, depending on their depth in the core.
   * **Carboxylate pairs** that share a proton use DelPhi's full coupling.
5. **Checks**: confidence call, effect of the blend limits, structure warnings, clusters of coupled groups.

DelPhiPKa is used in a modified form: pka-helper's patch (`delphipka-patch/`, applied at installation) adds the custom dielectric map for
the membrane, recomputation of selected sites, a faster solver setting, and fixes for repeated chain IDs, large clusters and pKa
interpolation. The list of changes is in [delphipka-patch/README.md](delphipka-patch/README.md).

---

## Accuracy

Mean absolute error in pK units / number of groups on the correct side of pH 7. All numbers come from `benchmark/` and can be reproduced.

| test set | how it was used | PROPKA | pka-helper |
|---|---|---|---|
| Membrane, 69 sites (OmpF, HV1, NhaA, proteorhodopsin, EmrE, BM2); mostly constant-pH MD references | rules were developed on it | 1.32 / 61 | **1.04 / 64** |
| Soluble proteins, 201 measured sites in 22 structures (PKAD-R) | two blend settings were chosen with it, each protein family held out in turn | 0.60 / 195 | **0.52 / 196** |
| Locked membrane test, 4 measured sites (bacteriorhodopsin Asp85/96/212; LacY Glu325) | never used for any choice | 3.17 / 3 | **2.54 / 3** |
| New membrane test, 4 measured sites in 3 proteins (ESR Asp85, Gloeobacter rhodopsin Asp121, GLIC Glu35/Glu181) | assembled after development, never used for any choice | 0.88 / 4 | **0.42 / 4** |

* Against **measured** values alone (211 sites), pka-helper is 0.58 vs PROPKA 0.69. The gain on the membrane set is larger (about 0.3 pK),
  but that set was used to develop the rules, and its references are mostly simulations.
* pka-helper is worse than PROPKA by more than 0.25 pK on about 1 group in 9, mostly acids whose measured pKa is raised by effects the
  continuum model misses (see [How far to trust the numbers](#how-far-to-trust-the-numbers)).
* The membrane data are small: six development proteins, four measured sites in the locked test and four in the new test (see
  `benchmark/MEMBRANE_TEST_SET.md`). Treat the membrane accuracy as indicative.

---

## Reproducing the benchmark

    python3 benchmark/run_benchmark.py                      # all suites, about 1–1.5 h on 2 cores
    python3 benchmark/run_benchmark.py --suite soluble      # membrane | peptides | soluble | locked

Structures, reference values and the scoring rules are in `benchmark/` (see [benchmark/README.md](benchmark/README.md)). Each run is kept
under `benchmark/runs/<tag>/` with a `report.md`.

Tests: `python3 -m pytest tests` (needs pytest).

---

## Credits and citation

pka-helper builds on:

* **PROPKA 3**: Olsson, Søndergaard, Rostkowski & Jensen, *J. Chem. Theory Comput.* 7, 525 (2011). Installed from PyPI (`propka`).
* **DelPhiPKa**: Wang, Li & Alexov, *Proteins* 83, 2186 (2015), and Wang, Zhang & Alexov, *Bioinformatics* 32, 614 (2016). Downloaded from
  its authors (Alexov lab, Clemson University) at installation and modified by pka-helper's patch; not included in pka-helper.
* **OPM / PPM** for membrane orientation: Lomize et al., *Nucleic Acids Res.* 40, D370 (2012).

If you use pka-helper, please cite it (the DOI of the release you used; see [CITATION.cff](CITATION.cff)) together with PROPKA and
DelPhiPKa, and give the version (`pka-helper --version`). See [NOTICE.md](NOTICE.md)
for the third-party terms.
