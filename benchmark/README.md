# pka-helper benchmark

One command rebuilds every accuracy number quoted in the main README:

    python3 benchmark/run_benchmark.py                          # all suites with the default settings (about 1-1.5 h on 2 cores)
    python3 benchmark/run_benchmark.py --suite membrane         # membrane | peptides | soluble | locked
    python3 benchmark/run_benchmark.py --tag eps10 -- --lipid-eps 10   # options after -- go to every `pka-helper run`

It calls the installed `pka-helper run` on each structure (set `PKA_HELPER="python3 -m pka_helper.cli"` or
`PKA_HELPER_FROM_SOURCE=1` to use a source tree), keeps every run in `runs/<tag>/`, reuses finished runs, and writes
`runs/<tag>/report.md` plus one `<suite>_sites.csv` per suite with the per-site values.

## Suites

| suite | sites | references | role |
|---|---|---|---|
| membrane | 91 (69 without the second HV1 structure) in OmpF, HV1 (AlphaFold model and 5OQK), NhaA (4AU5), PR, EmrE, BM2 | constant-pH MD (OmpF, HV1, NhaA); experiment (PR Asp97, EmrE Glu14, BM2 His27) | development set: the blend rules were built on it |
| peptides | 15 single titratable groups in a TM helix (GWALP23) | experiment + constant-pH MD | calibration of `--lipid-eps 10` |
| soluble | 201 sites in 22 PKAD-R structures, run with `--no-membrane` | experiment (PKAD-R, Main, wild type, no warning) | independent check: never used to tune anything |
| locked | bacteriorhodopsin Asp85 / Asp96 / Asp212, LacY Glu325 (4OAA, the measured G46W/G262W mutant); stepwise: bR Glu194/204, M2 His37 (2L0J, 3LBW) | experiment | independent test: **report only, never tune on it**. A site is scored only if its structure represents the measured state (`structure_matches_measurement` in sites_locked.csv); others are listed with the reason |

Notes on the references
* BM2 His27 7.9 is the first of four stepwise pKas of the His27 tetrad (7.9 / 5.9 / 5.9 / 5.8, mean 6.4). pka-helper gives one pKa per
  residue, so the report also prints the tetrad's stepwise values from the DelPhi titration for a like-with-like comparison.
* NhaA uses 4AU5 (Asp163-Lys300 salt bridge present, as in the simulations that produced the references), not 1ZCD.
* bacteriorhodopsin is run as a user would run it: the retinal (HETATM) is kept as an uncharged volume (if it were left out, its pocket
  would become water), and the Schiff-base lysine (Lys216, an ATOM residue) is titrated as usual.
* The M2 references are from DMPC/DMPG; in viral-envelope membranes the steps are 7.6 / 6.8 / 4.9 / 4.2.

Scoring: the mean over the listed chains; DelPhi values are clipped to 0..14; a reference written `<x` or `>x` only counts a prediction on
the wrong side of x. "right at pH 7" = same side of 7 as the reference. The soluble and membrane summaries give a bootstrap interval of
(pka-helper - PROPKA) over families / proteins.

## Files
* `sites_membrane.csv`, `sites_peptides.csv`, `sites_soluble.csv`, `sites_locked.csv`, `groups_locked.csv` - the references
* `structures/` - the input structures exactly as used (OPM-oriented where a membrane is needed)
* `run_benchmark.py` - runs and scores
