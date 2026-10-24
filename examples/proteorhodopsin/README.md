# Example: proteorhodopsin (PDB 2L6X)

Green proteorhodopsin, a light-driven proton pump: one chain, seven transmembrane helices. The file is the OPM-oriented structure
(membrane normal along z, with the `DUM` atoms that mark the membrane boundaries).

    pka-helper run proteorhodopsin_2l6x.pdb

About one minute on two cores. Output goes to `proteorhodopsin_2l6x_pka/`; `expected_results/` holds the results of a reference run
(pka-helper 0.1.1a1) and what was printed on screen, so you can compare. Small differences (a few hundredths of a pK) between machines are normal.

Things to look at:

* **Asp97**, the primary proton acceptor. Its pKa has been measured at about 6.4 (the value used in `benchmark/`). PROPKA gives 5.97,
  pka-helper 6.47. DelPhi alone gives 10.0; the blend only allows part of such an upward push for an acid.
* **Glu108**, the proton donor, is lipid-facing in this structure. DelPhi pushes it up strongly, and it is one of the two groups printed on screen (PROPKA says deprotonated at pH 7, pka-helper protonated),
  with a wide `limits varied` range (`!` = its pH 7 state flips within that range). Asp227, the other printed group, is pulled down
  almost 4 pK by DelPhi (PROPKA 8.23, pka-helper 4.41); whether that drop is right is the real question for it.
* **His75** gets a `deep-lipid` warning: it faces the lipid 7 Å inside the core in this model, so its value depends on the membrane position.
* The retinal is bound to Lys231 as one HETATM residue (LYR). pka-helper keeps it in the calculation with the +1 charge of the protonated
  Schiff base, and says so at the start of the run. Being one residue with the retinal, Lys231 is not titrated.
