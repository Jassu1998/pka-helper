# Independent membrane test set

Measured, residue-specific pKas in membrane proteins that were not used to develop pka-helper. The sites and their sources are listed in
`membrane_test_candidates.csv`; the structures are in `structures/membrane_test/`.

## Inclusion rules
1. A measured value (titration, NMR, FTIR, absorption) assigned to one residue. Apparent pKas of a function (gating, transport,
   conformational switches) or of unassigned sites are not included.
2. Asp, Glu, His, Lys or Arg (the groups DelPhiPKa titrates).
3. Measured in lipid bilayers, liposomes, native membranes or detergent; not in organic solvent.
4. A structure of the measured protein (wild type or the measured mutant) in the measured state exists.
5. Not used to develop pka-helper (development set, transmembrane peptides, locked set).

Tiers: **A** = direct single-site value from the primary paper; **B** = value from an abstract or a curated database (PKAD-R) whose
method or assignment is less certain; **C** = value from a secondary compilation, not checked against the primary paper.

Measured residue-level pKas in membrane proteins are scarce: PKAD-R, the largest curated experimental set, holds 9 membrane-protein entries
in 7 proteins, and most fail the rules above. Excluded, with reason:
* used in development or the locked set: bacteriorhodopsin Asp85/96/212, LacY Glu325, M2 His37, proteorhodopsin Asp97, EmrE Glu14,
  BM2 His27, GWALP/GKALP peptides
* ATP synthase subunit c (1A91): measured in an organic solvent mixture
* photosystem II PsbO (3WU2): water-soluble extrinsic subunit of a very large complex
* DsbD C-terminal domain: isolated soluble domain, and Cys
* EmrE Glu25: measured in the E14Q mutant
* retinal Schiff bases (bR Lys216, SRII) and rhodopsin Glu113 mutants: Schiff-base chemistry not modelled, or no structure of the mutant
* KcsA Glu71: no measured residue-specific value; KcsA activation, SERT, Spinster: apparent, not residue-specific
* Cys and Tyr: not titrated

## Results (default settings)
| site | measured | PROPKA | DelPhi | pka-helper |
|---|---|---|---|---|
| ESR Asp85 (4HYJ) | 2.3 | 1.87 | 6.47 | 2.37 |
| Gloeobacter rhodopsin Asp121 (6NWD, chains A/B) | ~5.9 | 4.62 | 6.03 | 5.04 |
| GLIC Glu35 (4HFI, 5 chains) | 5.8 | 6.85 | 5.66 | 5.53 |
| GLIC Glu181 (4HFI, 5 chains) | 5.5 | 6.27 | 6.57 | 5.98 |
| **mean error, these 4** | | **0.88** | 1.38 | **0.42** |
| ESR His57, coupled (tentative assignment) | ~6.5 | 6.87 | 1.75 | 5.37 |
| cytochrome c oxidase Glu286 (1M56), tier C | ~9.4 | 8.66 | 12.83 | 9.16 |

pka-helper is closer than PROPKA on all four tier A/B single-site values; PROPKA is closer for the tentatively assigned His57 transition.
Four sites cannot support a general claim, but the direction agrees with the development data. All seven values lie within pka-helper's
reported ±1 sigma.

Notes:
* Cytochrome c oxidase with its metal centres kept (`--keep-ions`): DelPhi 11.23, pka-helper 9.16.
* ESR: the retinal is a separate residue (RET), so the Schiff-base Lys225 is titrated as an ordinary lysine and comes out partly neutral
  (7.41; protonated Schiff bases have pKa > 12). Holding it charged (`--hold-schiff-base`) lowers DelPhi's Asp85 only from 6.47 to 5.88 and
  leaves the final value at 2.37.
* The salt-bridge partner check does not trigger in ESR, Gloeobacter rhodopsin or GLIC. In cytochrome c oxidase it triggers for Asp407 and
  Glu539 (partners His411 and His127 come out neutral; no measured values); Glu286 is unaffected.

## Partner-check test cases
| site | structure | measured | PROPKA | DelPhi, first -> second pass | pka-helper | check triggers? |
|---|---|---|---|---|---|---|
| alpha-lytic protease Asp102 (partner His57) | 1SSX (0.83 Å, pH 8), `--no-membrane` | ≤ 1.5 (13C NMR; Everill, Sudmeier & Bachovchin, JACS 2012) | 3.50 | 2.49 -> 0.00 | 0.00 | yes (His57 6.51) |
| xanthorhodopsin Asp96 (His62, Schiff-base Lys240) | 3DDL (OPM) | not used (value not checked) | 5.16 | 7.02 | 5.66 | no (His62 7.14, Lys240 12.85) |

alpha-lytic protease is the case where a wrong second pass would do harm: a buried acid next to a His that comes out neutral, but held
charged by several hydrogen bonds. The second pass moves it further down, the right direction.
