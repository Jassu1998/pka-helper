# Independent membrane test set: status (2026-10-24)

Goal: measured, residue-specific pKas in membrane proteins that were never used to develop pka-helper, to test the membrane claim.
Candidates are in `membrane_test_candidates.csv`.

## Inclusion rules
1. A measured value (titration, NMR, FTIR, absorption) assigned to one residue. Apparent pKas of a function (gating, transport, DEER
   conformational switches) or of unassigned sites are not included.
2. Asp, Glu, His, Lys or Arg (DelPhiPKa titrates only these).
3. Measured in lipid bilayers, liposomes, native membranes or detergent; not in organic solvent.
4. A structure of the measured protein (wild type or the measured mutant) in the measured state exists.
5. Not used to develop pka-helper (development set, TM peptides, locked set).

Tiers: **A** = direct single-site value, primary paper read; **B** = value from the abstract or a curated database (PKAD-R) whose method
or assignment still has to be checked; **C** = candidate from a secondary listing, primary paper not yet reached.

## Result of the search
Measured residue-level pKas in membrane proteins are scarce. The authors of MPKaDB (a database of 3.3 million *computed* membrane-protein
pKas, bioRxiv 2025 / J Mol Biol 2026) say the same: experimental values "remain scarce". PKAD-R, the largest curated experimental set,
holds 9 membrane-protein entries in 7 proteins, and most fail the rules above.

Found: 7 candidates in 5 proteins (2 tier A/B in ESR, 3 tier B in GR and GLIC, 2 tier C). That is far short of the 30-50 sites a paper
would need.

Excluded, with reason:
* already used: bacteriorhodopsin Asp85/96/212, LacY Glu325, M2 His37 (locked set); proteorhodopsin Asp97, EmrE Glu14, BM2 His27
  (development set); GWALP/GKALP peptides (calibration)
* ATP synthase subunit c (1A91): measured in an organic solvent mixture
* photosystem II PsbO (3WU2): water-soluble extrinsic subunit; complex too large to run here
* DsbD C-terminal domain: isolated soluble domain, and Cys
* EmrE Glu25: measured in the E14Q mutant
* retinal Schiff bases (bR Lys216, SRII), rhodopsin Glu113 mutants: Schiff-base chemistry not modelled / no structures of the mutants
* KcsA Glu71: only computed values found; KcsA activation, SERT, Spinster: apparent, not residue-specific
* Cys and Tyr values: not titrated by DelPhiPKa

## Results (default settings)
| site | measured | PROPKA | DelPhi | pka-helper |
|---|---|---|---|---|
| ESR Asp85 (4HYJ) | 2.3 | 1.87 | 6.47 | 2.37 |
| Gloeobacter rhodopsin Asp121 (6NWD, chains A/B) | ~5.9 | 4.62 | 6.03 | 5.04 |
| GLIC Glu35 (4HFI, 5 chains) | 5.8 | 6.85 | 5.66 | 5.53 |
| GLIC Glu181 (4HFI, 5 chains) | 5.5 | 6.27 | 6.57 | 5.98 |
| **mean error, these 4** | | **0.88** | 1.38 | **0.42** |
| ESR His57, coupled (tentative) | ~6.5 | 6.87 | 1.75 | 5.37 |
| cytochrome c oxidase Glu286 (1M56), tier C | ~9.4 | 8.66 | 12.83 | 9.16 |
| PR Asp227 (2L6X), tier C | ~3.0 | 8.23 | 4.41 | 4.41 |

pka-helper is closer than PROPKA on all four tier A/B single-site values; PROPKA is closer for the tentative His57 transition. Four sites
cannot support a claim, but the direction agrees with the development data. Cytochrome c oxidase with its metal centres kept
(`--keep-ions`): DelPhi 11.23, pka-helper 9.16. Note for ESR: the retinal (RET) is a separate residue, so the Schiff-base Lys225 is titrated as an ordinary lysine and
comes out partly neutral (pka-helper 7.41; real protonated Schiff bases have pKa > 12). Holding it charged (`--hold-schiff-base`) lowers
DelPhi's Asp85 only from 6.47 to 5.88 and leaves the final value at 2.37: the missing Schiff-base charge is not why DelPhi puts Asp85 high.
In bacteriorhodopsin (1M0L) DelPhi already gives the Schiff base 11.8-13.0 (measured 12.2-13.3).

## Salt-bridge partner check
The new partner check does not fire on ESR, GR or GLIC: the values above are unchanged. It fires on cytochrome c oxidase Asp407 and
Glu539 (partners His411 and His127 come out neutral; no measured values), final values Asp407 2.12 -> 1.10, Glu539 2.75 -> 3.04; Glu286
stays at 9.16.

## What is needed
1. Primary papers for the tier B and C entries (several publisher and PMC pages are blocked from here).
2. More sites. Places to look:
   * rhodopsin counterion titrations (xanthorhodopsin, KR2, thermophilic rhodopsin, channelrhodopsins), each with a crystal structure;
   * NMR titrations in nanodiscs or bicelles (OmpG His231/His261, VDAC1, small transporters);
   * FTIR assignments in transporters (LacY, CLC, NhaA mutants);
   * collaboration: new NMR/FTIR measurements on a handful of chosen sites would be the strongest test.
3. Meanwhile, a second, clearly separate test against constant-pH MD for proteins not used in development (e.g. GLIC, published
   CpHMD) would test the method's internal consistency, but not its accuracy against experiment.

## Partner-check test cases
| site | structure | measured | PROPKA | DelPhi pass 1 -> 2 | pka-helper | check fires? |
|---|---|---|---|---|---|---|
| alpha-lytic protease Asp102 (His57 partner) | 1SSX (0.83 A, pH 8), run --no-membrane | <= 1.5 (13C NMR, Everill et al. JACS 2012) | 3.50 | 2.49 -> 0.00* | 0.00 | yes (His57 6.51 in pass 1) |
| xanthorhodopsin Asp96 (His62, Schiff-base Lys240) | 3DDL (OPM) | ~6 (not verified; papers blocked) | 5.16 | 7.02 | 5.66 | no (His62 7.14, Lys240 12.85) |
The alpha-lytic case is the one where a wrong second pass would do harm: a buried acid next to a His that comes out neutral, but held
charged by several H bonds. The second pass lowers it further, the right direction.
