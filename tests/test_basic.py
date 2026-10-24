import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "pka_helper", "_tools"))
import blend_pka as B

P = dict(B.DEFAULT, lipid_cap=3.0)

def test_acid_push_is_capped():
    assert abs(B.blend("GLU", 16.0, 5.0, P, 0.5) - 5.5) < 1e-9

def test_lipid_facing_acid_gets_larger_cap():
    assert abs(B.blend("GLU", 16.0, 5.0, P, 0.5, False, True) - 8.0) < 1e-9

def test_lipid_facing_lys_gets_larger_downward_cap():
    Q = dict(P, lipid_cap_base=4.0)
    assert abs(B.blend("LYS", -7.0, 10.3, dict(Q, lys_down_c0=1.0), 0.0, False, False) - 9.3) < 1e-9      # older rule, burial 0: bound 1 pK
    assert abs(B.blend("LYS", -7.0, 10.3, Q, 0.0, False, True) - 6.3) < 1e-9       # lipid-facing: bound 4 pK

def test_lipid_flag_does_not_change_arg_or_upward_lys():
    Q = dict(P, lipid_cap_base=4.0)
    assert abs(B.blend("ARG", 0.0, 12.0, Q, 0.0, False, True) - 11.0) < 1e-9
    assert abs(B.blend("LYS", 16.0, 9.0, Q, 0.0, False, True) - 10.0) < 1e-9

def test_pair_keeps_full_bound():
    assert B.blend("ASP", 16.0, 5.0, P, 0.5, True) > 5.5 + 1.0

def test_his_saturated_low_keeps_propka():
    assert B.blend("HIS", -7.0, 6.5, P, 0.5) == 6.5

def test_dum_detection(tmp_path):
    from pka_helper.cli import dum_z_values
    f = tmp_path / "a.pdb"
    f.write_text("HETATM    1  N   DUM     1       0.000   0.000  11.700\nHETATM    2  O   DUM     2       0.000   0.000 -11.700\n")
    assert dum_z_values(str(f)) == {11.7, -11.7}


def test_depth_factor_scaling():
    B.DEPTH_MIN, B.DEPTH_FULL, B.DEPTH_ON = 4.0, 10.0, True
    B.DEPTH.clear(); B.DEPTH[("LYS", 37, "j")] = 4.0; B.DEPTH[("LYS", 7, "x")] = 7.0; B.DEPTH[("LYS", 9, "x")] = 14.0
    assert B.depth_f(("LYS", 37, "j")) == 0.0 and abs(B.depth_f(("LYS", 7, "x")) - 0.5) < 1e-9 and B.depth_f(("LYS", 9, "x")) == 1.0
    assert B.depth_f(("LYS", 1, "z")) == 1.0          # depth unknown: no scaling
    B.DEPTH_ON = False
    assert B.depth_f(("LYS", 37, "j")) == 1.0
    B.DEPTH_ON = True; B.DEPTH.clear()

def test_interface_lys_is_not_pushed_down_far():
    Q = dict(P, lipid_cap_base=4.0)
    # 6ZBB LYS37 chain j: PROPKA 10.66, DelPhi 1.85, about 4 A inside the core -> burial-bound only (1 pK), not the full 4 pK
    assert abs(B.blend("LYS", 1.85, 10.66, Q, 0.0, False, True, 0.0) - 9.66) < 1e-9
    assert abs(B.blend("LYS", 1.85, 10.66, Q, 0.0, False, True, 1.0) - 6.66) < 1e-9
    assert abs(B.blend("LYS", 1.85, 10.66, Q, 0.0, False, True, 0.5) - 8.66) < 1e-9

def test_acid_cap_not_scaled_by_default():
    assert abs(B.blend("GLU", 16.0, 5.0, P, 0.5, False, True, 0.0) - 8.0) < 1e-9
    assert abs(B.blend("GLU", 16.0, 5.0, dict(P, depth_acids=True), 0.5, False, True, 0.0) - 5.5) < 1e-9


def test_surface_lys_not_pushed_down_by_default():
    assert abs(B.blend("LYS", 8.0, 10.3, P, 0.0, False, False) - 10.3) < 1e-9          # surface Lys (burial 0): keeps PROPKA
    assert abs(B.blend("LYS", 8.0, 10.3, P, 0.5, False, False) - 8.3) < 1e-9           # burial 0.5: down by k*burial = 2 pK (0.5*4)
    assert abs(B.blend("LYS", 16.0, 9.0, P, 0.0, False, False) - 10.0) < 1e-9          # upward shift unchanged (bound 1 pK)


def test_lipid_threshold_follows_core_eps(tmp_path):
    import lipid_auto as L
    assert L.lipid_eps_threshold(str(tmp_path)) == 10.0                         # no stack file: old threshold
    (tmp_path / "membrane_stack.txt").write_text("# x eps\n-30 80\n0 2.0\n30 80\n")
    assert L.lipid_eps_threshold(str(tmp_path)) == 10.0                         # eps-2 core: unchanged
    (tmp_path / "membrane_stack.txt").write_text("# x eps\n-30 80\n0 10.0\n30 80\n")
    assert L.lipid_eps_threshold(str(tmp_path)) == 15.0                         # --lipid-eps 10: core still detected


def test_lipid_eps_option_reaches_membrane_stage():
    C = _cli()
    src = open(C.__file__).read()
    assert 'argv += ["--eps-core", "%g" % a.lipid_eps]' in src and '"--lipid-eps"' in src


def test_his_rules_default_and_lipid_eps10():
    assert abs(B.blend("HIS", 2.5, 6.3, P, 0.0, False, False, 1.0) - 4.8) < 1e-9                 # default: down by <= 1.5
    Q = dict(P, his_down=0.0, his_lip=2.0)
    assert abs(B.blend("HIS", 2.5, 6.3, Q, 0.0, False, False, 1.0) - 4.3) < 1e-9                 # eps-10 rules: deep in the core, <= 2
    assert abs(B.blend("HIS", 2.5, 6.3, Q, 0.0, False, False, 0.0) - 6.3) < 1e-9                 # outside the core: PROPKA
    assert abs(B.blend("HIS", -1.0, 6.3, Q, 0.0, False, False, 1.0) - 6.3) < 1e-9                # saturated: PROPKA


def test_rule_set_follows_lipid_eps():
    import argparse
    C = _cli()
    a = argparse.Namespace(lipid_eps=2.0, acid_cap=None, lipid_cap=None, his_down=None, his_lip_cap=None, acid_up_scale=None)
    assert C.rule_value(a, "acid_cap") == 0.5 and C.rule_value(a, "his_down") == 1.5
    a.lipid_eps = 10.0
    assert C.rule_value(a, "acid_cap") == 2.0 and C.rule_value(a, "lipid_cap") == 5.0 and C.rule_value(a, "his_lip_cap") == 2.0
    a.acid_cap = 1.0
    assert C.rule_value(a, "acid_cap") == 1.0                                                    # an explicit option wins


def _cli():
    """pka_helper.cli from this source tree (not an installed copy)."""
    src = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    for m in [m for m in sys.modules if m == "pka_helper" or m.startswith("pka_helper.")]: del sys.modules[m]
    sys.path.insert(0, src)
    import pka_helper.cli as C
    assert os.path.abspath(C.__file__).startswith(src)
    return C


def _gated_pore(gap):
    """Synthetic channel: wall r 7-10 A, closed gate at z 16-17 below the mid-plane (z 20), square side slit gap x gap A from z 8.
    gap 6: open to the 2 A probe that grows the lipid, closed to the 3 A lipid-tail probe (like the BM2 helix splay); gap 12: open to both."""
    n = 41; z, y, x = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    x, y, z = x.transpose(2, 1, 0), y.transpose(2, 1, 0), z.transpose(2, 1, 0)      # arrays indexed [x, y, z]
    r = np.hypot(x - 20, y - 20)
    epsM = np.where((z >= 6) & (z <= 34), 2.0, 80.0)
    wall = (r >= 7) & (r <= 10) & (z >= 2) & (z <= 38)
    x0 = 20 - gap // 2
    wall &= ~((x >= x0) & (x < x0 + gap) & (y > 20) & (z >= 8) & (z < 8 + gap))       # the slit
    gate = (r < 7) & (z >= 16) & (z <= 17)
    epsP = np.where(wall | gate, 4.0, 80.0)
    return epsP, epsM


def test_pore_below_closed_gate_stays_water():
    import epsmap as E
    epsP, epsM = _gated_pore(6)
    old, _ = E.lipid_region(epsP, epsM, 80.0, 1e-3, 2.0, np.ones(3), axis=2, seed="mid")
    new, _ = E.lipid_region(epsP, epsM, 80.0, 1e-3, 2.0, np.ones(3), axis=2, seed="all")
    assert old[20, 20, 13]                       # old rule: the pore below the gate is lipid (reached through the 6 A slit)
    assert not new[20, 20, 13]                   # new rule: water
    assert not new[20, 20, 25] and not old[20, 20, 25]     # pore above the gate (mid-plane side): water either way
    assert new[2, 2, 20] and new[2, 2, 10]       # bulk lipid stays lipid at every depth
    assert (new & ~old).sum() == 0               # nothing becomes lipid that was not


def test_wide_opening_stays_lipid():
    import epsmap as E
    epsP, epsM = _gated_pore(12)                 # a 12 A opening: lipid can get in
    new, _ = E.lipid_region(epsP, epsM, 80.0, 1e-3, 2.0, np.ones(3), axis=2, seed="all")
    assert new[20, 20, 12]                       # pore beside the opening: lipid


def test_selenomethionine_read_as_methionine(tmp_path):
    C = _cli()
    p = tmp_path / "x.pdb"
    p.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
                 "HETATM    2  N   MSE A   2       1.000   0.000   0.000  1.00  0.00           N\n"
                 "HETATM    3 SE   MSE A   2       2.000   0.000   0.000  1.00  0.00          SE\n"
                 "HETATM    4  C1  LMU A 901       9.000   0.000   0.000  1.00  0.00           C\n")
    path, msg = C.standard_residues(str(p), str(tmp_path))
    L = open(path).read().splitlines()
    assert msg and "1 selenomethionine" in msg
    assert L[1].startswith("ATOM") and L[1][17:20] == "MET"
    assert L[2][12:16] == " SD " and L[2][17:20] == "MET" and L[2][76:78] == " S"
    assert L[3].startswith("HETATM") and L[3][17:20] == "LMU"          # other HETATM untouched
    q = tmp_path / "y.pdb"; q.write_text(L[0] + "\n")
    assert C.standard_residues(str(q), str(tmp_path)) == (str(q), None)


def test_salt_bridged_acids_exempt_from_solid_zone_by_default():
    C = _cli()
    i = C.UNIFORM.index("--solid-skip-partnered")
    assert float(C.UNIFORM[i + 1]) == 4.0


def test_tool_copies_identical():
    """The shared tools exist twice (pka-helper _tools and a delphipka-custom checkout next to it). They drifted once (the His caps
    of 0.1.7 were missing from the research copy). Skipped when no delphipka-custom checkout is present."""
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dst = os.path.join(os.path.dirname(root), "delphipka-custom", "tools"); tool = os.path.join(root, "tools", "sync_tools.py")
    if not (os.path.isdir(dst) and os.path.isfile(tool)): return       # release copy: no research checkout to compare with
    spec = importlib.util.spec_from_file_location("sync_tools", tool); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert m.differing(dst) == [], "run tools/sync_tools.py: %s differ" % m.differing(dst)
    assert m.differing_cpp(os.path.dirname(dst)) == [], "C++ sources differ: %s" % m.differing_cpp(os.path.dirname(dst))


def _bench(*p):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark", "structures", *p)


def test_structure_check_protonated_geometry():
    import structure_check as SC
    at, _, _, _ = SC._atoms(_bench("membrane", "3pox_trimer.pdb"))
    hits = {k for k, c, _ in SC.protonated_geometry(at)}
    assert ("ASP", 127, "A") in hits and ("ASP", 256, "B") in hits          # OmpF: CpHMD pKa > 8
    at, _, _, _ = SC._atoms(_bench("membrane", "4au5.pdb"))
    hits = {k for k, c, _ in SC.protonated_geometry(at)}
    assert ("ASP", 163, "A") not in hits                                    # salt-bridged to Lys300: not flagged
    at, _, _, _ = SC._atoms(_bench("soluble", "1PGB.pdb"))
    assert SC.protonated_geometry(at) == []


def test_structure_check_input_warnings(tmp_path):
    import structure_check as SC
    p = tmp_path / "m.pdb"
    p.write_text("REMARK   2 RESOLUTION.    3.80 ANGSTROMS.\nMODEL        1\n"
                 "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\nENDMDL\nMODEL        2\n"
                 "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\nENDMDL\n")
    codes = [m for k, c, m in SC.check(str(p)) if c == "input"]
    assert any("2 MODELs" in m for m in codes) and any("3.80" in m for m in codes)


def test_macroscopic_steps_of_identical_sites():
    import group_pka as G
    ph = np.arange(0.0, 14.01, 0.1)
    f = 1.0 / (1.0 + 10 ** (ph - 6.0))                      # a base with pKa 6, protonated fraction
    st = G.macro_general(ph, {"HIS0001A": f, "HIS0001B": f}, ["HIS0001A", "HIS0001B"])
    assert st[0] > st[1]                                    # highest first
    assert abs(st[0] - (6 + np.log10(2) * 0.0) - 0.0) < 0.6 and abs(np.mean(st) - 6.0) < 0.05   # two independent sites: mean of steps = site pKa


def test_sub_run_arguments():
    import argparse
    C = _cli()
    a = argparse.Namespace(_argv=["run", "x.pdb", "--np", "2", "--ensemble", "--ensemble-max", "3", "--out", "o", "--lipid-eps", "10",
                                  "--membrane-arg", "--lumen-seed", "mid"], _pdb_arg="x.pdb")
    assert C._sub_argv(a, "m1.pdb", "d1") == ["m1.pdb", "--out", "d1", "--np", "2", "--lipid-eps", "10", "--membrane-arg", "--lumen-seed", "mid"]
    a = argparse.Namespace(_argv=["run", "--np", "2", "x.pdb", "--membrane-scan", "--out=o"], _pdb_arg="x.pdb")
    assert C._sub_argv(a, "x.pdb", "s") == ["x.pdb", "--out", "s", "--np", "2"]


def test_alternate_locations_keep_first(tmp_path):
    C = _cli()
    p = tmp_path / "alt.pdb"
    p.write_text("ATOM      1  CA AASP A   5       0.000   0.000   0.000  0.50  0.00           C\n"
                 "ATOM      2  CA BASP A   5       0.300   0.000   0.000  0.50  0.00           C\n"
                 "ATOM      3  CA  GLY A   6       3.800   0.000   0.000  1.00  0.00           C\n")
    path, msg = C.standard_residues(str(p), str(tmp_path))
    L = [l for l in open(path).read().splitlines() if l.startswith("ATOM")]
    assert len(L) == 2 and L[0][16] == " " and "alternate" in msg
    assert C.unique_chains(path, str(tmp_path))[1] is None                 # no longer looks like a repeated chain


def test_summary_with_warning_on_some_chains_only():
    C = _cli()
    base = dict(PROPKA="4.0", DelPhiPKa="4.2", blended="4.1", state_at_pH7="deprotonated", confidence="uncertain", param_lo="4.0", param_hi="4.2",
                param_flips_call="0", note="", residue="ASP7")
    rows = [dict(base, chain="A", structure_warning="protonated-geometry"), dict(base, chain="B", structure_warning="")]
    for r in rows: r.update(_p=4.0, _b=4.1, _type="ASP", _num=7)
    txt = C.render_summary(rows, 7.0, "all")
    assert "check" in txt and "H-geom" in txt


def test_terminal_lists_only_propka_disagreements():
    """The terminal summary lists a group only when PROPKA and pka-helper are on opposite sides of the pH; groups that are merely close to
    the pH or carry a structure warning stay in the files."""
    C = _cli()
    def row(res, num, p, b, conf="confident", warn=""):
        return dict(residue="%s%d" % (res, num), chain="A", PROPKA=str(p), DelPhiPKa=str(b), blended=str(b), _p=p, _b=b, _type=res, _num=num,
                    state_at_pH7="protonated" if b > 7 else "deprotonated", confidence=conf, param_lo=str(b), param_hi=str(b),
                    param_flips_call="0", note="", structure_warning=warn)
    rows = [row("GLU", 108, 4.98, 7.98, "uncertain"),            # disagree -> listed
            row("LYS", 126, 7.91, 7.26, "uncertain"),            # agree, close to pH -> not listed
            row("HIS", 75, 6.01, 6.01, warn="deep-lipid"),       # agree, warning -> not listed
            row("ASP", 97, 5.97, 6.47)]
    txt = C.render_summary(rows, 7.0, "notable")
    assert "GLU108" in txt and "LYS126" not in txt and "HIS75" not in txt and "ASP97" not in txt
    assert "3 other groups" in txt and "1 of them is close" in txt


def test_no_membrane_uses_default_rules():
    import argparse
    C = _cli()
    a = argparse.Namespace(lipid_eps=10.0, no_membrane=True, acid_cap=None, lipid_cap=None, his_down=None, his_lip_cap=None)
    assert C.rule_value(a, "acid_cap") == 0.5 and C.rule_value(a, "his_down") == 1.5


def test_hbond_term_counts_and_shift():
    import hbond_term as H
    pqr = _bench("..", "structures", "membrane", "3pox_trimer.pdb")
    # count on a PQR-like reading of the PDB (same columns for coordinates)
    n = H.acceptor_counts(_bench("membrane", "3pox_trimer.pdb"))
    assert n.get(("ASP", 256, "B"), 0) >= 1                          # backbone O of Gln255
    assert H.shift_for(("ASP", 256, "B"), n, 0.97, 1.5) == 1.5 * min(n[("ASP", 256, "B")], 2)
    assert H.shift_for(("ASP", 256, "B"), n, 0.2, 1.5) == 0.0        # not buried: no shift
    assert H.shift_for(("HIS", 1, "A"), {("HIS", 1, "A"): 2}, 1.0, 1.5) == 0.0


def test_cofactor_pqr_lines(tmp_path):
    C = _cli()
    p = tmp_path / "c.pdb"
    p.write_text("ATOM      1  CA  LYS A 216       0.000   0.000   0.000  1.00  0.00           C\n"
                 "HETATM    2  C1  RET A 301       1.000   0.000   0.000  1.00  0.00           C\n"
                 "HETATM    3  C2  RET A 301       2.000   0.000   0.000  1.00  0.00           C\n"
                 "HETATM    4 ZN    ZN A 400       5.000   0.000   0.000  1.00  0.00          ZN\n"
                 "HETATM    5  O   HOH A 500       9.000   0.000   0.000  1.00  0.00           O\n"
                 "HETATM    6  O   DUM A 600       0.000   0.000  15.000  1.00  0.00           O\n")
    path, msg = C.apply_cofactors(str(p), ["RET", "ZN:+2"], str(tmp_path))
    L = open(path).read().splitlines()
    het = [l for l in L if l.startswith("HETATM")]
    assert len(het) == 3 and not any("HOH" in l or "DUM" in l for l in L)
    assert het[0].split()[-2:] == ["0.0000", "1.7000"] and het[2].split()[-2:] == ["2.0000", "1.3900"]


def test_hetero_policy(tmp_path):
    """HETATM groups: modified residues with a standard charge read as the parent, known charges kept on the right atoms, unknown ligands
    kept as uncharged volume, additives / waters / dummy atoms left out."""
    import hetero as HT
    C = _cli()
    p = tmp_path / "h.pdb"
    p.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
                 "HETATM    2  N   MLY A   2       1.000   0.000   0.000  1.00  0.00           N\n"
                 "HETATM    3  NZ  MLY A   2       2.000   0.000   0.000  1.00  0.00           N\n"
                 "HETATM    4  CH1 MLY A   2       3.000   0.000   0.000  1.00  0.00           C\n"
                 "HETATM    5  P   SEP A   3       4.000   0.000   0.000  1.00  0.00           P\n"
                 "HETATM    6  O1P SEP A   3       5.000   0.000   0.000  1.00  0.00           O\n"
                 "HETATM    7  O2P SEP A   3       6.000   0.000   0.000  1.00  0.00           O\n"
                 "HETATM    8  C1  RET A 301       7.000   0.000   0.000  1.00  0.00           C\n"
                 "HETATM    9 CA    CA A 400       8.000   0.000   0.000  1.00  0.00          CA\n"
                 "HETATM   10  S   SO4 A 500       9.000   0.000   0.000  1.00  0.00           S\n"
                 "HETATM   11  O   HOH A 600       9.000   1.000   0.000  1.00  0.00           O\n"
                 "HETATM   12  O   DUM A 700       0.000   0.000  15.000  1.00  0.00           O\n")
    plan = HT.classify(str(p))
    assert plan["MLY"][0] == "parent" and plan["SEP"][:2] == ("charged", -2.0) and plan["RET"][0] == "neutral"
    assert plan["CA"][0] == "ion" and plan["SO4"][0] == "drop" and "HOH" not in plan and "DUM" not in plan
    assert HT.classify(str(p), keep_ions=True)["CA"][:2] == ("charged", 2.0)
    plan = HT.classify(str(p), keep_ions=True)
    std, msg = C.standard_residues(str(p), str(tmp_path), plan)
    L = open(std).read().splitlines()
    lys = [l for l in L if l[17:20] == "LYS"]
    assert len(lys) == 2 and all(l.startswith("ATOM") for l in lys) and "MLY x1 -> LYS" in msg      # CH1 left out
    path, summ = HT.write_delphi_input(std, str(tmp_path), plan)
    het = {l[12:16].strip() + l[17:20].strip(): float(l[54:61]) for l in open(path) if l.startswith("HETATM")}
    assert abs(het["O1PSEP"] + 1.0) < 1e-6 and abs(het["PSEP"]) < 1e-6           # -2 on the two phosphate O present
    assert abs(het["CACA"] - 2.0) < 1e-6 and abs(het["C1RET"]) < 1e-6 and not any("SO4" in k for k in het)
    assert HT.classify(str(p), mode="drop")["RET"][0] == "drop" and HT.classify(str(p), ["RET:+1"])["RET"][:2] == ("charged", 1.0)
    assert HT.classify(str(p), drop=["RET"])["RET"][0] == "drop"


def test_schiff_base_lysine_found_and_held():
    import hetero as HT
    L = ["ATOM      1  NZ  LYS A 216       0.000   0.000   0.000  1.00  0.00           N",
         "ATOM      2  NZ  LYS A 172       9.000   0.000   0.000  1.00  0.00           N",
         "HETATM    3  C15 RET A 301       1.300   0.000   0.000  1.00  0.00           C"]
    keys = HT.schiff_base_lysines(L)
    assert keys == [("A", " 216 ")]
    out = HT.hold_schiff_base(L, keys)
    assert out[0].startswith("HETATM") and out[0][17:20] == "LYR" and out[1].startswith("ATOM")


def test_hetatm_charge_sign_survives():
    """DelPhiPKa reads a HETATM charge from substr(55, 7); a negative charge must keep its sign."""
    import hetero as HT
    l = HT.pqr_het("HETATM    1  P   CDL A 301      -1.000 -30.284-100.454", -1.0, 1.8)
    assert float(l[55:62]) == -1.0
    dl = l[:55] + " 0.0000" + l[62:]                       # what DelPhiPKa passes on to DelPhi (charge zeroed)
    assert abs(float(dl[62:70]) - 1.8) < 1e-9 and float(dl[54:62]) == 0.0


def test_partner_check(tmp_path):
    """A buried acid spared the low-dielectric zone by a partner that the first pass finds neutral loses the exemption."""
    C = _cli()
    (tmp_path / "solid_partners.txt").write_text("GLU325A HIS322A\nASP85A LYS216A\nGLU9B HIS7B ARG11B\n")
    (tmp_path / "pKa.csv").write_text(" ResName  pKa\nHIS0322A  2.43\nLYS0216A  12.17\nHIS0007B  3.0\nARG0011B  12.0\n")
    neutral, lost = C.neutral_partners(str(tmp_path), 7.0)
    assert lost == {"GLU325A"} and neutral == {"HIS322A", "HIS7B"}       # GLU9B keeps its charged Arg partner
    # default: partners judged by the first-pass blended values; a retinal Schiff-base Lys always counts as charged
    (tmp_path / "solid_partners.txt").write_text("GLU325A HIS322A\nASP85A HIS57A LYS225A\nASP121A HIS87A\n")
    (tmp_path / "pass1.csv").write_text("residue,chain,PROPKA,DelPhiPKa,saturated,blended\nHIS322,A,6.0,2.4,0,5.75\nHIS57,A,6.9,1.8,0,5.37\n"
                                        "LYS225,A,9.0,3.0,0,6.50\nHIS87,A,7.2,3.1,0,7.24\n")
    neutral, lost = C.neutral_partners(str(tmp_path), 7.0, str(tmp_path / "pass1.csv"), {"LYS225A"})
    assert lost == {"GLU325A"} and neutral == {"HIS322A", "HIS57A"}


def test_tautomer_swap_and_select(tmp_path):
    import tautomer as TT
    p = tmp_path / "a.pdb"
    p.write_text("ATOM      1  OD1 ASP A  96       0.000   0.000   0.000  1.00  0.00           O\n"
                 "ATOM      2  OD2 ASP A  96       2.200   0.000   0.000  1.00  0.00           O\n"
                 "ATOM      3  OD1 ASP A  97       9.000   0.000   0.000  1.00  0.00           O\n")
    q = tmp_path / "b.pdb"; TT.swapped_pdb(str(p), str(q), [("ASP", 96, "A")])
    L = open(q).read().splitlines()
    assert L[0][12:16] == " OD2" and L[1][12:16] == " OD1" and L[0][30:38] == p.read_text().splitlines()[0][30:38]   # names swap, coordinates stay
    assert L[2][12:16] == " OD1"                                                                                         # other residues untouched


def test_acid_up_scale():
    Q = dict(P, acid_up=0.5, acid_up_w=0.3)
    assert abs(B.blend("ASP", 5.0, 4.0, Q, 0.5, False, False) - 4.3) < 1e-9          # push 1.0 -> 0.3
    assert abs(B.blend("ASP", 8.0, 4.0, Q, 0.5, False, False) - 4.5) < 1e-9          # push 4.0 -> 1.2, capped at 0.5
    assert abs(B.blend("ASP", 3.0, 4.0, Q, 0.5, False, False) - 3.0) < 1e-9          # downward pushes unchanged
    assert abs(B.blend("ASP", 5.0, 4.0, Q, 0.5, True, False) - 5.0) < 1e-9           # carboxylate pair: unscaled, full bound


def test_error_model():
    s_surf = B.sigma_of("GLU", 4.4, 4.4, 0.1, False); s_bur = B.sigma_of("GLU", 4.4, 4.4, 0.9, False)
    s_mem = B.sigma_of("GLU", 4.4, 4.4, 0.1, True); s_dis = B.sigma_of("GLU", 9.4, 4.4, 0.1, False)
    assert 0.5 < s_surf < 0.6 and s_bur > s_surf and s_mem > s_surf and s_dis > s_surf
    assert abs(B.sigma_of("ASP", -7.0, 1.0, 0.0, False) - B.sigma_of("ASP", 0.0, 1.0, 0.0, False)) < 1e-12   # DelPhi clipped to 0..14
    assert abs(B.p_protonated(7.0, 1.0, 7.0) - 0.5) < 1e-12 and B.p_protonated(9.0, 1.0, 7.0) > 0.97 and B.p_protonated(5.0, 1.0, 7.0) < 0.03
