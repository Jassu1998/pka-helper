#!/usr/bin/env python3
"""
membrane_shape.py - fit a (curved) membrane mid-surface to lipid coordinates and evaluate signed distances to it.

A flat slab is described by one number (its mid-plane).  A vesicle, a tube, a caveola or the dome around a mechanosensitive channel
is not.  Everything the membrane model needs -- the dielectric profile (core | head-group zone | water), the lumen-water seed and
the layered Born correction -- depends on the position only through the SIGNED DISTANCE d to the membrane mid-surface (d > 0 on
the side the surface normal points to).  For a plane d = z - z_mid, so the old behaviour is the special case of a plane.

Shapes
    plane     point p0, normal n
    sphere    centre c, mid-surface radius R               d = |x - c| - R
    cylinder  point p0 on the axis, axis a, radius R       d = distance from the axis - R
    poly      height field w = f(u, v) of degree 2-4 in a local frame (any gently curved or saddle-shaped surface);
              the distance is found by nearest-point search on a dense sampling of the surface

Fitting (CLI, python3 membrane_shape.py --help)
    Input: the phosphate atoms (or any chosen atoms) of both leaflets, from a PDB with lipids (CHARMM-GUI / MD snapshot),
    or an xyz text file.  The two leaflets are fitted TOGETHER as two parallel (concentric) shells at -/+ T/2 from the mid-surface,
    so the output is the mid-surface and the phosphate-phosphate thickness T.  Candidate shapes are compared by RMSE and BIC and the
    simplest adequate one is suggested.  Without lipid coordinates describe the surface yourself (--manual).

The JSON written here is read by epsmap.py (membrane / combine --shape-file) and membrane_pka.py (--membrane-shape).
"""
import argparse
import json
import math
import os
import sys

import numpy as np

# --------------------------------------------------------------------------------------------------------------- shapes


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _frame(normal):
    """orthonormal (e1, e2, e3=normal)"""
    n = _unit(normal)
    t = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = _unit(t - n * (t @ n))
    e2 = np.cross(n, e1)
    return e1, e2, n


def monomials(u, v, deg):
    cols = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            cols.append((u ** i) * (v ** j))
    return np.stack(cols, axis=-1)


def monomial_list(deg):
    return [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]


class Shape:
    """kind + parameters; .dist(X) gives the signed distance to the mid-surface for an (N, 3) array."""

    def __init__(self, kind, params, thickness_pp=None, info=None):
        self.kind = kind
        self.p = params
        self.thickness_pp = thickness_pp
        self.info = info or {}
        self._tree = None

    # ---- evaluation
    def dist(self, X):
        X = np.asarray(X, float).reshape(-1, 3)
        k, p = self.kind, self.p
        if k == "plane":
            return (X - np.asarray(p["point"])) @ _unit(p["normal"])
        if k == "sphere":
            return np.linalg.norm(X - np.asarray(p["center"]), axis=1) - p["radius"]
        if k == "cylinder":
            a = _unit(p["axis"])
            q = X - np.asarray(p["point"])
            return np.linalg.norm(q - np.outer(q @ a, a), axis=1) - p["radius"]
        if k == "poly":
            return self._poly_dist(X)
        raise ValueError("unknown shape kind %r" % k)

    def _poly_setup(self):
        p = self.p
        o = np.asarray(p["origin"]); e1, e2, e3 = [np.asarray(p[x]) for x in ("e1", "e2", "e3")]
        deg = p["degree"]; c = np.asarray(p["coef"]); s = p.get("scale", 100.0)
        u0, u1, v0, v1 = p["extent"]
        h = 0.4
        uu = np.arange(u0, u1 + h, h); vv = np.arange(v0, v1 + h, h)
        U, V = np.meshgrid(uu, vv, indexing="ij")
        W = monomials(U / s, V / s, deg) @ c
        pts = o + U[..., None] * e1 + V[..., None] * e2 + W[..., None] * e3
        # normals from finite differences of the height field
        Wu = np.gradient(W, h, axis=0); Wv = np.gradient(W, h, axis=1)
        nrm = e3 - Wu[..., None] * e1 - Wv[..., None] * e2
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        from scipy.spatial import cKDTree
        self._tree = cKDTree(pts.reshape(-1, 3))
        self._nrm = nrm.reshape(-1, 3)
        self._pts = pts.reshape(-1, 3)

    def _poly_dist(self, X):
        if self._tree is None:
            self._poly_setup()
        d, idx = self._tree.query(X, workers=-1)
        side = np.einsum("ij,ij->i", X - self._pts[idx], self._nrm[idx])
        return np.where(side >= 0, d, -d)

    def poly_height(self, u, v):
        p = self.p
        return monomials(np.asarray(u) / p.get("scale", 100.0), np.asarray(v) / p.get("scale", 100.0), p["degree"]) @ np.asarray(p["coef"])

    # ---- description
    def curvature_radii(self):
        if self.kind == "sphere":
            return [self.p["radius"], self.p["radius"]]
        if self.kind == "cylinder":
            return [self.p["radius"], float("inf")]
        if self.kind == "plane":
            return [float("inf"), float("inf")]
        return None

    def describe(self):
        k, p = self.kind, self.p
        t = "" if self.thickness_pp is None else ", P-P thickness %.1f A" % self.thickness_pp
        if k == "plane":
            return "plane through %s, normal %s%s" % (np.round(p["point"], 2).tolist(), np.round(_unit(p["normal"]), 3).tolist(), t)
        if k == "sphere":
            return "sphere centre %s, mid-surface radius %.1f A%s" % (np.round(p["center"], 1).tolist(), p["radius"], t)
        if k == "cylinder":
            return "cylinder axis through %s direction %s, mid-surface radius %.1f A%s" % (
                np.round(p["point"], 1).tolist(), np.round(_unit(p["axis"]), 3).tolist(), p["radius"], t)
        return "degree-%d height field in a local frame (normal %s)%s" % (p["degree"], np.round(p["e3"], 3).tolist(), t)

    def to_json(self, path=None, **extra):
        def cv(x):
            if isinstance(x, np.ndarray): return x.tolist()
            if isinstance(x, (np.floating, np.integer)): return x.item()
            if isinstance(x, dict): return {k: cv(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)): return [cv(v) for v in x]
            return x
        d = dict(kind=self.kind, params=cv(self.p), thickness_pp=self.thickness_pp, info=cv(self.info), **cv(extra))
        if path:
            json.dump(d, open(path, "w"), indent=1)
        return d

    @staticmethod
    def from_json(path_or_dict):
        d = json.load(open(path_or_dict)) if isinstance(path_or_dict, str) else path_or_dict
        return Shape(d["kind"], d["params"], d.get("thickness_pp"), d.get("info"))


def load_shape(path):
    return Shape.from_json(path)


def distance_on_grid(shape, origin, delta, n, chunk=2_000_000):
    """signed distance on a regular grid (x slowest, z fastest); float32 array of shape n"""
    ax = [origin[i] + delta[i] * np.arange(n[i]) for i in range(3)]
    out = np.empty(tuple(n), np.float32)
    # slab by slab along x so memory stays bounded
    yy, zz = np.meshgrid(ax[1], ax[2], indexing="ij")
    yz = np.stack([yy.ravel(), zz.ravel()], 1)
    per = max(1, chunk // len(yz))
    for i0 in range(0, n[0], per):
        i1 = min(n[0], i0 + per)
        X = np.empty(((i1 - i0) * len(yz), 3))
        X[:, 0] = np.repeat(ax[0][i0:i1], len(yz))
        X[:, 1:] = np.tile(yz, (i1 - i0, 1))
        out[i0:i1] = shape.dist(X).reshape(i1 - i0, n[1], n[2]).astype(np.float32)
    return out


# --------------------------------------------------------------------------------------------------------------- fitting
def _kmeans1d(d, iters=30):
    c1, c2 = np.percentile(d, 25), np.percentile(d, 75)
    for _ in range(iters):
        lab = np.abs(d - c2) < np.abs(d - c1)
        if lab.all() or (~lab).all():
            break
        n1, n2 = d[~lab].mean(), d[lab].mean()
        if abs(n1 - c1) + abs(n2 - c2) < 1e-6:
            break
        c1, c2 = n1, n2
    return c1, c2, lab


def _lsq(fun, x0):
    from scipy.optimize import least_squares
    return least_squares(fun, x0, x_scale="jac", method="lm" if len(x0) <= len(x0) else "trf")


def _refine(P, build, x0):
    """least squares on the distance to the nearest leaflet shell: r = |d| - T/2 (no leaflet labels needed); x = [shape params..., T]"""
    from scipy.optimize import least_squares
    f = lambda x: np.abs(build(x[:-1]).dist(P)) - abs(x[-1]) / 2
    r = least_squares(f, x0, x_scale="jac")
    return r.x, float(np.sum(r.fun ** 2))


def _poly2_curv(P):
    """local quadratic fit -> (origin, e1, e2, e3, k1, k2, v1, v2, T); k are curvatures along e3, v principal directions"""
    sh, _ = fit_poly(P, 2)
    p = sh.p; sc = p["scale"]; coef = dict(zip(monomial_list(2), p["coef"]))
    H = np.array([[2 * coef[(2, 0)], coef[(1, 1)]], [coef[(1, 1)], 2 * coef[(0, 2)]]]) / sc ** 2
    w, V = np.linalg.eigh(H)
    e1, e2, e3, o = [np.asarray(p[k]) for k in ("e1", "e2", "e3", "origin")]
    vecs = [V[0, i] * e1 + V[1, i] * e2 for i in range(2)]
    return o, e3, w, vecs, sh.thickness_pp


def _ang(v):
    v = _unit(v)
    return math.acos(np.clip(v[2], -1, 1)), math.atan2(v[1], v[0])


def _dirv(th, ph):
    return np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])


def fit_plane(P):
    o, e3, k, vecs, T = _poly2_curv(P)
    c = P.mean(0)
    def build(t):
        n = _dirv(t[0], t[1]); return Shape("plane", dict(point=(c + n * t[2]).tolist(), normal=n.tolist()), None)
    th, ph = _ang(e3)
    x, rss = _refine(P, build, np.array([th, ph, (o - c) @ e3, T]))
    sh = build(x[:-1]); sh.thickness_pp = float(abs(x[-1]))
    return sh, 4


def fit_sphere(P):
    o, e3, k, vecs, T = _poly2_curv(P)
    best = None
    km = (k[0] + k[1]) / 2
    for sg in ([1.0] if abs(km) > 1e-4 else [1.0, -1.0]):
        R0 = min(1.0 / max(abs(km), 1e-4), 3000.0)
        for side in (+1, -1):
            c0 = o + side * e3 * R0 * (1 if km >= 0 else -1) * -1 * -1
            build = lambda t: Shape("sphere", dict(center=t[:3].tolist(), radius=float(t[3])), None)
            try:
                x, rss = _refine(P, build, np.concatenate([c0, [R0, T]]))
            except Exception:
                continue
            if best is None or rss < best[0]:
                best = (rss, x)
        break
    x = best[1]
    sh = Shape("sphere", dict(center=x[:3].tolist(), radius=float(abs(x[3]))), float(abs(x[4])))
    return sh, 5


def fit_cylinder(P):
    o, e3, k, vecs, T = _poly2_curv(P)
    best = None
    ibig = int(np.argmax(np.abs(k)))
    axis0 = vecs[1 - ibig]
    R0 = min(1.0 / max(abs(k[ibig]), 1e-4), 3000.0)
    c = P.mean(0)
    def build(t):
        a = _dirv(t[0], t[1]); b1, b2, _ = _frame(a)
        return Shape("cylinder", dict(point=(c + t[2] * b1 + t[3] * b2).tolist(), axis=a.tolist(), radius=float(t[4])), None)
    th, ph = _ang(axis0)
    a0 = _dirv(th, ph); b1, b2, _ = _frame(a0)
    for side in (+1, -1):
        p0 = o + side * e3 * R0
        x0 = np.array([th, ph, (p0 - c) @ b1, (p0 - c) @ b2, R0, T])
        try:
            x, rss = _refine(P, build, x0)
        except Exception:
            continue
        if best is None or rss < best[0]:
            best = (rss, x)
    x = best[1]
    sh = build(x[:-1]); sh.p["radius"] = abs(sh.p["radius"]); sh.thickness_pp = float(abs(x[-1]))
    return sh, 6


def fit_poly(P, deg):
    c = P.mean(0)
    u, s, vt = np.linalg.svd(P - c, full_matrices=False)
    e1, e2, e3 = vt[0], vt[1], vt[2]
    if np.linalg.det(np.stack([e1, e2, e3])) < 0:
        e2 = -e2
    q = (P - c)
    U, V, W = q @ e1, q @ e2, q @ e3
    sc = 100.0
    M = monomials(U / sc, V / sc, deg)
    ridge = 1e-6 * np.eye(M.shape[1])
    coef = np.linalg.solve(M.T @ M + ridge, M.T @ W)
    lab = (W - M @ coef) > np.median(W - M @ coef)
    s_i = np.where(lab, 1.0, -1.0)
    T = 0.0
    for _ in range(8):
        # slope factor g = sqrt(1 + |grad f|^2)
        h = 1e-3
        fu = (monomials((U + h * sc) / sc, V / sc, deg) @ coef - monomials((U - h * sc) / sc, V / sc, deg) @ coef) / (2 * h * sc)
        fv = (monomials(U / sc, (V + h * sc) / sc, deg) @ coef - monomials(U / sc, (V - h * sc) / sc, deg) @ coef) / (2 * h * sc)
        g = np.sqrt(1 + fu ** 2 + fv ** 2)
        A = np.hstack([M, (s_i * g / 2)[:, None]])
        sol = np.linalg.solve(A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ W)
        coef, T = sol[:-1], sol[-1]
        res = (W - M @ coef) / g
        s_i = np.where(res > 0, 1.0, -1.0)
    ext = [U.min() - 40, U.max() + 40, V.min() - 40, V.max() + 40]
    shape = Shape("poly", dict(origin=c.tolist(), e1=e1.tolist(), e2=e2.tolist(), e3=e3.tolist(), degree=deg, coef=coef.tolist(),
                               scale=sc, extent=ext), float(abs(T)))
    return shape, len(coef) + 1


def residuals(shape, P):
    """distance of each point from its nearest leaflet shell (A)"""
    d = shape.dist(P)
    T = shape.thickness_pp or 0.0
    return np.minimum(np.abs(d - T / 2), np.abs(d + T / 2))


def fit_all(P, models=("plane", "sphere", "cylinder", "poly2", "poly3", "poly4")):
    out = []
    n = len(P)
    for m in models:
        try:
            if m == "plane": sh, k = fit_plane(P)
            elif m == "sphere": sh, k = fit_sphere(P)
            elif m == "cylinder": sh, k = fit_cylinder(P)
            elif m.startswith("poly"): sh, k = fit_poly(P, int(m[4:]))
            else: continue
        except Exception as e:                                   # pragma: no cover
            print("  %s: fit failed (%s)" % (m, e)); continue
        r = residuals(sh, P)
        rss = float(np.sum(r ** 2))
        out.append(dict(model=m, shape=sh, rmse=math.sqrt(rss / n), bic=n * math.log(max(rss / n, 1e-9)) + k * math.log(n), k=k))
    return out


def choose(results, dbic=10.0):
    """simplest model whose BIC is within dbic of the best (order: plane < sphere ~ cylinder < poly2 < poly3 < poly4)"""
    order = {"plane": 0, "sphere": 1, "cylinder": 1, "poly2": 2, "poly3": 3, "poly4": 4}
    best = min(r["bic"] for r in results)
    ok = [r for r in results if r["bic"] <= best + dbic]
    return min(ok, key=lambda r: (order[r["model"]], r["bic"]))


# --------------------------------------------------------------------------------------------------------------- input
PROTEIN = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
           "HSD", "HSE", "HSP", "HID", "HIE", "HIP", "ASH", "GLH", "CYX", "CYM", "LYN", "NME", "ACE", "HOH", "TIP", "TIP3", "SOL", "WAT", "NA", "CL",
           "SOD", "CLA", "POT", "K", "MG", "CAL", "ZN", "DUM"}


def read_atoms(path, names, resnames=None):
    pts, res = [], set()
    nm = {x.strip() for x in names.split(",")}
    rn = {x.strip().upper() for x in resnames.split(",")} if resnames else None
    for l in open(path):
        if l.startswith(("ATOM", "HETATM")) and len(l) > 54:
            r = l[17:21].strip().upper(); a = l[12:16].strip()
            if rn is not None:
                if r not in rn: continue
            elif r in PROTEIN:
                continue
            if a in nm or "*" in nm:
                pts.append([float(l[30:38]), float(l[38:46]), float(l[46:54])]); res.add(r)
    return np.array(pts), res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--lipids", help="PDB with lipids (the protein may be in it; protein, water and ions are ignored)")
    src.add_argument("--points", help="text file, one 'x y z' per line (leaflet marker atoms of BOTH leaflets)")
    src.add_argument("--manual", choices=["plane", "sphere", "cylinder"], help="no fit: describe the surface (needs --thickness-pp)")
    ap.add_argument("--atom-names", default="P,P31,P8,P1", help="atoms that mark a leaflet (default: phosphorus P, P31, P8, P1)")
    ap.add_argument("--resnames", default=None, help="restrict to these lipid residue names (default: every non-protein, non-water, non-ion residue)")
    ap.add_argument("--within", type=float, nargs=4, metavar=("X", "Y", "Z", "R"), help="use only markers within R A of this point (e.g. the protein centre)")
    ap.add_argument("--model", default="auto", help="auto | plane | sphere | cylinder | poly2 | poly3 | poly4 (default auto: BIC with a preference for simple shapes)")
    ap.add_argument("--p-offset", type=float, default=4.5,
                    help="distance from the phosphate shell to the edge of the hydrocarbon core, A (default 4.5): core thickness = P-P thickness - 2*offset")
    ap.add_argument("--out", default="membrane_fit.json")
    ap.add_argument("--center", type=float, nargs=3, help="--manual sphere: centre; plane/cylinder: a point on the surface / axis")
    ap.add_argument("--normal", type=float, nargs=3, help="--manual plane: normal; cylinder: axis direction")
    ap.add_argument("--radius", type=float, help="--manual sphere / cylinder: radius of the MID-surface (A)")
    ap.add_argument("--thickness-pp", type=float, help="--manual: phosphate-phosphate thickness (A)")
    a = ap.parse_args()

    if a.manual:
        T = a.thickness_pp or sys.exit("--manual needs --thickness-pp")
        if a.manual == "plane":
            sh = Shape("plane", dict(point=list(a.center or [0, 0, 0]), normal=list(_unit(a.normal or [0, 0, 1]))), T)
        elif a.manual == "sphere":
            sh = Shape("sphere", dict(center=list(a.center or sys.exit("--center needed")), radius=a.radius or sys.exit("--radius needed")), T)
        else:
            sh = Shape("cylinder", dict(point=list(a.center or sys.exit("--center needed")), axis=list(_unit(a.normal or sys.exit("--normal (axis) needed"))),
                                        radius=a.radius or sys.exit("--radius needed")), T)
        info = dict(source="manual")
    else:
        if a.points:
            P = np.loadtxt(a.points).reshape(-1, 3); res = {"points file"}
        elif a.lipids:
            P, res = read_atoms(a.lipids, a.atom_names, a.resnames)
        else:
            sys.exit("give --lipids, --points or --manual")
        if a.within:
            P = P[np.linalg.norm(P - np.array(a.within[:3]), axis=1) <= a.within[3]]
        if len(P) < 30:
            sys.exit("only %d marker atoms found (names %s); check --atom-names / --resnames" % (len(P), a.atom_names))
        print("fitting %d marker atoms (%s)" % (len(P), ",".join(sorted(res))))
        results = fit_all(P)
        print("%-9s %8s %10s %4s   thickness" % ("model", "RMSE(A)", "BIC", "k"))
        for r in results:
            print("%-9s %8.2f %10.1f %4d   %.1f A" % (r["model"], r["rmse"], r["bic"], r["k"], r["shape"].thickness_pp or 0))
        pick = choose(results) if a.model == "auto" else next((r for r in results if r["model"] == a.model), None)
        if pick is None:
            sys.exit("model %s not available" % a.model)
        sh = pick["shape"]; info = dict(source=a.lipids or a.points, n_markers=len(P), model=pick["model"], rmse=pick["rmse"],
                                       table={r["model"]: dict(rmse=r["rmse"], bic=r["bic"]) for r in results})
        print("selected: %s  (RMSE %.2f A)" % (pick["model"], pick["rmse"]))
    sh.info = info
    core = max(sh.thickness_pp - 2 * a.p_offset, 10.0)
    sh.to_json(a.out, core_thickness=core, p_offset=a.p_offset)
    print(sh.describe())
    print("suggested hydrocarbon core thickness %.1f A (P-P %.1f minus 2 x %.1f)  ->  %s" % (core, sh.thickness_pp, a.p_offset, a.out))
    rad = sh.curvature_radii()
    if rad and min(rad) < 1e9:
        print("radius of curvature %.0f A: the layered Born correction treats the membrane as locally flat, error grows when R is below ~100 A" % min(rad))


if __name__ == "__main__":
    main()
