"""
minirefmac.sfcalc -- structure factors, bulk-solvent mask and scale fitting.

Implements the interface contract declared at the bottom of core.py:
    calc_fc(data, xyz, b_iso, occ=None, chunk=64)
    calc_fmask(data, xyz=None)
    fit_scales(data, fc_atoms, fmask=None, solvent="mask",
               fit_aniso=True, params0=None)
    build_fc_total(data, fc_atoms, fmask, scale)

SYMMETRY CONVENTION (verified against gemmi 0.7.5, see selftest):
    gemmi.StructureFactorCalculatorX(cell).calculate_sf_from_model(...)
    returns the FULL-UNIT-CELL structure factor, i.e. it sums over ALL
    symmetry-equivalent atoms.  The P1 asymmetric-unit-only sum does NOT match
    it (off by >1000x on real data).  Therefore calc_fc performs an explicit
    symmetry expansion over ``sg.operations()`` (which already includes lattice
    centering translations):

        Fc(h) = sum_ops sum_j occ_j f_j(stol2) exp(-B_j stol2)
                        * exp(2*pi*i * h . (R_op x_j + t_op))

    For P1 this collapses to exactly core.py's FOURIER_NOTE.  Every consumer
    (fit_scales, build_fc_total, likelihood, r_factors) must use THIS Fc; the
    atomic-gradient module must likewise sum over symmetry ops when differentiating
    (its P1-only C_j(h) formula is the derivative of the ASU-only term).

SOLVENT MASK:
    calc_fmask uses gemmi.SolventMasker(Refmac) on a FloatGrid, an FFT
    (gemmi.transform_map_to_f_phi) and ReciprocalComplexGrid.get_value_by_hkl
    to obtain Fmask at every requested hkl.  If that FFT path is unavailable or
    raises for any reason, calc_fmask prints an explicit warning and returns an
    all-zero complex array; the caller then falls back to the Babinet (or none)
    bulk-solvent model, which does not need a mask.
"""
import warnings
import numpy as np
import gemmi
from scipy.optimize import least_squares

try:
    from .core import ScaleParams
except ImportError:  # pragma: no cover - allow running as a flat module
    from core import ScaleParams


# ---- symmetry operations ------------------------------------------------
def _sym_ops(data):
    """List of (R, t) for all symmetry ops incl. centering, cached on data."""
    ops = getattr(data, "_fc_ops", None)
    if ops is not None:
        return ops
    out = []
    for op in data.sg.operations():
        den = float(op.DEN)
        R = np.array([[op.rot[i][j] / den for j in range(3)] for i in range(3)],
                     dtype=np.float64)
        t = np.array([op.tran[i] / den for i in range(3)], dtype=np.float64)
        out.append((R, t))
    data._fc_ops = out
    return out


# ---- atomic structure factors -------------------------------------------
def calc_fc(data, xyz, b_iso, occ=None, chunk=64):
    """Full-unit-cell Fc(h), shape (N,) complex128, symmetry-expanded.

    Atoms are processed in column chunks so no (N, M) temporary is kept.
    """
    n, m = data.n_ref, data.n_atoms
    xyz = np.asarray(xyz, dtype=np.float64)
    b_iso = np.asarray(b_iso, dtype=np.float64)
    if occ is None:
        occ = data.occ
    occ = np.asarray(occ, dtype=np.float64)
    hkl = np.asarray(data.hkl, dtype=np.float64)       # (N,3)
    stol2 = np.asarray(data.stol2, dtype=np.float64)   # (N,)
    ops = _sym_ops(data)

    fc = np.zeros(n, dtype=np.complex128)
    chunk = max(1, int(chunk))
    for c0 in range(0, m, chunk):
        c1 = min(c0 + chunk, m)
        xc = xyz[c0:c1]                                  # (nc,3)
        bc = b_iso[c0:c1]
        oc = occ[c0:c1]
        sfc = data.sf_block(c0, c1)                       # (N,nc)
        # real weight per reflection/atom: occ * f_j(h) * exp(-B stol2)
        W = (oc[None, :] * sfc) * np.exp(-bc[None, :] * stol2[:, None])
        for (R, t) in ops:
            xt = xc @ R.T + t                            # (nc,3) fractional
            ang = 2.0 * np.pi * (hkl @ xt.T)             # (N,nc)
            fc += np.sum(W * np.exp(1j * ang), axis=1)
        del W
    return fc


# ---- bulk-solvent mask --------------------------------------------------
def _write_mask_coords(data, st, xyz):
    """Put fractional xyz into a gemmi Structure copy (for the masker)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    sel = getattr(data, "_atom_sel", None)
    if sel is None:
        cell = data.cell
        g = 0
        j = 0
        for model in st:
            for chain in model:
                for res in chain:
                    for atom in res:
                        if (atom.altloc in ("", " ", "A", "1", "\x00", "\0")
                                and atom.element.name not in ("H", "D")):
                            if j < len(xyz):
                                atom.pos = cell.orthogonalize(
                                    gemmi.Fractional(float(xyz[j, 0]),
                                                     float(xyz[j, 1]),
                                                     float(xyz[j, 2])))
                            j += 1
                        g += 1
        return
    sel_map = {int(g): j for j, g in enumerate(sel)}
    cell = data.cell
    g = 0
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    jj = sel_map.get(g)
                    if jj is not None and jj < len(xyz):
                        atom.pos = cell.orthogonalize(
                            gemmi.Fractional(float(xyz[jj, 0]),
                                             float(xyz[jj, 1]),
                                             float(xyz[jj, 2])))
                    g += 1


def calc_fmask(data, xyz=None):
    """Complex bulk-solvent mask structure factor, shape (N,) complex128.

    Returns an all-zero array (with a warning) if the FFT path is unavailable.
    """
    n = data.n_ref
    try:
        try:
            st = data.st.clone()
        except Exception:
            st = gemmi.read_structure(data.pdb_path)
        eff_xyz = data.xyz if xyz is None else xyz
        _write_mask_coords(data, st, eff_xyz)

        dpos = data.d[np.isfinite(data.d) & (data.d > 0.0)]
        dmin = float(np.min(dpos)) if dpos.size else 2.0
        spacing = float(np.clip(dmin / 2.0, 0.6, 2.5))
        grid = gemmi.FloatGrid()
        grid.setup_from(st, spacing)
        masker = gemmi.SolventMasker(gemmi.AtomicRadiiSet.Refmac)
        masker.put_mask_on_float_grid(grid, st[0])
        sf_grid = gemmi.transform_map_to_f_phi(grid)
        hkli = np.ascontiguousarray(data.hkl, dtype=np.int32)
        vals = sf_grid.get_value_by_hkl(hkli)
        return np.asarray(vals).astype(np.complex128)
    except Exception as exc:
        warnings.warn(
            "calc_fmask: FFT bulk-solvent mask unavailable (%r); "
            "returning zeros -- caller should fall back to Babinet/none."
            % (exc,))
        print("WARNING: calc_fmask FFT path failed (%s); returning zeros; "
              "use solvent='babinet'." % exc)
        return np.zeros(n, dtype=np.complex128)


# ---- scale / bulk-solvent fitting ---------------------------------------
def fit_scales(data, fc_atoms, fmask=None, solvent="mask",
               fit_aniso=True, params0=None):
    """Fit k_overall, b_over, k_sol, b_sol and anisotropic scaling on work set.

    Minimises a robust (clipped) weighted residual
        (fo - k_total(h) * |fc_total(h)|) / sigfo
    with scipy.optimize.least_squares.  Returns a ScaleParams with a traceless
    6-component b_aniso (b33 = -(b11+b22)).
    """
    solvent = str(solvent).lower()
    if solvent not in ("mask", "babinet", "none"):
        raise ValueError("solvent must be 'mask', 'babinet' or 'none'")
    if solvent == "mask" and fmask is None:
        warnings.warn("fit_scales: solvent='mask' but fmask is None; "
                      "using solvent='none'")
        solvent = "none"

    w = data.work
    fo = np.asarray(data.fo, dtype=np.float64)[w]
    sig = np.maximum(np.asarray(data.sigfo, dtype=np.float64)[w], 1e-3)
    st2 = np.asarray(data.stol2, dtype=np.float64)[w]
    s = np.asarray(data.recip_vec, dtype=np.float64)[w]       # (Nw,3)
    fca = np.asarray(fc_atoms, dtype=np.complex128)[w]
    fm = (np.asarray(fmask, dtype=np.complex128)[w]
          if (solvent == "mask" and fmask is not None) else None)

    has_sol = solvent != "none"
    i_sol = 2
    i_aniso = i_sol + (2 if has_sol else 0)
    npar = i_aniso + (5 if fit_aniso else 0)

    # ---- initial guess: robust amplitude ratio -------------------------
    fo_med = float(np.median(fo)) if fo.size else 1.0
    fc_med = float(np.median(np.abs(fca))) if fca.size else 1.0
    if not np.isfinite(fc_med) or fc_med <= 0:
        fc_med = 1.0
    k0 = max(fo_med / max(fc_med, 1e-6), 1e-6)

    p0 = np.array([np.log(k0), 0.0], dtype=np.float64)
    if has_sol:
        p0 = np.concatenate([p0, [0.35, np.log(100.0)]])
    if fit_aniso:
        p0 = np.concatenate([p0, np.zeros(5, dtype=np.float64)])

    # optional user-supplied start
    if params0 is not None:
        if isinstance(params0, ScaleParams):
            q = [np.log(max(params0.k_overall, 1e-6)), params0.b_over]
            if has_sol:
                q += [params0.k_sol, np.log(max(params0.b_sol, 10.0))]
            if fit_aniso:
                ba = np.asarray(params0.b_aniso, dtype=np.float64)
                q += [ba[0], ba[1], ba[3], ba[4], ba[5]]  # b11,b22,b12,b13,b23
            q = np.asarray(q, dtype=np.float64)
            if q.size == npar:
                p0 = q
        else:
            q = np.asarray(params0, dtype=np.float64).ravel()
            if q.size == npar:
                p0 = q.copy()

    # ---- bounds --------------------------------------------------------
    lb = np.array([p0[0] - 6.0, -20.0], dtype=np.float64)
    ub = np.array([p0[0] + 6.0, 50.0], dtype=np.float64)
    if has_sol:
        lb = np.concatenate([lb, [0.0, np.log(10.0)]])
        ub = np.concatenate([ub, [0.95, np.log(400.0)]])
    if fit_aniso:
        lb = np.concatenate([lb, np.full(5, -30.0)])
        ub = np.concatenate([ub, np.full(5, 30.0)])
    p0 = np.clip(p0, lb + 1e-9, ub - 1e-9)

    sxsy = s[:, 0] * s[:, 1]
    sxsz = s[:, 0] * s[:, 2]
    sysz = s[:, 1] * s[:, 2]
    sx2 = s[:, 0] ** 2
    sy2 = s[:, 1] ** 2
    sz2 = s[:, 2] ** 2

    def residual(p):
        k = np.exp(p[0])
        bo = p[1]
        if has_sol:
            ks = p[2]
            bs = np.exp(p[3])
            if solvent == "mask":
                fct = fca + ks * np.exp(-bs * st2) * fm
            else:  # babinet
                fct = fca * (1.0 - ks * np.exp(-bs * st2))
        else:
            fct = fca
        if fit_aniso:
            b11, b22, b12, b13, b23 = p[i_aniso:i_aniso + 5]
            b33 = -(b11 + b22)
            quad = (b11 * sx2 + b22 * sy2 + b33 * sz2
                    + 2.0 * b12 * sxsy + 2.0 * b13 * sxsz
                    + 2.0 * b23 * sysz)
        else:
            quad = 0.0
        ktot = k * np.exp(-bo * st2) * np.exp(-0.25 * quad)
        r = (fo - ktot * np.abs(fct)) / sig
        # robust compression of outliers beyond |r| = 10
        a = np.abs(r)
        # np.maximum keeps log1p's argument >= 0 for ALL entries, so the
        # (discarded) a<=10 branch cannot emit an invalid-value warning.
        compressed = 10.0 + np.log1p(np.maximum(a - 10.0, 0.0))
        return np.sign(r) * np.where(a > 10.0, compressed, a)

    res = least_squares(residual, p0, bounds=(lb, ub), max_nfev=300)
    p = res.x

    k_overall = float(np.exp(p[0]))
    b_over = float(p[1])
    k_sol = float(p[2]) if has_sol else 0.0
    b_sol = float(np.exp(p[3])) if has_sol else 100.0
    if fit_aniso:
        b11, b22, b12, b13, b23 = p[i_aniso:i_aniso + 5]
        b33 = -(b11 + b22)
        b_aniso = np.array([b11, b22, b33, b12, b13, b23], dtype=np.float64)
    else:
        b_aniso = np.zeros(6, dtype=np.float64)

    return ScaleParams(k_overall=k_overall, b_over=b_over, b_aniso=b_aniso,
                       k_sol=k_sol, b_sol=b_sol, solvent=solvent)


# ---- total (solvent-corrected) model structure factor -------------------
def build_fc_total(data, fc_atoms, fmask, scale):
    """fc_total = fc_atoms + k_sol*exp(-b_sol*stol2)*fmask (mask),
    or fc_atoms*(1 - k_sol*exp(-b_sol*stol2)) (babinet), or fc_atoms (none).
    Returns (N,) complex128.
    """
    fca = np.asarray(fc_atoms, dtype=np.complex128)
    if scale.solvent == "mask":
        fm = np.asarray(fmask, dtype=np.complex128)
        return fca + scale.k_sol * np.exp(-scale.b_sol * data.stol2) * fm
    if scale.solvent == "babinet":
        return fca * (1.0 - scale.k_sol * np.exp(-scale.b_sol * data.stol2))
    return fca
