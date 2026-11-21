"""
minirefmac.maps -- sigmaA-weighted map coefficients and MTZ output.

Interface (declared at the bottom of core.py)::

    write_output_mtz(data, path, fc_total, scale, mlpar, m, phase,
                     xyz, b_iso, extra_columns=True,
                     output_scale="absolute", fc_atoms=None) -> None
    map_coefficients(data, fc_total, scale, mlpar, m, phase,
                     output_scale="absolute", fc_atoms=None) -> dict[str, array]

SCALE CONVENTIONS
-----------------
REFMAC5 writes its output MTZ on the ABSOLUTE (electron) scale: the observed
amplitudes and their sigmas are divided by the overall/anisotropic scale
    Fo_abs = fo / k_total,  SigFo_abs = sigfo / k_total
    k_total = k_overall * exp(-b_over*stol2) * exp(-0.25*aniso_quad(h))
and all map coefficients are expressed in the same units as Fc.  This is the
DEFAULT here (``output_scale="absolute"``).  On the case1/case2 reference data
REFMAC5 rescales F/SIGF by 0.3438 / 0.9735 respectively; using the same
convention additionally makes F, FWT, DELFWT, FC_ALL and FC mutually
comparable, which is what downstream map/phase programs expect.

``output_scale="observed"`` keeps the legacy minirefmac behaviour: F/SIGF are
left exactly as in the input MTZ and every map coefficient is multiplied by
k_total so that it is on the observed scale.  Both modes are internally
consistent; only the units differ.  A single global multiplier cannot convert
one into the other because k_total is resolution dependent.

Complex form (REFMAC / Servalcat sigmaA weights)::

    Fo_c   = (fo * f_fo) * exp(i*phase)    # phase = best/combined phase (rad)
    DFc_c  = D * k_fc * fc_total
    FWT_c  = 2*m*Fo_c - DFc_c             # 2mFo - DFc  (2Fo-Fc map)
    DEL_c  =   m*Fo_c - DFc_c             #  mFo - DFc  (difference map)

    absolute mode:  f_fo = 1/k_total,  k_fc = 1
    observed mode:  f_fo = 1,          k_fc = k_total

    FWT    = |FWT_c|          PHWT    = deg(arg FWT_c)
    DELFWT = |DEL_c|          PHDELWT = deg(arg DEL_c)

With no experimental phase prior phase == arg(fc_total), so this reduces to
the familiar REFMAC  2mF_o - D F_c, phi_c  coefficients.

FREE / MISSING RESTORE (REFMAC "MAPC FREE RESTORE")
    For  ~data.work  (the free/test set AND unobserved reflections)::
        FWT_c = DFc_c        ->  FWT   = D * k_fc * |fc_total|,
                                 PHWT  = deg(arg fc_total)
        DEL_c = 0            ->  DELFWT = 0, PHDELWT = 0
    FOM (m) is still reported on these reflections.

FC_ALL / PHIC_ALL : k_fc*|fc_total| and deg(arg fc_total) -- the FULL
                    calculated structure factor (bulk solvent included).
FC / PHIC         : k_fc*|fc_atoms| and deg(arg fc_atoms) -- the ATOM-ONLY
                    contribution, when ``fc_atoms`` is supplied (refine.py
                    always supplies it).  If it is omitted, fc_total is used
                    and FC/PHIC == FC_ALL/PHIC_ALL (documented, not faked).
FOM               : figure of merit m.

All phases are in DEGREES (MTZ column type 'P'), in [0, 360).  Non-finite
values are replaced by amplitude 0 / phase 0.  Column types: F (amplitude),
P (phase, degrees), W (weight/FOM).

The per-reflection k_total used for the conversion is always recorded in the
MTZ history together with k_overall/b_over, and in report.json (cycle scale).
"""

import numpy as np
import gemmi

# core columns, always written
_CORE_COLS = (
    ("FWT", "F"),
    ("PHWT", "P"),
    ("DELFWT", "F"),
    ("PHDELWT", "P"),
    ("FOM", "W"),
)
# columns written only when extra_columns=True
_EXTRA_COLS = (
    ("FC_ALL", "F"),
    ("PHIC_ALL", "P"),
    ("FC", "F"),
    ("PHIC", "P"),
)

_MODES = ("absolute", "observed")


def _check_mode(mode):
    mode = str(mode).lower()
    if mode not in _MODES:
        raise ValueError("output_scale must be one of %s, got %r"
                         % (_MODES, mode))
    return mode


def _finite(x, fill=0.0):
    """Replace non-finite entries of a float array with `fill`."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(x), x, fill)


def _phase_deg(z):
    """deg(arg(z)) in [0,360), with 0 for non-finite / zero complex values."""
    p = np.degrees(np.angle(np.asarray(z, dtype=np.complex128)))
    p = np.mod(p, 360.0)
    # np.mod of a tiny negative angle can round to 360.0; fold it to 0 so the
    # result is strictly in [0, 360).
    p = np.where(p >= 360.0 - 1e-6, 0.0, p)
    return np.where(np.isfinite(p), p, 0.0)


def all_rows_k_total(mtz, scale):
    """k_total for EVERY row of ``mtz`` (not just the retained reflections).

    Uses exactly the same reciprocal-space construction as dataio.load_data
    so that the rows present in ``data`` reproduce scale.k_total(data)
    bit-for-bit.  Needed to rescale inherited F/SIGF columns even when a
    resolution filter was applied to the working data.
    """
    arr = np.asarray(mtz.array, dtype=np.float64)
    hkl = np.rint(arr[:, :3]).astype(np.float64)
    frac_mat = np.asarray(mtz.cell.frac.mat.tolist(), dtype=np.float64)
    rv = hkl @ frac_mat
    s2 = np.einsum("ij,ij->i", rv, rv)
    holder = _ScaleRows(stol2=0.25 * s2, recip_vec=rv, n_ref=len(s2))
    return _finite(scale.k_total(holder)), holder


class _ScaleRows(object):
    """Duck-typed view consumed by ScaleParams.k_total / aniso_quad."""

    def __init__(self, stol2, recip_vec, n_ref):
        self.stol2 = stol2
        self.recip_vec = recip_vec
        self.n_ref = n_ref


def scale_factors(data, scale, output_scale):
    """Return (f_fo, k_fc): the observed-amplitude and Fc multipliers.

    f_fo multiplies the input fo/sigfo when they are written out;
    k_fc multiplies every calculated structure factor / map coefficient.
    """
    mode = _check_mode(output_scale)
    k = _finite(scale.k_total(data))
    k = np.where(k > 1e-12, k, 1.0)
    if mode == "absolute":
        return 1.0 / k, np.ones_like(k)
    return np.ones_like(k), k


def map_coefficients(data, fc_total, scale, mlpar, m, phase,
                     output_scale="absolute", fc_atoms=None):
    """Return sigmaA-weighted map columns as a dict of (N,) float64 arrays."""
    n = data.n_ref
    f_fo, k_fc = scale_factors(data, scale, output_scale)
    D = _finite(np.asarray(mlpar.D, dtype=np.float64))    # (N,)
    fc = np.asarray(fc_total, dtype=np.complex128)
    fc = np.where(np.isfinite(fc.real) & np.isfinite(fc.imag),
                  fc, np.complex128(0.0)).astype(np.complex128)
    if fc_atoms is None:
        fca = fc
    else:
        fca = np.asarray(fc_atoms, dtype=np.complex128)
        fca = np.where(np.isfinite(fca.real) & np.isfinite(fca.imag),
                       fca, np.complex128(0.0)).astype(np.complex128)
    fo = _finite(np.asarray(data.fo, dtype=np.float64))   # NaN missing -> 0
    m = _finite(np.asarray(m, dtype=np.float64))          # (N,)
    phase = _finite(np.asarray(phase, dtype=np.float64))  # radians

    # ---- complex coefficients -------------------------------------------
    Fo_c = (fo * f_fo) * np.exp(1j * phase)
    DFc_c = D * k_fc * fc
    FWT_c = 2.0 * m * Fo_c - DFc_c
    DEL_c = m * Fo_c - DFc_c

    # ---- FREE / missing RESTORE -----------------------------------------
    work = np.asarray(data.work, dtype=bool)
    restore = ~work
    FWT_c = np.array(FWT_c, dtype=np.complex128, copy=True)
    DEL_c = np.array(DEL_c, dtype=np.complex128, copy=True)
    FWT_c[restore] = DFc_c[restore]
    DEL_c[restore] = 0.0

    FWT = np.abs(FWT_c)
    DELFWT = np.abs(DEL_c)
    PHWT = _phase_deg(FWT_c)
    PHDELWT = _phase_deg(DEL_c)

    # ---- full calculated SF (bulk solvent included) ---------------------
    FC_ALL = k_fc * np.abs(fc)
    PHIC_ALL = _phase_deg(fc)

    # ---- atom-only FC/PHIC ----------------------------------------------
    FC = k_fc * np.abs(fca)
    PHIC = _phase_deg(fca)

    FOM = m.copy()

    # ---- final sanitise: amplitude 0 / phase 0 where non-finite ---------
    out = {
        "FWT": _finite(FWT),
        "PHWT": _finite(PHWT),
        "DELFWT": _finite(DELFWT),
        "PHDELWT": _finite(PHDELWT),
        "FOM": _finite(FOM),
        "FC_ALL": _finite(FC_ALL),
        "PHIC_ALL": _finite(PHIC_ALL),
        "FC": _finite(FC),
        "PHIC": _finite(PHIC),
    }
    for key, val in out.items():
        if val.shape != (n,):
            raise ValueError("map_coefficients: %s has wrong shape %s"
                             % (key, val.shape))
    return out


def _hkl_rows(mtz, data):
    """Map each data reflection to its row index in the (copied) MTZ.

    data.hkl may differ from mtz reflections when a resolution filter was
    applied in load_data; matching by (h,k,l) is therefore robust.  Returns
    (target, valid): target[i] is the MTZ row for data reflection i, or -1.
    """
    arr = np.asarray(mtz.array)
    mh = np.rint(arr[:, :3]).astype(np.int64)
    row_of = {}
    for r in range(mh.shape[0]):
        row_of[(int(mh[r, 0]), int(mh[r, 1]), int(mh[r, 2]))] = r
    dh = np.asarray(data.hkl, dtype=np.int64)
    target = np.fromiter(
        (row_of.get((int(dh[i, 0]), int(dh[i, 1]), int(dh[i, 2])), -1)
         for i in range(dh.shape[0])),
        dtype=np.int64, count=dh.shape[0])
    valid = target >= 0
    return target, valid


def write_output_mtz(data, path, fc_total, scale, mlpar, m, phase,
                     xyz, b_iso, extra_columns=True,
                     output_scale="absolute", fc_atoms=None, history_extra=None):
    """Copy data.mtz, add/overwrite the map columns and write ``path``.

    xyz / b_iso are kept in the signature to match the core.py contract; the
    map coefficients depend only on fc_total / scale / mlpar / m / phase (the
    refined coordinates themselves are written by dataio.write_refined_pdb).

    ``output_scale``:
      * "absolute" (default, REFMAC5 convention) -- inherited F/SIGF and all
        generated amplitudes are divided by k_total, i.e. everything is on
        the electron scale of Fc.
      * "observed" (legacy) -- F/SIGF untouched, amplitudes on the input scale.

    ``fc_atoms`` -- optional atom-only structure factor; when given, FC/PHIC
    is the atom-only contribution instead of a copy of FC_ALL/PHIC_ALL.

    extra_columns=True additionally writes FC_ALL, PHIC_ALL, FC, PHIC.
    """
    mode = _check_mode(output_scale)
    coeffs = map_coefficients(data, fc_total, scale, mlpar, m, phase,
                              output_scale=mode, fc_atoms=fc_atoms)

    # fresh copy so we never mutate data.mtz in place
    mtz = gemmi.read_mtz_file(data.mtz_path)
    target, valid = _hkl_rows(mtz, data)
    nrows = mtz.nreflections

    cols = list(_CORE_COLS)
    if extra_columns:
        cols += list(_EXTRA_COLS)

    for label, ctype in cols:
        vals = np.asarray(coeffs[label], dtype=np.float64)
        # existing column -> overwrite; otherwise add a new one
        idx = None
        for i, c in enumerate(mtz.columns):
            if c.label == label:
                idx = i
                break
        if idx is None:
            mtz.add_column(label, ctype)
            idx = len(mtz.columns) - 1
        colvals = np.zeros(nrows, dtype=np.float64)
        colvals[target[valid]] = vals[valid]
        # column.array is a writable view onto mtz.array (verified on 0.7.5)
        mtz.column_with_label(label).array[:] = colvals.astype(
            np.asarray(mtz.array).dtype)

    # ---- put the inherited observed amplitudes on the same scale --------
    if mode == "absolute":
        k_all, _ = all_rows_k_total(mtz, scale)
        f_fo_all = 1.0 / np.where(k_all > 1e-12, k_all, 1.0)
        for label in (data.fp_label, data.sigfp_label):
            if not label:
                continue
            col = mtz.column_with_label(label)
            if col is None:
                continue
            col.array[:] = (np.asarray(col.array, dtype=np.float64)
                            * f_fo_all).astype(np.asarray(mtz.array).dtype)

    # The copied input MTZ carries NaN in F/SIGF for unobserved reflections
    # (857 here).  Those are the input's own missing-value markers; make the
    # OUTPUT file free of NaN/Inf by folding them to 0 (amplitude 0 == missing,
    # which does not affect the observed(fo>0) test).  All columns we generated
    # are already finite; this only sanitises the inherited columns.
    arr = mtz.array
    bad = ~np.isfinite(arr)
    if bad.any():
        arr[bad] = 0.0

    # ---- history: make the scale convention self-describing -------------
    k_work, _ = all_rows_k_total(mtz, scale)
    history = list(mtz.history)
    history.append("MINIREFMAC output_scale=%s" % mode)
    history.append("MINIREFMAC k_overall=%.6g b_over=%.6g "
                   "k_sol=%.6g b_sol=%.6g solvent=%s"
                   % (scale.k_overall, scale.b_over, scale.k_sol,
                      scale.b_sol, scale.solvent))
    history.append("MINIREFMAC k_total min=%.5g mean=%.5g max=%.5g"
                   % (float(np.min(k_work)), float(np.mean(k_work)),
                      float(np.max(k_work))))
    history.append("MINIREFMAC F_abs = F_obs / k_total (absolute mode only)")
    history.append("MINIREFMAC FC/PHIC atom-only"
                   if fc_atoms is not None else
                   "MINIREFMAC FC/PHIC = FC_ALL (no fc_atoms)")
    if history_extra:
        history.extend(str(h) for h in history_extra)
    mtz.history = history

    mtz.write_to_file(path)
