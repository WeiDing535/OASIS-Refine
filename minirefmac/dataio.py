"""
minirefmac.dataio -- MTZ / PDB input, reflection bookkeeping, IT92 scattering
factors and refined-coordinate output.

Implements the interface contract declared at the bottom of core.py:
    load_data(...), make_sf_table(data), write_refined_pdb(data, xyz, b_iso, path)

Conventions / measured facts (gemmi 0.7.5; verified on Python 3.9 and
Python 3.12 installations):
  * data.xyz holds FRACTIONAL coordinates.  gemmi's ``atom.pos`` is a
    gemmi.Position (Cartesian, Angstrom), so it MUST be converted with
    ``cell.fractionalize(atom.pos)`` -- feeding atom.pos.tolist() directly as
    fractional coordinates gives completely wrong structure factors.
  * An empty altloc is the NUL character '\\x00' in gemmi 0.7.5 (not '').
  * recip_vec is the Cartesian reciprocal vector s = hkl @ cell.frac.mat,
    which satisfies |s|**2 == cell.calculate_1_d2(hkl) == 1/d**2.
  * Unknown/pseudo elements (atomic_number == 0, e.g. 'X', 'MSE') fall back to
    carbon for the scattering factor table.
"""
import os
import warnings
import numpy as np
import gemmi

try:
    from .core import RefineData
except ImportError:  # pragma: no cover - allow running as a flat module
    from core import RefineData


# ---- column label candidates, in priority order -------------------------
_FP_LABELS = ("FP", "F", "F_native", "F-obs", "FOBS", "Fobs")
_SIG_LABELS = ("SIGFP", "SIGF", "SIGF_native")
_FREE_LABELS = ("FreeR_flag", "FREE", "RFREE", "FREER", "R-free-flags",
                "FreeR", "R_FREE")
_PHIB_LABELS = ("PHIB", "DM_PHIB", "PHI")
_FOM_LABELS = ("FOM", "DM_FOM")
_HL_LABELS = ("HLA", "HLB", "HLC", "HLD")
_ALTLOC_OK = ("", " ", "A", "1", "\x00", "\0")
_HYDROGEN = ("H", "D")


# ---- small helpers ------------------------------------------------------
def _col(mtz, label):
    """Return a column by exact label, or None."""
    if label is None:
        return None
    for c in mtz.columns:
        if c.label == label:
            return c
    return None


def _pick_optional(mtz, candidates, ctype, explicit):
    """Pick a column: explicit label, else candidate label with matching type,
    else first column with matching type; return None if nothing found."""
    if explicit is not None:
        c = _col(mtz, explicit)
        if c is None:
            raise KeyError("MTZ column %r not found" % explicit)
        return c
    for lab in candidates:
        c = _col(mtz, lab)
        if c is not None and c.type == ctype:
            return c
    for c in mtz.columns:
        if c.type == ctype:
            return c
    return None


def _pick(mtz, candidates, ctype, explicit):
    c = _pick_optional(mtz, candidates, ctype, explicit)
    if c is None:
        raise KeyError("cannot find an MTZ column of type %r (tried %r)"
                       % (ctype, candidates))
    return c


def _values(col):
    return np.asarray(col.array, dtype=np.float64)


def resolve_free_value(free_flags, observed, free_value="auto"):
    """Decide which ``FreeR_flag`` value marks the EXCLUDED (test) set.

    ``free_value`` may be

    * an ``int``  -- used verbatim (the old behaviour), or
    * ``"auto"`` / ``None`` -- decided from the flag histogram.

    Auto rule (deliberately conservative):

    1. the CCP4 convention wins when it is self-consistent, i.e. flag 0 makes
       up 1-25 % of the observed reflections -> ``0``;
    2. otherwise the minority flag whose fraction is 1-25 % and closest to
       5 % is used (this is the ``0 = work, 1 = free`` files produced by some
       in-house pipelines);
    3. otherwise fall back to ``0`` and let the caller warn.

    Returns ``(value, info)``; ``info`` carries the histogram and the mode so
    the decision can be logged and stored in report.json.
    """
    flags = np.rint(np.asarray(free_flags, dtype=np.float64)).astype(np.int64)
    obs = np.asarray(observed, dtype=bool)
    sel = flags[obs] if obs.any() else flags
    n = max(int(sel.size), 1)
    vals, counts = np.unique(sel, return_counts=True)
    frac = {int(v): float(c) / n for v, c in zip(vals, counts)}
    info = {
        "requested": str(free_value),
        "mode": "explicit",
        "flag_values": [int(v) for v in vals],
        "flag_counts": [int(c) for c in counts],
        "flag_fractions": {int(v): round(f, 6) for v, f in frac.items()},
        "chosen": None,
        "n_observed": n,
    }

    if isinstance(free_value, str) and free_value.strip().lower() not in ("", "auto"):
        free_value = int(free_value)
    if isinstance(free_value, (int, np.integer)):
        info["chosen"] = int(free_value)
        info["free_fraction"] = round(frac.get(int(free_value), 0.0), 6)
        if info["free_fraction"] > 0.30:
            info["suspicious"] = True
        return int(free_value), info

    def usable(f):
        return 0.01 <= f <= 0.25

    chosen = None
    if 0 in frac and usable(frac[0]):
        chosen = 0
    else:
        cand = [(v, f) for v, f in frac.items() if v != 0 and usable(f)]
        if cand:
            chosen = min(cand, key=lambda vf: abs(vf[1] - 0.05))[0]
    if chosen is None:
        chosen = 0
        info["fallback"] = True
    info["mode"] = "auto"
    info["chosen"] = int(chosen)
    info["free_fraction"] = round(frac.get(int(chosen), 0.0), 6)
    # A "free" set larger than 30 % is almost always a mis-read convention:
    # Rfree then measures the work set and the number is meaningless.  The
    # caller (load_data) warns once with the column name attached.
    if info["free_fraction"] > 0.30:
        info["suspicious"] = True
    return int(chosen), info


def _resolve_element(symbol):
    """gemmi Element, with unknown/pseudo atoms ('X','MSE',...) -> carbon."""
    try:
        el = gemmi.Element(symbol)
    except Exception:
        return gemmi.Element("C")
    if el.atomic_number == 0:
        return gemmi.Element("C")
    return el


def _iter_atoms(st, keep_hydrogens):
    """Yield (global_index, atom, residue, chain, keep) in a fixed order.

    The SAME order/filter is used by load_data and by coordinate write-back so
    that data.xyz lines up with the gemmi Structure atoms exactly.
    """
    g = 0
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    keep = atom.altloc in _ALTLOC_OK
                    if atom.element.name in _HYDROGEN and not keep_hydrogens:
                        keep = False
                    yield g, atom, res, chain, keep
                    g += 1


# ---- scattering-factor table -------------------------------------------
def make_sf_table(data):
    """Precompute IT92 Cromer-Mann scattering factors for every reflection.

    Two storage layouts, chosen by the (N, M) full-table byte size:

      * small/medium (<= MINIREFMAC_SFTABLE_MAX_MB MB, default 200) or
        MINIREFMAC_SFTABLE_FORCE_FULL=1: keep the classic (N, M) float64
        ``data.sf_table`` -- numerically bit-for-bit identical to the original
        implementation, so the small-system path is unchanged;
      * large: set ``data.sf_table = None`` and keep only the de-duplicated
        ``data.sf_unique`` (N, n_elem) curves plus ``data.elem_index`` (M,)
        column map.  ``RefineData.sf_block(c0, c1)`` then gathers a block on
        demand with numpy advanced indexing.

    Both layouts populate sf_unique, elem_index and sf_table_nbytes, and both
    reproduce the ideal (N, M) table exactly via ``data.sf_block(0, M)``.
    """
    n, m = data.n_ref, data.n_atoms
    stol2 = np.asarray(data.stol2, dtype=np.float64)

    # ---- distinct element symbols + per-atom column map ----------------
    # De-duplication key is the raw symbol, exactly like the original
    # per-symbol cache; _resolve_element keeps the 'X'/pseudo -> carbon
    # fallback so the resulting numbers stay unchanged.
    index_of = {}
    uniq = []
    elem_index = np.empty(m, dtype=np.int64)
    for j, sym in enumerate(data.elements):
        c = index_of.get(sym)
        if c is None:
            c = len(uniq)
            index_of[sym] = c
            uniq.append(sym)
        elem_index[j] = c

    # ---- one (N,) IT92 curve per distinct element ----------------------
    # gemmi 0.7.5 calculate_sf takes a scalar stol2 only -> loop over hkl.
    sf_unique = np.empty((n, len(uniq)), dtype=np.float64)
    for c, sym in enumerate(uniq):
        el = _resolve_element(sym)
        sf_unique[:, c] = np.array(
            [el.it92.calculate_sf(float(t)) for t in stol2],
            dtype=np.float64)

    # ---- choose layout -------------------------------------------------
    try:
        max_mb = float(os.environ.get("MINIREFMAC_SFTABLE_MAX_MB", "200"))
    except ValueError:
        max_mb = 200.0
    force_full = (os.environ.get("MINIREFMAC_SFTABLE_FORCE_FULL", "") == "1")
    full_bytes = n * m * 8                       # (N, M) float64
    use_full = force_full or (full_bytes <= max_mb * 1e6)

    if use_full:
        table = np.empty((n, m), dtype=np.float64)
        for j in range(m):
            table[:, j] = sf_unique[:, elem_index[j]]
        data.sf_table = table
    else:
        data.sf_table = None

    data.sf_unique = sf_unique
    data.elem_index = elem_index
    if data.sf_table is not None:
        data.sf_table_nbytes = int(data.sf_table.nbytes)
    else:
        data.sf_table_nbytes = int(sf_unique.nbytes + elem_index.nbytes)
    return data.sf_table


# ---- coordinate write-back ---------------------------------------------
def _apply_xyz_to_structure(data, st, xyz, b_iso=None):
    """Write fractional xyz (and optional B) into a gemmi Structure copy."""
    xyz = np.asarray(xyz, dtype=np.float64)
    sel = getattr(data, "_atom_sel", None)
    if sel is None:                       # rebuild selector if unavailable
        keep_h = bool(getattr(data, "_keep_hydrogens", False))
        sel = [g for (g, _a, _r, _c, keep) in _iter_atoms(st, keep_h) if keep]
    sel_map = {int(g): j for j, g in enumerate(sel)}
    cell = data.cell
    b_iso = None if b_iso is None else np.asarray(b_iso, dtype=np.float64)
    for g, atom, _r, _c, _keep in _iter_atoms(st, True):
        j = sel_map.get(g)
        if j is None or j >= len(xyz):
            continue
        atom.pos = cell.orthogonalize(
            gemmi.Fractional(float(xyz[j, 0]), float(xyz[j, 1]),
                             float(xyz[j, 2])))
        if b_iso is not None and j < len(b_iso):
            atom.b_iso = float(b_iso[j])


def write_refined_pdb(data, xyz, b_iso, path):
    """Write a refined model to a PDB file, leaving data.st untouched."""
    try:
        st_out = data.st.clone()
    except Exception:
        st_out = gemmi.read_structure(data.pdb_path)
    _apply_xyz_to_structure(data, st_out, xyz, b_iso)
    # clone()/fresh read keeps CRYST1 and the space group
    st_out.write_pdb(path)


# ---- main loader --------------------------------------------------------
def load_data(mtz_path, pdb_path, fp_label=None, sigfp_label=None,
              free_label=None, free_value="auto", phib_label=None, fom_label=None,
              hla_label=None, hlb_label=None, hlc_label=None, hld_label=None,
              d_min=0.0, d_max=0.0, keep_hydrogens=False):
    """Read an MTZ + PDB/mmCIF and assemble a RefineData instance.

    ``free_value`` is the FreeR_flag value that marks the EXCLUDED (test) set.
    ``"auto"`` (default) honours the CCP4 convention (0 = free) when flag 0 is
    a 1-25 % minority, otherwise picks the minority flag closest to 5 %.
    """
    mtz = gemmi.read_mtz_file(mtz_path)
    st = gemmi.read_structure(pdb_path)
    cell = mtz.cell
    sg = mtz.spacegroup

    # ---- crystal consistency (warn only) -------------------------------
    try:
        pc = st.cell.parameters
        mc = cell.parameters
        if max(abs(float(a) - float(b)) for a, b in zip(pc, mc)) > 1e-2:
            warnings.warn("PDB cell %s differs from MTZ cell %s"
                          % (pc, mc))
    except Exception:
        pass
    try:
        p_hm = (st.spacegroup_hm or "").replace(" ", "").upper()
        m_hm = (sg.hm or "").replace(" ", "").upper()
        if p_hm and m_hm and p_hm != m_hm:
            warnings.warn("PDB space group %r differs from MTZ %r"
                          % (st.spacegroup_hm, sg.hm))
    except Exception:
        pass

    # ---- Miller indices -------------------------------------------------
    hcols = [_col(mtz, x) for x in ("H", "K", "L")]
    if all(c is not None for c in hcols):
        hkl_all = np.column_stack([np.rint(_values(c)) for c in hcols])
    else:
        hkl_all = np.asarray(mtz.array)[:, :3]
    hkl_all = np.ascontiguousarray(np.rint(hkl_all).astype(np.int32))
    n0 = len(hkl_all)

    # ---- observed amplitudes / sigma / free ----------------------------
    fp_col = _pick(mtz, _FP_LABELS, "F", fp_label)
    sig_col = _pick(mtz, _SIG_LABELS, "Q", sigfp_label)
    fo_all = _values(fp_col)
    sig_all = _values(sig_col)

    free_col = _pick_optional(mtz, _FREE_LABELS, "I", free_label)
    if free_col is None:
        warnings.warn("no FreeR_flag column found; all reflections -> work set")
        free_all = np.zeros(n0, dtype=np.float64)
    else:
        free_all = _values(free_col)

    # ---- optional phase prior ------------------------------------------
    phib_col = (_col(mtz, phib_label) if phib_label else
                _pick_optional(mtz, _PHIB_LABELS, "P", None))
    fom_col = (_col(mtz, fom_label) if fom_label else
               _pick_optional(mtz, _FOM_LABELS, "W", None))
    phib_all = np.deg2rad(_values(phib_col)) if phib_col is not None else None
    fom_all = _values(fom_col) if fom_col is not None else None

    explicit_hl = [hla_label, hlb_label, hlc_label, hld_label]
    if any(x is not None for x in explicit_hl):
        if not all(x is not None for x in explicit_hl):
            raise KeyError("HLA/HLB/HLC/HLD labels must all be given together")
        hl_cols = [_col(mtz, x) for x in explicit_hl]
        if any(c is None for c in hl_cols):
            raise KeyError("one of the requested HL columns is missing")
    else:
        hl_cols = [_col(mtz, x) for x in _HL_LABELS]
        if any(c is None for c in hl_cols):
            hl_cols = None
    hl_all = (np.column_stack([_values(c) for c in hl_cols]).astype(np.float64)
              if hl_cols is not None else None)

    # ---- d / s2 / stol2 / recip_vec (full, before filtering) -----------
    frac_mat = np.asarray(cell.frac.mat.tolist(), dtype=np.float64)  # (3,3)
    rv_all = hkl_all.astype(np.float64) @ frac_mat                    # (N0,3)
    s2_all = np.einsum("ij,ij->i", rv_all, rv_all)                    # 1/d^2
    with np.errstate(divide="ignore", invalid="ignore"):
        d_all = 1.0 / np.sqrt(s2_all)
    stol2_all = 0.25 * s2_all

    # ---- resolution filter ---------------------------------------------
    mask = np.isfinite(d_all) & (d_all > 0.0) & (s2_all > 0.0)
    if d_min and d_min > 0:
        mask &= (d_all >= d_min)
    if d_max and d_max > 0:
        mask &= (d_all <= d_max)

    def cut(a, axis0=True):
        if a is None:
            return None
        return a[mask] if axis0 else a

    hkl = np.ascontiguousarray(hkl_all[mask])
    d = np.ascontiguousarray(d_all[mask])
    s2 = np.ascontiguousarray(s2_all[mask])
    stol2 = np.ascontiguousarray(stol2_all[mask])
    recip_vec = np.ascontiguousarray(rv_all[mask])
    fo = np.ascontiguousarray(cut(fo_all))
    sigfo_raw = np.ascontiguousarray(cut(sig_all))
    free = np.rint(cut(free_all)).astype(np.int64)
    phib = np.ascontiguousarray(cut(phib_all)) if phib_all is not None else None
    fom = np.ascontiguousarray(cut(fom_all)) if fom_all is not None else None
    hl = np.ascontiguousarray(cut(hl_all)) if hl_all is not None else None

    # ---- epsilon / centric ---------------------------------------------
    ops = sg.operations()
    hkli = np.ascontiguousarray(hkl, dtype=np.int32)
    try:                       # fast array API (gemmi >= 0.6)
        epsilon = np.asarray(ops.epsilon_factor_array(hkli),
                             dtype=np.int32).astype(np.float64)
        centric = np.asarray(ops.centric_flag_array(hkli)).astype(np.int64)
    except Exception:          # scalar fallback, reflection by reflection
        epsilon = np.array([float(ops.epsilon_factor([int(x) for x in h]))
                            for h in hkl], dtype=np.float64)
        centric = np.array([int(ops.is_reflection_centric([int(x) for x in h]))
                            for h in hkl], dtype=np.int64)
    epsilon = np.maximum(epsilon, 1.0)

    # ---- observed / free / sigma cleanup -------------------------------
    observed = np.isfinite(fo) & (fo > 0.0) & np.isfinite(sigfo_raw)
    sigfo = sigfo_raw.copy()
    bad = ~(np.isfinite(sigfo) & (sigfo > 0.0))
    if bad.any():
        good = np.isfinite(sigfo_raw) & (sigfo_raw > 0.0)
        med = float(np.median(sigfo_raw[good])) if good.any() else 1.0
        if not np.isfinite(med) or med <= 0:
            med = 1.0
        sigfo[bad] = med

    # Which FreeR_flag value is the EXCLUDED (test) set?  Default is "auto":
    # the CCP4 convention (0 = free) when flag 0 is a plausible minority,
    # otherwise the minority flag closest to 5 %.  This prevents the classic
    # silent failure of a 0=work/1=free file being refined with 95 % of the
    # data in the "free" set (Rfree then means nothing).
    if free_col is None:
        # No flag column at all: everything is work.  (Previously the zero
        # placeholder made every reflection "free" and the refinement a no-op.)
        free_value_used = -1
        is_free = np.zeros(len(free), dtype=bool)
        free_info = dict(mode="absent", chosen=-1, free_fraction=0.0,
                         requested=str(free_value), flag_values=[],
                         flag_counts=[], flag_fractions={},
                         n_observed=int(np.sum(observed)))
    else:
        free_value_used, free_info = resolve_free_value(free, observed, free_value)
        is_free = (free == int(free_value_used))

    frac_free = (float(np.sum(is_free & observed)) / float(np.sum(observed))
                 if np.any(observed) else 0.0)
    free_info["free_fraction"] = round(frac_free, 6)
    if free_col is not None and not (0.01 <= frac_free <= 0.25):
        warnings.warn(
            "free set is %.1f %% of the observed reflections (flag %d, "
            "column %r).  Expected 2-10 %%: check --free-value / the flag "
            "convention, otherwise Rfree is meaningless."
            % (100.0 * frac_free, int(free_value_used), free_col.label))

    n_dup = int(len(hkl) - len(np.unique(hkl, axis=0)))
    if n_dup:
        warnings.warn("%d duplicate HKL rows in the input MTZ" % n_dup)

    # ---- model atoms ----------------------------------------------------
    xyz_l, b_l, occ_l = [], [], []
    elements, atom_names, altlocs = [], [], []
    res_names, res_seq, chain_ids, is_h = [], [], [], []
    sel = []
    for g, atom, res, chain, keep in _iter_atoms(st, keep_hydrogens):
        if not keep:
            continue
        frac = cell.fractionalize(atom.pos)
        xyz_l.append([frac.x, frac.y, frac.z])
        b_l.append(float(atom.b_iso))
        occ_l.append(float(atom.occ))
        elements.append(atom.element.name)
        atom_names.append(atom.name)
        altlocs.append(atom.altloc)
        res_names.append(res.name)
        res_seq.append(res.seqid.num)
        chain_ids.append(chain.name)
        is_h.append(atom.element.name in _HYDROGEN)
        sel.append(g)

    data = RefineData(
        cell=cell, sg=sg,
        spacegroup_name=sg.hm,
        cell_parameters=tuple(float(x) for x in cell.parameters),
        hkl=hkl, d=d, stol2=stol2, s2=s2,
        epsilon=epsilon, centric=centric, recip_vec=recip_vec,
        fo=fo, sigfo=sigfo, free=free, is_free=is_free, observed=observed,
        phib=phib, fom=fom, hl=hl,
        xyz=np.asarray(xyz_l, dtype=np.float64),
        b_iso=np.asarray(b_l, dtype=np.float64),
        occ=np.asarray(occ_l, dtype=np.float64),
        elements=elements, atom_names=atom_names, altloc=altlocs,
        res_names=res_names, res_seq=np.asarray(res_seq),
        chain_ids=chain_ids,
        is_hydrogen=np.asarray(is_h, dtype=bool),
        st=st, mtz=mtz,
        mtz_path=mtz_path, pdb_path=pdb_path,
        fp_label=fp_col.label, sigfp_label=sig_col.label,
        free_label=(free_col.label if free_col is not None else ""),
        free_value=int(free_value_used),
        free_info=free_info,
        data_stats=dict(
            n_ref=int(len(hkl)),
            n_observed=int(np.sum(observed)),
            n_missing_f=int(np.sum(~np.isfinite(fo) | (fo <= 0.0))),
            n_bad_sigma=int(np.sum(bad)),
            n_duplicate_hkl=n_dup,
            n_free=int(np.sum(is_free & observed)),
            n_work=int(np.sum(observed & ~is_free)),
            free_fraction=round(frac_free, 6),
            n_altloc_atoms=int(sum(1 for a in altlocs if a not in _ALTLOC_OK)),
            n_hydrogen_atoms=int(np.sum(is_h)) if len(is_h) else 0,
            n_atoms=len(xyz_l),
        ),
        # report the ACTUAL resolution limits of the retained reflections
        # (0/0 in the arguments means "no filter was requested")
        d_min=float(np.min(d)) if len(d) else float(d_min),
        d_max=float(np.max(d)) if len(d) else float(d_max),
    )
    # private bookkeeping: which gemmi atom each data.xyz row came from
    data._atom_sel = np.asarray(sel, dtype=np.int64)
    data._keep_hydrogens = bool(keep_hydrogens)

    make_sf_table(data)
    return data
