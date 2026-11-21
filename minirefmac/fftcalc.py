"""
minirefmac.fftcalc -- STEP 3: O(G log G) structure factor via real-space
Gaussian density on a grid + FFT + sampling at the observed Miller indices.

This replaces the O(N*M) direct reciprocal-space sum (fastkernel /
gradient.py) for the *structure factor value* Fc(h) with an FFT:

    For each symmetry-expanded atom j with element E, isotropic B_j and
    occupancy occ_j, its scattering in reciprocal space is

        A_j(h) = occ_j * f0_E(s^2) * exp(-B_j s^2)
               = occ_j * [ sum_i a_i exp(-b_i s^2) + c ] * exp(-B_j s^2)
               = sum_i occ_j a_i exp(-(b_i+B_j) s^2)      (4 Gaussian terms)
               + occ_j c   exp(-B_j s^2)                  (const term)

    Each reciprocal Gaussian  C*exp(-beta*s^2)  (s^2 = |S|^2/4, S in A^-1) is
    the Fourier transform (F(S)=int rho(r) e^{2pi i S.r} dr) of an isotropic
    REAL-SPACE Gaussian

        rho(r) = A * exp(-gamma |r - r_j|^2),   gamma = 4*pi^2/beta,
                                              A = C * (4*pi/beta)^{3/2}.

    So we scatter, for every atom, a sum of 5 isotropic real-space Gaussians
    onto a 3D grid covering one unit cell (all symmetry mates placed
    explicitly), run one real-to-complex FFT (gemmi.transform_map_to_f_phi),
    and sample at the observed HKL.  Cost: O(G log G) with G = grid voxels
    instead of O(N * M).

Gradients: the atomic gradient is obtained by the adjoint-FFT ("scatter the
derivative of the Gaussian") route -- see fft_gradients.

The absolute scale is fixed from the h=(0,0,0) term: after gemmi's FFT
normalisation F_fft(0) = scale * sum_j Z_j (total electron count), so
Fc(h) = (F_fft(0)/sum Z_j) * F_fft(h).  No direct-sum Fc is needed.
"""
import numpy as np
import gemmi

__all__ = ["fft_fc", "fft_gradients", "GAUSSIAN_TERMS", "fft_fc_scaled",
           "atom_radius"]


# ---------------------------------------------------------------------------
# Per-atom density block extent (adaptive radius)
# ---------------------------------------------------------------------------
# The real-space expansion of one atom is  sum_t A_t exp(-gamma_t |r|^2)  with
# gamma_t = 4 pi^2 / (b_t + B)  (and 4 pi^2 / B  for the constant term).  The
# *widest* Gaussian is the one with the *largest* beta -> smallest gamma, i.e.
# sigma_t = sqrt(beta_t) / (2 pi).  Since  b_t > 0,  beta_max is always
# b_max + B, so the cutoff radius scales as sqrt(b_max + B):
#
#     r(B) = clip( sigma_cut * sqrt(b_max + B) / (2 pi), RADIUS_MIN, RADIUS_CAP )
#
# RADIUS_SIGMA is set empirically, NOT from a "negligible tail" argument: the
# truncation loss is largely absorbed by the F(0)=Z scale calibration, so the
# useful criterion is agreement of Fc and the gradients with the direct sum.
# Measured against the direct sum on the three benchmark cases (spacing 0.5 A,
# radius sigma -> err_gx / err_fc):
#
#             3o2e (all B=35)      7enm (B 0.8-75)     5hhk (B 8-80)
#   s=4.0     2.6e-07 / 4.7e-05   3.2e-01 / 9.8e-04   6.8e-03 / 7.6e-04
#   s=3.0     6.6e-07 / 4.7e-05   3.2e-01 / 9.8e-04   6.8e-03 / 7.6e-04
#   s=2.5     6.5e-06 / 5.0e-05   3.2e-01 / 9.8e-04   6.8e-03 / 7.6e-04
#   fixed 2.8 1.5e-04 / 2.1e-04   3.2e-01 / 9.8e-04   6.8e-03 / 7.5e-04
#
# i.e. on 7enm/5hhk the radius is irrelevant (the grid-aliasing error of the
# sharp b_i terms dominates by 2-4 orders of magnitude), while 3o2e -- a small,
# uniformly B=35, low-aliasing structure -- is the sensitive case.  sigma=2.6
# sits where the truncation error is still >=100x below the aliasing error
# there, and the gather cost is block-size bound, so it is the knee of the
# curve.  High-B atoms get a correspondingly larger block (sqrt(b_max+B)),
# capped so a B=200 atom cannot blow up the block.
#
# _place_atom (forward scatter) and fft_gradients (adjoint gather) must use the
# SAME radius for every atom, otherwise the adjoint is not exact.
RADIUS_SIGMA = 2.6      # cutoff in units of the widest Gaussian's sigma
RADIUS_CAP = 5.0        # hard cap on the cutoff (A)
RADIUS_MIN = 2.5        # floor, so the sharpest atoms keep a few voxels


def atom_radius(B, terms, sigma=None, cap=None, rmin=None):
    """Real-space cutoff (A) of the density block scattered for one atom."""
    sigma = RADIUS_SIGMA if sigma is None else sigma
    cap = RADIUS_CAP if cap is None else cap
    rmin = RADIUS_MIN if rmin is None else rmin
    b_max = max([bi for (_, bi) in terms] + [0.0])
    beta_max = max(b_max + max(float(B), 0.0), 1e-3)
    r = sigma * np.sqrt(beta_max) / (2.0 * np.pi)
    return float(min(max(r, rmin), cap))


# ---------------------------------------------------------------------------
# Adjoint-FFT gradient
# ---------------------------------------------------------------------------
# Forward:  Fc(h) = voxel * N * ifftn(rho)[h]   (gemmi transform_map_to_f_phi
#           returns exactly this with voxel = V/N; i.e. Fc = V * ifftn(rho)).
# Adjoint:  given G(h) = dL/dFc*(h), the gradient w.r.t. the real density grid
#           is  gl[r] = voxel * real( fftn(G_grid)[r] )   (fftn == adjoint of
#           the unnormalised ifftn, up to the voxel scale).  G_grid is the full
#           reciprocal grid with Gbar(h)=conj(G)*fc_atom_factor placed at the
#           observed hkl (zero elsewhere).
#
# Then for each symmetry mate k of atom j (position r_jk = R_k x_j + t_k) the
# atomic gradients are the inner product of gl with the density basis and its
# derivatives w.r.t. occ / B / x:
#   d rho_k / d occ_j = sum_term A0 exp(-gamma d^2)                        (occ)
#   d rho_k / d B_j   = occ_j * sum_term A0 exp(-gamma d^2)/beta*(gamma d^2 - 1.5)
#   d rho_k / d x_j,a = occ_j * sum_term 2*gamma*A0*exp(-gamma d^2)
#                                   * (disp @ M^T @ R_k)_a
# where A0 = a_i*(4*pi/beta)^(3/2) (no occ), beta = b_i + B_j, gamma = 4*pi^2/beta,
# disp = (r - r_jk)_frac @ M (A displacement), M = orthographic matrix.
# This is the exact adjoint of _place_atom, so it shares the same O(G log G)
# cost plus a cheap O(N_atoms * block^3) gather.


def fft_gradients(data, xyz, b_iso, occ, G, fc_atom_factor, symops=None,
                  spacing=None, radius_A=None, want_fc=True, _xfm="fftn",
                  radius_sigma=None, grid=None):
    """Adjoint-FFT atomic gradients.  Same contract as
    gradient.fc_and_atomic_gradients:
        (fc, grad_xyz (M,3), grad_b (M,))
    G            : (N,) complex, dL/dFc_total*  (zeroed on free/missing)
    fc_atom_factor: (N,) real
    radius_A     : None -> per-atom adaptive radius (see atom_radius); a number
                   forces a fixed block radius for every atom.
    grid         : optional pre-scattered grid (see _scatter_density).  When
                   given the density is not rebuilt -- use this to share one
                   scatter between fft_fc and fft_gradients in a refine cycle.
    """
    if symops is None:
        from . import sfcalc
        symops = sfcalc._sym_ops(data)
    cell = data.cell
    xyz = np.asarray(xyz, dtype=np.float64)
    b_iso = np.asarray(b_iso, dtype=np.float64)
    occ = np.asarray(occ, dtype=np.float64)
    G = np.asarray(G, dtype=np.complex128)
    fcf = np.asarray(fc_atom_factor, dtype=np.float64)
    hkl = np.asarray(data.hkl, dtype=np.int32)
    m = xyz.shape[0]
    if spacing is None:
        spacing = auto_spacing(cell, getattr(data, "d_min", None))

    # 1. build real-space density grid (same as fft_fc forward)
    if grid is None:
        g, atoms, orth_mat = _scatter_density(
            data, spacing, symops, radius_A, radius_sigma,
            xyz=xyz, b_iso=b_iso, occ=occ)
    else:
        g = grid
        atoms = _expanded_atoms(data, symops, xyz=xyz, b_iso=b_iso, occ=occ)
        orth_mat = _orth_matrix(cell)
    nu, nv, nw = g.nu, g.nv, g.nw
    N = nu * nv * nw
    V = cell.volume
    voxel = V / N

    # 2. optional Fc from the same grid
    if want_fc:
        fg, scale = _absolute_scale(g, atoms)
        fc = fg.get_value_by_hkl(hkl).astype(np.complex128) * scale
    else:
        fc = None

    # 3. reciprocal gradient grid -> adjoint FFT -> real-space grid gradient
    # f = sum_h conj(G_h) * fcf_h * Fc_h.  With Fc[h] = voxel*sum_r rho[r] e^{+2pi i h.r},
    #   d f / d rho[r] = voxel * real( conj( fftn( conj(Gbar) )[r] ) )
    #                 = voxel * real( fftn( Gbar )[r] )   (real(conj z)==real z)
    # where Gbar = conj(G)*fcf, so conj(Gbar) = G*fcf (fcf real).  Hence the
    # reciprocal grid holds G*fcf (NOT conj(G)*fcf).
    Gbar = G * fcf                                # conj(conj(G)*fcf) = G*fcf
    G_grid = np.zeros((nu, nv, nw), dtype=np.complex128)
    hi = hkl[:, 0] % nu
    ki = hkl[:, 1] % nv
    li = hkl[:, 2] % nw
    G_grid[hi, ki, li] = Gbar
    xfm = getattr(np.fft, _xfm)
    gl = voxel * np.real(xfm(G_grid))             # (nu, nv, nw) real

    # 4. gather per atom / symmetry mate (occ fixed -> no occ gradient)
    grad_xyz = np.zeros((m, 3), dtype=np.float64)
    grad_b = np.zeros(m, dtype=np.float64)

    sv = np.array([cell.a / nu, cell.b / nv, cell.c / nw], dtype=np.float64)
    MR_list = [orth_mat.T @ np.asarray(R, dtype=np.float64) for (R, _t) in symops]

    # Canonical block geometry (term- and atom-independent), keyed by the block
    # half-extents (i0, j0, k0).  With the adaptive radius the extents vary per
    # atom, but only over a handful of distinct values, so cache per shape and
    # compute each only once.
    # For an interior atom the displacement of a block voxel from the atom is
    #   disp = base + off,  base = (m/nu) @ M  (canonical),  off = delta @ M
    # (delta = sub-voxel residual).  So d2 = base_sq + 2 base.off + |off|^2 and
    # disp@MR = base_MR + off@MR, with only a 3-vector `off` changing per atom.
    _geom_cache = {}
    _mr_cache = {}

    def _block_geom(i0, j0, k0):
        base, base_sq = _canonical_block(_geom_cache, i0, j0, k0, nu, nv, nw,
                                         orth_mat)
        key = (i0, j0, k0)
        bmr = _mr_cache.get(key)
        if bmr is None:
            bmr = [base @ MR for MR in MR_list]
            _mr_cache[key] = bmr
        return base, base_sq, bmr

    for j in range(m):
        elem = data.elements[j]
        xj = xyz[j]
        Bj = b_iso[j]
        ocj = occ[j]
        terms, c = GAUSSIAN_TERMS(elem)
        betas = np.array([bi + Bj for (_, bi) in terms] + [Bj], dtype=np.float64)
        amps0 = np.array([ai for (ai, _) in terms] + [c], dtype=np.float64)
        gammas = 4.0 * np.pi ** 2 / np.maximum(betas, 1e-3)
        A0 = amps0 * (4.0 * np.pi / betas) ** 1.5
        # stacked (5, nb) coefficient rows:  S_B = sum_t cB_t (gam_t d2 - 1.5) e,
        #                                    S2  = sum_t c2_t e
        cB = (A0 / betas)[:, None]
        c2 = (2.0 * A0 * gammas)[:, None]
        gcol = gammas[:, None]

        rad = atom_radius(Bj, terms, sigma=radius_sigma) \
            if radius_A is None else radius_A
        i0 = int(np.ceil(rad / sv[0]))
        j0 = int(np.ceil(rad / sv[1]))
        k0 = int(np.ceil(rad / sv[2]))
        base, base_sq, base_MR_all = _block_geom(i0, j0, k0)
        mI = np.arange(-i0, i0 + 1)
        mJ = np.arange(-j0, j0 + 1)
        mK = np.arange(-k0, k0 + 1)
        small_cell = not (2 * i0 + 2 <= nu and 2 * j0 + 2 <= nv and
                          2 * k0 + 2 <= nw)

        for s, (R, t) in enumerate(symops):
            R = np.asarray(R, dtype=np.float64)
            tt = np.asarray(t, dtype=np.float64)
            rjk = R.dot(xj) + tt
            rjk = rjk - np.floor(rjk)                     # wrap into [0,1)^3
            ci = int(round(rjk[0] * nu)) % nu
            cj = int(round(rjk[1] * nv)) % nv
            ck = int(round(rjk[2] * nw)) % nw

            if (not small_cell and
                    ci - i0 >= 0 and ci + i0 + 1 <= nu and
                    cj - j0 >= 0 and cj + j0 + 1 <= nv and
                    ck - k0 >= 0 and ck + k0 + 1 <= nw):
                off = np.array([ci / nu - rjk[0], cj / nv - rjk[1],
                                ck / nw - rjk[2]], dtype=np.float64) @ orth_mat
                d2 = base_sq + 2.0 * (base @ off) + float(off @ off)
                dispMR = base_MR_all[s] + (off @ MR_list[s])[None, :]
            else:
                fu = (ci + mI) / nu - rjk[0]
                fv = (cj + mJ) / nv - rjk[1]
                fw = (ck + mK) / nw - rjk[2]
                fu -= np.round(fu); fv -= np.round(fv); fw -= np.round(fw)
                FX, FY, FZ = np.meshgrid(fu, fv, fw, indexing="ij")
                frac_off = np.stack([FX.ravel(), FY.ravel(), FZ.ravel()], axis=1)
                disp = frac_off @ orth_mat                       # (block,3) A
                d2 = np.einsum("ij,ij->i", disp, disp)
                dispMR = disp @ MR_list[s]

            ii = (ci + mI) % nu
            jj = (cj + mJ) % nv
            kk = (ck + mK) % nw
            block_gl = gl[np.ix_(ii, jj, kk)].reshape(-1)

            # basis: S_B for dB, S2 for dx; dispMR is term-independent.
            # One exp() call on the (5, nb) stack instead of five on (nb,).
            w = gcol * d2                                 # (5, nb)
            E = np.exp(-w)
            grad_b[j] += ocj * float(np.dot(block_gl, (cB * (w - 1.5) * E).sum(0)))
            grad_xyz[j] += ocj * (dispMR.T @ (block_gl * (c2 * E).sum(0)))

    return fc, grad_xyz, grad_b



def _it92_terms(elem):
    """Return (list_of_(a,b), c) for the Cromer-Mann/IT92 form factor of elem."""
    coef = gemmi.Element(elem).it92
    a = list(coef.a)
    b = list(coef.b)
    return list(zip(a, b)), float(coef.c)


# cache IT92 expansion per element
_TERM_CACHE = {}


def GAUSSIAN_TERMS(elem):
    if elem not in _TERM_CACHE:
        _TERM_CACHE[elem] = _it92_terms(elem)
    return _TERM_CACHE[elem]


def _build_grid(data, spacing):
    """Create a gemmi.FloatGrid covering one unit cell with the given spacing."""
    cell = data.cell
    g = gemmi.FloatGrid()
    g.set_unit_cell(cell)
    g.set_size_from_spacing(spacing, gemmi.GridSizeRounding.Nearest)
    return g


def auto_spacing(cell, d_min=None, target=0.5, vmax=80e6):
    """Memory-aware grid spacing for FFT structure factors.

    The grid must resolve the *sharpest real-space Gaussian* (controlled by the
    smallest b_i term and the smallest B_iso), independent of the data
    resolution: a sharp atom aliases at any d_min.  Empirically ~0.5 A gives a
    global Fc error ~1e-3 (0.3 A -> ~2.5e-4), which is plenty for training-data
    phases while keeping the O(G log G) speed win on large reflection sets.
    `target` is inflated only if the grid would exceed `vmax` voxels (avoids OOM
    on very large unit cells).  Pass an explicit `spacing` to override.
    """
    a, b, c = cell.a, cell.b, cell.c
    sp = float(target)
    for _ in range(40):
        nu = max(1, int(round(a / sp)))
        nv = max(1, int(round(b / sp)))
        nw = max(1, int(round(c / sp)))
        if nu * nv * nw <= vmax:
            break
        sp *= 1.1
    return sp


def _expanded_atoms(data, symops, xyz=None, b_iso=None, occ=None):
    """Yield (element, fractional_xyz, B_iso, occ) for every symmetry mate
    (and the original atom) placed inside one unit cell.

    `xyz`/`b_iso`/`occ` override the values stored on `data`, so a caller that
    is evaluating a *trial* model (e.g. an L-BFGS step) scatters the trial
    coordinates rather than the last accepted ones.
    """
    out = []
    xyzc = np.asarray(data.xyz if xyz is None else xyz, dtype=np.float64)
    b = np.asarray(data.b_iso if b_iso is None else b_iso, dtype=np.float64)
    occ = np.asarray(data.occ if occ is None else occ, dtype=np.float64)
    elems = list(data.elements)
    for j in range(data.n_atoms):
        x = xyzc[j]
        for (R, t) in symops:
            R = np.asarray(R, dtype=np.float64)
            tt = np.asarray(t, dtype=np.float64)
            xs = R.dot(x) + tt
            xs = xs - np.floor(xs)            # wrap into [0,1)^3
            out.append((elems[j], xs, b[j], occ[j]))
    return out


def _orth_matrix(cell):
    """Return the 3x3 array M such that cartesian = fracs @ M for row-vector fracs."""
    e1 = cell.orthogonalize(gemmi.Fractional(1, 0, 0))
    e2 = cell.orthogonalize(gemmi.Fractional(0, 1, 0))
    e3 = cell.orthogonalize(gemmi.Fractional(0, 0, 1))
    return np.array([[e1.x, e2.x, e3.x],
                     [e1.y, e2.y, e3.y],
                     [e1.z, e2.z, e3.z]], dtype=np.float64)


def _place_atom(g, cell, orth_mat, elem, xj, B, occ, spacing, radius_A=None,
                radius_sigma=None, geom=None):
    """Scatter one atom's 5 real-space Gaussians onto the gemmi FloatGrid.

    The form factor f0(s^2) = sum_i a_i exp(-b_i s^2) + c, multiplied by the
    Debye-Waller exp(-B s^2), becomes sum_i a_i exp(-(b_i+B) s^2) + c exp(-B s^2)
    in reciprocal space.  Each term C*exp(-beta s^2) is the FT of a real-space
    Gaussian  A*exp(-gamma|r-r_j|^2)  with  gamma = 4*pi^2/beta,
    A = C*(4*pi/beta)^{3/2}.  So the constant c is a Gaussian of width B
    (thermal broadening of the core), correctly handled here.

    Offsets are computed relative to the nearest grid index so the atom centre
    lands at displacement 0 rather than at a half-voxel (0.5) shift.
    `radius_A=None` selects the per-atom adaptive cutoff (see atom_radius).

    `geom` is an optional dict caching the canonical block geometry per block
    shape -- pass the same dict for all atoms of one run (see _scatter_density).
    With it, the per-atom work is reduced to a 3-vector `off`: the voxel
    displacements are  disp = base + off,  base = (m/nu) @ M,  off = delta @ M,
    hence  d2 = base_sq + 2 base.off + |off|^2.
    """
    terms, c = GAUSSIAN_TERMS(elem)
    nu, nv, nw = g.nu, g.nv, g.nw
    arr = g.array  # (nu, nv, nw) float32 -- a live view of the grid buffer

    radius_A = atom_radius(B, terms, sigma=radius_sigma) \
        if radius_A is None else radius_A

    betas = np.array([bi + B for (_, bi) in terms] + [B], dtype=np.float64)
    amps = np.array([occ * ai for (ai, _) in terms] + [occ * c], dtype=np.float64)
    gammas = 4.0 * np.pi ** 2 / np.maximum(betas, 1e-3)
    # term amplitudes as one (5,1) row:  dens = sum_t Acoef_t exp(-gam_t d2)
    Acoef = (amps * (4.0 * np.pi / betas) ** 1.5)[:, None]
    gcol = gammas[:, None]

    sv = np.array([cell.a / nu, cell.b / nv, cell.c / nw], dtype=np.float64)
    i0 = int(np.ceil(radius_A / sv[0]))
    j0 = int(np.ceil(radius_A / sv[1]))
    k0 = int(np.ceil(radius_A / sv[2]))

    # nearest grid index to the atom centre (voxel centres taken at i/nu)
    ci = int(round(xj[0] * nu)) % nu
    cj = int(round(xj[1] * nv)) % nv
    ck = int(round(xj[2] * nw)) % nw

    # interior: the whole block fits and the fractional offsets stay inside
    # (-1/2, 1/2) so no periodic wrap is needed -> canonical base + 3-vector off
    if (2 * i0 + 2 <= nu and 2 * j0 + 2 <= nv and 2 * k0 + 2 <= nw and
            ci - i0 >= 0 and ci + i0 + 1 <= nu and
            cj - j0 >= 0 and cj + j0 + 1 <= nv and
            ck - k0 >= 0 and ck + k0 + 1 <= nw):
        base, base_sq = _canonical_block(geom, i0, j0, k0, nu, nv, nw, orth_mat)
        off = np.array([ci / nu - xj[0], cj / nv - xj[1], ck / nw - xj[2]],
                       dtype=np.float64) @ orth_mat
        d2 = base_sq + 2.0 * (base @ off) + float(off @ off)
        dens = (Acoef * np.exp(-(gcol * d2))).sum(0).astype(np.float32)
        arr[ci - i0:ci + i0 + 1, cj - j0:cj + j0 + 1,
            ck - k0:ck + k0 + 1] += dens.reshape(2 * i0 + 1, 2 * j0 + 1, 2 * k0 + 1)
        return

    # boundary / small-cell fallback: explicit wrapped offsets + np.add.at
    mI = np.arange(-i0, i0 + 1)
    mJ = np.arange(-j0, j0 + 1)
    mK = np.arange(-k0, k0 + 1)
    fu = (ci + mI) / nu - xj[0]
    fv = (cj + mJ) / nv - xj[1]
    fw = (ck + mK) / nw - xj[2]
    fu = fu - np.round(fu)
    fv = fv - np.round(fv)
    fw = fw - np.round(fw)
    FU, FV, FW = np.meshgrid(fu, fv, fw, indexing="ij")
    frac_off = np.stack([FU.ravel(), FV.ravel(), FW.ravel()], axis=1)
    disp = frac_off @ orth_mat
    r2 = np.einsum("ij,ij->i", disp, disp)          # A^2
    dens = (Acoef * np.exp(-(gcol * r2))).sum(0).astype(np.float32)
    dens = dens.reshape(FU.shape)
    ii = (ci + mI) % nu
    jj = (cj + mJ) % nv
    kk = (ck + mK) % nw
    np.add.at(arr, np.ix_(ii, jj, kk), dens)


def _canonical_block(geom, i0, j0, k0, nu, nv, nw, orth_mat):
    """Cached canonical voxel-offset geometry for a (2i0+1,2j0+1,2k0+1) block."""
    if geom is None:
        mI = np.arange(-i0, i0 + 1)
        mJ = np.arange(-j0, j0 + 1)
        mK = np.arange(-k0, k0 + 1)
        MX, MY, MZ = np.meshgrid(mI / nu, mJ / nv, mK / nw, indexing="ij")
        base = np.stack([MX.ravel(), MY.ravel(), MZ.ravel()], axis=1) @ orth_mat
        return base, np.einsum("ij,ij->i", base, base)
    key = (i0, j0, k0)
    cg = geom.get(key)
    if cg is None:
        mI = np.arange(-i0, i0 + 1)
        mJ = np.arange(-j0, j0 + 1)
        mK = np.arange(-k0, k0 + 1)
        MX, MY, MZ = np.meshgrid(mI / nu, mJ / nv, mK / nw, indexing="ij")
        base = np.stack([MX.ravel(), MY.ravel(), MZ.ravel()], axis=1) @ orth_mat
        cg = (base, np.einsum("ij,ij->i", base, base))
        geom[key] = cg
    return cg


def _scatter_density(data, spacing, symops, radius_A=None, radius_sigma=None,
                     xyz=None, b_iso=None, occ=None):
    """Build the full real-space density grid (all atoms + symmetry mates).

    Returns (grid, atoms, orth_mat).  Reusable by fft_fc and fft_gradients so a
    refine cycle scatters the density only once.
    """
    cell = data.cell
    g = _build_grid(data, spacing)
    orth_mat = _orth_matrix(cell)
    atoms = _expanded_atoms(data, symops, xyz=xyz, b_iso=b_iso, occ=occ)
    geom = {}
    for (elem, xj, B, occ_j) in atoms:
        _place_atom(g, cell, orth_mat, elem, xj, B, occ_j, spacing, radius_A,
                    radius_sigma=radius_sigma, geom=geom)
    return g, atoms, orth_mat


def _absolute_scale(g, atoms):
    """Scale factor bringing the grid FFT onto the absolute (electron) scale,
    fixed by F(0) == total electron count."""
    fg = gemmi.transform_map_to_f_phi(g, half_l=True)
    z_tot = 0.0
    for (elem, _, _, oc) in atoms:
        z_tot += oc * gemmi.Element(elem).atomic_number
    f0 = fg.get_value_by_hkl(np.array([[0, 0, 0]], dtype=np.int32))[0]
    return fg, (f0 / z_tot if z_tot != 0 else 1.0)


def fft_fc(data, spacing=None, symops=None, radius_A=None, return_grid=False,
           radius_sigma=None, xyz=None, b_iso=None, occ=None, grid=None,
           atoms=None):
    """Compute Fc(h) for all observed reflections via grid FFT.

    Returns a complex128 numpy array of length data.n_ref on the absolute
    scale (Fc(0) == total electron count).  `spacing=None` selects a
    memory-aware spacing (~0.3 A) that keeps the Fc integral error small.
    `radius_A=None` uses the per-atom adaptive density cutoff (atom_radius).
    `xyz`/`b_iso`/`occ` override data.xyz/data.b_iso/data.occ (trial model);
    `grid`/`atoms` reuse a grid already built by _scatter_density.
    """
    if symops is None:
        from . import sfcalc
        symops = sfcalc._sym_ops(data)
    cell = data.cell
    if spacing is None:
        spacing = auto_spacing(cell, getattr(data, "d_min", None))

    if grid is None:
        g, atoms, _orth = _scatter_density(
            data, spacing, symops, radius_A, radius_sigma,
            xyz=xyz, b_iso=b_iso, occ=occ)
    else:
        g = grid
        if atoms is None:
            atoms = _expanded_atoms(data, symops, xyz=xyz, b_iso=b_iso, occ=occ)
    fg, scale = _absolute_scale(g, atoms)

    hkl = np.ascontiguousarray(data.hkl, dtype=np.int32)
    fc = fg.get_value_by_hkl(hkl).astype(np.complex128) * scale
    if return_grid:
        return fc, g, fg
    return fc


def fft_fc_scaled(data, direct_fc, spacing=0.7, symops=None, radius_A=None,
                  radius_sigma=None):
    """Like fft_fc but calibrate the scale against a trusted direct-sum Fc
    (used only for validation/benchmarking)."""
    fc = fft_fc(data, spacing=spacing, symops=symops, radius_A=radius_A,
                radius_sigma=radius_sigma)
    # best complex scale s.t. fc_cal == s*fc matches direct_fc
    s = np.vdot(direct_fc, fc) / np.vdot(fc, fc)
    return fc * s
