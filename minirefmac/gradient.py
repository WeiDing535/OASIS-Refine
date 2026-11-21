"""
minirefmac.gradient -- atomic-parameter gradients of the ML target using the
FULL-UNIT-CELL symmetry expansion (identical convention to sfcalc.calc_fc).

Implements the interface contract declared at the bottom of core.py:
    fc_and_atomic_gradients(data, xyz, b_iso, occ, G, fc_atom_factor,
                            chunk=48, want_fc=True)
        -> (fc, grad_xyz (M,3), grad_b (M,))

SYMMETRY CONVENTION (must match sfcalc.calc_fc exactly)
------------------------------------------------------
    Fc(h) = sum_op sum_j w_j(h) exp(2*pi*i h.(R_op x_j + t_op))
    w_j(h) = occ_j * sf_table[h,j] * exp(-b_iso_j * stol2[h])     (real)

Ops come from sfcalc._sym_ops(data) and already include centering translations;
R = op.rot/op.DEN, t = op.tran/op.DEN.  For P1 this collapses to the ASU-only
formula in core.py.  The phase is evaluated exactly as in calc_fc:
    xt  = x @ R.T + t          (fractional, (nc,3))
    ang = 2*pi * (hkl @ xt.T)  ((N,nc))
    E   = exp(i ang)

DERIVATION (Wirtinger; G_h = 2 d(-LL)/d fc_total*, and for a real parameter p
    df/dp = Re sum_h conj(G_h) * d fc_total_h/dp
with d fc_total/dp = fc_atom_factor(h) * d fc_atoms/dp, fc_atom_factor real):

Define, for atom j and reflection h,
    A_j(h)  = conj(G_h) * fc_atom_factor(h) * w_j(h)          (complex)
    S_j(h)  = sum_op E_op,j(h)                                (complex)
    V_jk(h) = sum_op (R_op^T h)_k * E_op,j(h)                 (complex)
Note (R^T h)_k = (h @ R)[k], and h.(R x + t) = (R^T h).x + h.t, so
    dE_op,j/dx_j,k = 2*pi*i (R^T h)_k E_op,j
    dw_j/db_j      = -stol2[h] w_j(h)
Then
    d fc_atoms/db_j     = -stol2[h] w_j(h) S_j(h)
    d fc_atoms/dx_j,k   = 2*pi*i w_j(h) V_jk(h)
and therefore
    df/dB_j   = -sum_h stol2[h] * Re[ A_j(h) S_j(h) ]
    df/dx_j,k = -2*pi * sum_h Im[ A_j(h) V_jk(h) ]
(The stol2 = s2/4 form is core.py's -0.25*sum s2 Re[C]; here C is the full
 symmetry sum.  Re[i*z] = -Im[z] turns the 2*pi*i into -2*pi*Im.)

The returned `fc` (want_fc=True) is sum_j w_j(h) S_j(h), i.e. numerically the
same quantity as sfcalc.calc_fc (verified to <1e-6 relative error).

MEMORY: atoms are processed in column chunks so no (N,M,nops) tensor is ever
built; the largest temporaries are (N, chunk) complex arrays per symmetry op
(N=11057, chunk=48, nops=4 -> ~2.1M complex numbers total, acceptable).
"""
import numpy as np

try:
    from . import sfcalc
except ImportError:  # pragma: no cover - allow running as a flat module
    import sfcalc


__all__ = ["fc_and_atomic_gradients"]

_TWO_PI = 2.0 * np.pi


def fc_and_atomic_gradients(data, xyz, b_iso, occ, G, fc_atom_factor,
                            chunk=48, want_fc=True):
    """Return (fc, grad_xyz, grad_b).

    fc          : (N,) complex128 (None if want_fc=False)
    grad_xyz    : (M,3) float64  -- d f / d x_j,k
    grad_b      : (M,)   float64  -- d f / d b_iso_j

    `G` is the (N,) complex gradient of the target w.r.t. fc_total*
    (from likelihood.negll_and_G); it MUST already be zero on free/missing
    reflections, otherwise the gradient would include the wrong reflection set.
    """
    n, m = data.n_ref, data.n_atoms
    xyz = np.asarray(xyz, dtype=np.float64)
    b_iso = np.asarray(b_iso, dtype=np.float64)
    if occ is None:
        occ = data.occ
    occ = np.asarray(occ, dtype=np.float64)
    G = np.asarray(G, dtype=np.complex128)
    fcf = np.asarray(fc_atom_factor, dtype=np.float64)   # (N,) real
    hkl = np.asarray(data.hkl, dtype=np.float64)         # (N,3)
    stol2 = np.asarray(data.stol2, dtype=np.float64)     # (N,)
    ops = sfcalc._sym_ops(data)                          # same ops as calc_fc

    # conj(G) * fc_atom_factor is independent of atom/op: the "C" kernel base.
    AG = np.conj(G) * fcf                               # (N,) complex

    grad_xyz = np.zeros((m, 3), dtype=np.float64)
    grad_b = np.zeros(m, dtype=np.float64)
    fc = np.zeros(n, dtype=np.complex128) if want_fc else None

    chunk = max(1, int(chunk))
    for c0 in range(0, m, chunk):
        c1 = min(c0 + chunk, m)
        nc = c1 - c0
        xc = xyz[c0:c1]                                 # (nc,3)
        bc = b_iso[c0:c1]                              # (nc,)
        oc = occ[c0:c1]                                # (nc,)
        sfc = data.sf_block(c0, c1)                     # (N,nc)

        # real weight w_j(h) = occ * f_j(h) * exp(-B_j stol2)
        W = (oc[None, :] * sfc) * np.exp(-bc[None, :] * stol2[:, None])
        # A_j(h) = conj(G) * fc_atom_factor * w_j(h)
        A = AG[:, None] * W                             # (N,nc) complex

        # accumulators over symmetry ops
        AS_re = np.zeros((n, nc), dtype=np.float64)    # sum_op Re[A*E]  (for dB)
        S = np.zeros((n, nc), dtype=np.complex128) if want_fc else None
        gx = np.zeros((nc, 3), dtype=np.float64)       # sum_op sum_h (R^T h)_k Im[A*E]

        for (R, t) in ops:
            xt = xc @ R.T + t                          # (nc,3) fractional
            ang = _TWO_PI * (hkl @ xt.T)               # (N,nc) -- same as calc_fc
            E = np.exp(1j * ang)                       # (N,nc)
            AE = A * E                                 # (N,nc) complex
            AS_re += AE.real
            AE_im = AE.imag                            # (N,nc) real
            hR = hkl @ R                               # (N,3) = (R^T h) rows
            for k in range(3):
                gx[:, k] += np.sum(hR[:, k][:, None] * AE_im, axis=0)
            if want_fc:
                S += E

        # df/dx_j,k = -2*pi * sum_h Im[A_j(h) V_jk(h)]
        grad_xyz[c0:c1] = -_TWO_PI * gx
        # df/dB_j = -sum_h stol2[h] * Re[A_j(h) S_j(h)]
        grad_b[c0:c1] = -np.sum(stol2[:, None] * AS_re, axis=0)
        if want_fc:
            fc += np.sum(W * S, axis=1)               # (N,) = sum_j w_j S_j

    return fc, grad_xyz, grad_b
