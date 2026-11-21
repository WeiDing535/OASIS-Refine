"""
minirefmac.fastkernel -- FUSED Fc + atomic-gradient kernel (acceleration prototype).

================================================================================
WHAT THIS IS  --  KERNEL FUSION, NOT AN ALGORITHMIC CHANGE
================================================================================
This module does NOT change the mathematics of the structure-factor /
gradient calculation.  It is still the *exact reciprocal-space direct sum*

    Fc(h) = sum_op sum_j occ_j f_j(stol2) exp(-B_j stol2)
                    * exp(2*pi*i * h . (R_op x_j + t_op))

over every symmetry operator and every atom.  There is NO FFT, NO density
gridding, NO approximation, and NO change of symmetry convention.  The only
difference from the existing code is that the single most expensive temporary
-- the complex phase matrix

    ang = 2*pi * (hkl @ xt.T)          (N, nc)
    E   = exp(i * ang)                 (N, nc) complex128

is computed ONCE per (atom-chunk, symmetry-op) instead of TWICE.

THE REDUNDANCY THAT IS BEING REMOVED
------------------------------------
In refine.Refiner._make_target each L-BFGS objective evaluation does

    fc_atoms = sfcalc.calc_fc(d, xyz, b_iso, occ)        # exp(i*ang) pass #1
    fc_total = sfcalc.build_fc_total(...)
    negll, G, ... = likelihood.negll_and_G(...)           # G depends on Fc
    _, gx, gb = gradient.fc_and_atomic_gradients(
        d, xyz, b_iso, occ, G, fcf, want_fc=False)        # exp(i*ang) pass #2

The second call recomputes EXACTLY the same ang / E (same xyz, b_iso, occ,
same hkl, same symmetry ops) purely to differentiate Fc.  Fusing the two
passes into one loop halves the number of complex exponentials -- by far the
dominant cost of the direct sum on the reference machine (numpy with a
reference BLAS and no SIMD transcendental library).

THE DATA DEPENDENCY ON G  (important, read before wiring into refine.py)
-----------------------------------------------------------------------
The ML coefficient G = 2 d(-LL)/d fc_total* is produced by negll_and_G and
depends on the CURRENT Fc.  The atomic gradient is linear in conj(G):

    grad_b_j   = -sum_h stol2[h] * Re[ conj(G_h) fcf_h W_j(h) S_j(h) ]
    grad_x_jk  = -2*pi * sum_h  Im[ conj(G_h) fcf_h W_j(h) V_jk(h) ]

so the gradient accumulation needs G_h at the moment E is available.  Because
G is a function of the FULL Fc (all atom chunks summed), and Fc is only
complete at the END of the chunk loop, a strictly G-current, single-pass,
low-memory fusion is impossible without either (i) caching the (N, M, nops)
phase matrix (several GB for case3) or (ii) warm-starting G from the previous
objective evaluation.  This module therefore takes G as an EXPLICIT INPUT:
it is a pure, stateless kernel that, for a GIVEN G, returns the same
(fc, grad_xyz, grad_b) as calc_fc + gradient(want_fc=False) but computes the
phase once.  With the very same G handed to both paths the outputs agree to
~machine precision (verified in benchmarks/bench_accel.py).  refine.py uses a
warm-started G for the experimental fused path (Options.fused_kernel=True);
the default path is untouched.

COMPLEX-EXPONENTIAL VARIANTS  (see cexp / _USE_FAST_EXP)
-------------------------------------------------------
    (a) np.exp(1j * ang)                  -> complex128  (EXACT, DEFAULT)
    (b) np.cos(ang) + 1j*np.sin(ang)      -> complex128  (mathematically equal,
                                             ~1 ulp; kept for timing only)
    (c) ang.astype(float32) -> cos/sin    -> complex64 -> complex128
                                             (phase quantised to float32;
                                              DESTROYS ~1e-4 relative accuracy
                                              for large |ang|; NEVER default)

On the reference numpy build, np.exp(1j*ang) evaluates cos/sin in float64.
Variant (c) quantises the phase (which can be O(10^2..10^3) radians) to
float32, so its absolute phase error is O(|ang| * 2^-24); the resulting E
error is far above the 1e-10 tolerance required for refinement-grade Fc.
It is measured and reported as a NEGATIVE result, not enabled by default.
Set _USE_FAST_EXP = 'b' / 'c' (or env MINIREFMAC_FAST_EXP=b/c/1/true/f32)
to experiment; leave unset / 'a' / '0' for the exact path.

SYMMETRY & LOW MEMORY
---------------------
Full-unit-cell symmetry expansion uses sfcalc._sym_ops(data) (identical ops
to calc_fc, including lattice centering).  Scattering factors are fetched with
data.sf_block(c0, c1) so the (N, M) table is never materialised for large
cases (the low-memory sf_unique/elem_index gather works unchanged).
"""

import os
import numpy as np

try:
    from . import sfcalc
except ImportError:  # pragma: no cover - allow flat-module execution
    import sfcalc


__all__ = [
    "fused_fc_and_gradients",
    "cexp",
    "exp_variant_error",
    "EXP_MODES",
]

_TWO_PI = 2.0 * np.pi

#: complex-exponential implementations available for timing / experimentation
EXP_MODES = ("a", "b", "c")

#: ---------------------------------------------------------------------------
#: Optional module-level switch.  ``None`` / False / 'a'  -> exact (a) path.
#: 'b' -> cos+isin, 'c' -> float32 phase.  Overridden by the environment
#: variable MINIREFMAC_FAST_EXP at import time (unless the user reassigns it).
#: ---------------------------------------------------------------------------
_USE_FAST_EXP = None


def _normalise_mode(value):
    """Map a loose spelling to one of 'a' / 'b' / 'c' (default 'a')."""
    v = str(value).strip().lower()
    if v in ("", "0", "false", "off", "none", "a", "exact", "exp"):
        return "a"
    if v in ("1", "true", "on", "yes", "b", "cos", "fast"):
        return "b"
    if v in ("c", "f32", "float32", "phase32"):
        return "c"
    return "a"


def _init_from_env():
    """Initialise _USE_FAST_EXP from MINIREFMAC_FAST_EXP (import-time)."""
    global _USE_FAST_EXP
    env = os.environ.get("MINIREFMAC_FAST_EXP", "").strip()
    if env:
        _USE_FAST_EXP = _normalise_mode(env)


_init_from_env()


def _resolve_mode(exp_mode):
    """Effective exp mode: explicit arg > module switch > env > exact 'a'."""
    if exp_mode is not None:
        return _normalise_mode(exp_mode)
    if _USE_FAST_EXP:
        return _normalise_mode(_USE_FAST_EXP)
    return "a"


def cexp(ang, mode="a"):
    """Complex exponential exp(i*ang) under the requested variant.

    Parameters
    ----------
    ang  : (N, nc) float64 phase matrix 2*pi*(hkl @ xt.T)
    mode : 'a' | 'b' | 'c'  (see module docstring)

    Returns
    -------
    (N, nc) complex128.  Variants (b) and (a) are numerically ~identical;
    variant (c) quantises the phase to float32 and is NOT lossless.
    """
    m = _normalise_mode(mode)
    if m == "b":
        return np.cos(ang) + 1j * np.sin(ang)
    if m == "c":
        a32 = ang.astype(np.float32)
        return (np.cos(a32) + 1j * np.sin(a32)).astype(np.complex128)
    return np.exp(1j * ang)  # (a): exact reference path


def exp_variant_error(ang, mode, reference=None):
    """Numerical error of an exp variant relative to the exact (a) path.

    Returns (max_abs_err, rel_l2_err) of cexp(ang, mode) vs cexp(ang, 'a').
    Used by the benchmark to quantify why (c) must not be the default.
    """
    ref = cexp(ang, "a") if reference is None else reference
    got = np.asarray(cexp(ang, mode), dtype=np.complex128)
    diff = got - ref
    max_abs = float(np.max(np.abs(diff)))
    denom = float(np.linalg.norm(ref))
    rel_l2 = float(np.linalg.norm(diff) / denom) if denom > 0 else 0.0
    return max_abs, rel_l2


# ---------------------------------------------------------------------------
# Fused kernel
# ---------------------------------------------------------------------------
def fused_fc_and_gradients(data, xyz, b_iso, occ, G, fc_atom_factor,
                           chunk=48, exp_mode=None):
    """Return (fc, grad_xyz, grad_b) computing the phase/E ONCE.

    This is a pure KERNEL FUSION of sfcalc.calc_fc and
    gradient.fc_and_atomic_gradients(want_fc=False): for a GIVEN ``G`` the
    returned arrays are numerically identical (to ~1 ulp) to

        fc_ref = sfcalc.calc_fc(data, xyz, b_iso, occ, chunk=chunk)
        _, gx_ref, gb_ref = gradient.fc_and_atomic_gradients(
            data, xyz, b_iso, occ, G, fc_atom_factor,
            chunk=chunk, want_fc=False)

    but the complex phase matrix E = exp(2*pi*i*(hkl @ xt.T)) is evaluated a
    single time per (atom chunk, symmetry op) and shared between the Fc sum
    and the gradient accumulation.  It remains an EXACT direct reciprocal
    sum over all symmetry ops -- NO FFT, NO algorithmic approximation.

    Parameters
    ----------
    data            : RefineData (uses .hkl/.stol2/.sf_block/_sym_ops)
    xyz, b_iso, occ : (M,3) / (M,) / (M,) model parameters
    G               : (N,) complex128, 2*d(-LL)/d fc_total*, already zero on
                      free / missing reflections (same contract as gradient)
    fc_atom_factor  : (N,) real d fc_total / d fc_atoms
    chunk           : atom-column block size (default 48, matches gradient)
    exp_mode        : None -> honour _USE_FAST_EXP / env; or 'a'/'b'/'c'

    Notes on numerical identity
    ---------------------------
    * The Fc accumulation copies calc_fc EXACTLY: ``fc += sum(W*E, axis=1)``
      inside the symmetry-op loop (per-op reduction order preserved).
    * The gradient accumulation copies gradient EXACTLY: AS_re / gx reduced
      inside the op loop, then grad_xyz / grad_b formed per chunk.
    * W is computed once and reused for both fc and gradient.
    * Use exp_mode='a' (the default) for bit-level agreement; 'b' differs by
      ~1 ulp, 'c' is deliberately low precision (see module docstring).
    """
    n, m = data.n_ref, data.n_atoms
    xyz = np.asarray(xyz, dtype=np.float64)
    b_iso = np.asarray(b_iso, dtype=np.float64)
    if occ is None:
        occ = data.occ
    occ = np.asarray(occ, dtype=np.float64)
    G = np.asarray(G, dtype=np.complex128)
    # fc_atom_factor is real by construction (bulk-solvent factor); be
    # explicit so a stray complex array cannot be silently truncated.
    fcf = np.asarray(np.real(np.asarray(fc_atom_factor)), dtype=np.float64)
    hkl = np.asarray(data.hkl, dtype=np.float64)         # (N,3)
    stol2 = np.asarray(data.stol2, dtype=np.float64)     # (N,)
    ops = sfcalc._sym_ops(data)                          # full-cell ops
    mode = _resolve_mode(exp_mode)

    # conj(G) * fc_atom_factor: the atom/op-independent gradient "C" base.
    AG = np.conj(G) * fcf                               # (N,) complex

    fc = np.zeros(n, dtype=np.complex128)
    grad_xyz = np.zeros((m, 3), dtype=np.float64)
    grad_b = np.zeros(m, dtype=np.float64)

    chunk = max(1, int(chunk))
    for c0 in range(0, m, chunk):
        c1 = min(c0 + chunk, m)
        nc = c1 - c0
        xc = xyz[c0:c1]                                  # (nc,3)
        bc = b_iso[c0:c1]                               # (nc,)
        oc = occ[c0:c1]                                 # (nc,)
        sfc = data.sf_block(c0, c1)                      # (N,nc) low-mem ok

        # single real weight W_j(h) = occ * f_j(h) * exp(-B_j stol2)
        W = (oc[None, :] * sfc) * np.exp(-bc[None, :] * stol2[:, None])
        # A_j(h) = conj(G) * fc_atom_factor * W_j(h)
        A = AG[:, None] * W                             # (N,nc) complex

        AS_re = np.zeros((n, nc), dtype=np.float64)     # sum_op Re[A*E] (dB)
        gx = np.zeros((nc, 3), dtype=np.float64)        # sum_op sum_h (R^T h) Im[A*E]

        for (R, t) in ops:
            xt = xc @ R.T + t                           # (nc,3) fractional
            ang = _TWO_PI * (hkl @ xt.T)                # (N,nc) -- ONCE
            E = cexp(ang, mode)                         # (N,nc) complex128 -- ONCE

            # ---- Fc term (bit-identical reduction order to calc_fc) ----
            fc += np.sum(W * E, axis=1)

            # ---- gradient terms (bit-identical to gradient.py) ----
            AE = A * E                                  # (N,nc) complex
            AS_re += AE.real
            AE_im = AE.imag                            # (N,nc) real
            hR = hkl @ R                               # (N,3) = (R^T h) rows
            for k in range(3):
                gx[:, k] += np.sum(hR[:, k][:, None] * AE_im, axis=0)
            del AE, AE_im, E

        grad_xyz[c0:c1] = -_TWO_PI * gx
        grad_b[c0:c1] = -np.sum(stol2[:, None] * AS_re, axis=0)
        del W, A, AS_re, gx

    return fc, grad_xyz, grad_b
