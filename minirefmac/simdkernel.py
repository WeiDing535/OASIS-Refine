"""
minirefmac.simdkernel -- STEP 2 acceleration: AVX-512 SIMD fused Fc+gradient.

This module is the C-accelerated twin of fastkernel.fused_fc_and_gradients.
It computes EXACTLY the same mathematics (full-unit-cell reciprocal-space
direct sum over all symmetry ops, identical W / A / gradient formulas) but the
per-(symop, atom-chunk) inner loop -- the complex exponential exp(2*pi*i*ang)
and the Fc / gradient accumulations -- runs in hand-written AVX-512 C
(see _simd.c).  The only numerical difference is the custom SIMD sincos
(range reduction + degree-8 minimax poly), accurate to ~1e-14 absolute.

Public API (drop-in for fastkernel.fused_fc_and_gradients):
    simd_fc_and_gradients(data, xyz, b_iso, occ, G, fc_atom_factor,
                          chunk=48) -> (fc, grad_xyz, grad_b)

The shared object _simd.so is built on demand by ensure_simd_lib() if missing.
"""
import os
import ctypes
import numpy as np

try:
    from . import sfcalc
except ImportError:  # pragma: no cover
    import sfcalc


__all__ = ["simd_fc_and_gradients", "ensure_simd_lib", "HAVE_SIMD"]

_TWO_PI = 2.0 * np.pi
_P = ctypes.c_void_p
_I = ctypes.c_int

_lib = None
HAVE_SIMD = False


def _lib_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_simd.so")


def ensure_simd_lib():
    """Build _simd.so if it does not exist. Returns True if usable."""
    global _lib, HAVE_SIMD
    if _lib is not None:
        return HAVE_SIMD
    so = _lib_path()
    if not os.path.exists(so):
        try:
            import subprocess
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_simd.c")
            import shutil
            cc = shutil.which("clang") or shutil.which("gcc")
            if cc is None:
                print("simdkernel: no C compiler found (clang / gcc)")
                HAVE_SIMD = False
                return False
            r = subprocess.run(
                [cc, "-O3", "-mavx512f", "-mavx512dq", "-mavx512vl",
                 "-mfma", "-funroll-loops", "-shared", "-fPIC",
                 "-o", so, src], capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(so):
                print("simdkernel: build failed:\n", r.stderr)
                HAVE_SIMD = False
                return False
        except Exception as exc:
            print("simdkernel: build error:", exc)
            HAVE_SIMD = False
            return False
    try:
        lib = ctypes.CDLL(so)
        lib.simd_fc_grad_panel.argtypes = [
            _P, _P, _P, _P, _P, _P, _P, _P, _I, _I, _P, _P, _P, _P]
        lib.simd_fc_grad_panel.restype = None
    except (OSError, AttributeError) as exc:
        # A stale shared object from another OS / CPU cannot be loaded.
        # Nothing in the default refinement path needs it; removing the file
        # makes ensure_simd_lib() rebuild it locally from _simd.c on next use.
        print("simdkernel: cannot load %s (%s). Remove it to rebuild from "
              "_simd.c, or run without --simd-kernel." % (so, exc))
        HAVE_SIMD = False
        return False
    _lib = lib
    HAVE_SIMD = True
    return True


def _load():
    if not ensure_simd_lib():
        raise RuntimeError("AVX-512 SIMD library unavailable on this machine")
    return _lib


def simd_fc_and_gradients(data, xyz, b_iso, occ, G, fc_atom_factor,
                          chunk=48, exp_mode=None):
    """Return (fc, grad_xyz, grad_b) with the AVX-512 inner loop.

    Same contract as fastkernel.fused_fc_and_gradients: for a GIVEN ``G`` the
    returned arrays match the numpy reference to ~1e-13 (SIMD sincos accuracy);
    pass the SAME G to both to compare.  exp_mode is accepted only for API
    compatibility with the numpy kernel and is ignored (the C path is exact).
    """
    n, m = data.n_ref, data.n_atoms
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    b_iso = np.ascontiguousarray(b_iso, dtype=np.float64)
    if occ is None:
        occ = data.occ
    occ = np.ascontiguousarray(occ, dtype=np.float64)
    G = np.ascontiguousarray(G, dtype=np.complex128)
    fcf = np.ascontiguousarray(fc_atom_factor, dtype=np.float64)
    hkl = np.ascontiguousarray(data.hkl, dtype=np.float64)
    stol2 = np.ascontiguousarray(data.stol2, dtype=np.float64)
    ops = sfcalc._sym_ops(data)
    lib = _load()

    AG = np.conj(G) * fcf
    AG_re = np.ascontiguousarray(AG.real, dtype=np.float64)
    AG_im = np.ascontiguousarray(AG.imag, dtype=np.float64)
    # hkl must be COLUMN-MAJOR (3 x n_ref) for the SIMD kernel; each Miller
    # index column hklT[k*n_ref+n] = hkl[n,k] is contiguous.
    hklT = np.ascontiguousarray(data.hkl.T, dtype=np.float64)

    fc_re = np.zeros(n, dtype=np.float64)
    fc_im = np.zeros(n, dtype=np.float64)
    grad_xyz = np.zeros((m, 3), dtype=np.float64)
    grad_b = np.zeros(m, dtype=np.float64)

    chunk = max(1, int(chunk))
    for c0 in range(0, m, chunk):
        c1 = min(c0 + chunk, m)
        nc = c1 - c0
        xc = np.ascontiguousarray(xyz[c0:c1], dtype=np.float64)     # (nc,3)
        bc = b_iso[c0:c1]
        oc = occ[c0:c1]
        sfc = data.sf_block(c0, c1)                                # (N,nc)
        W = (oc[None, :] * sfc) * np.exp(-bc[None, :] * stol2[:, None])  # (N,nc)
        Wt = np.ascontiguousarray(W.T, dtype=np.float64)           # (nc,N)
        gx_chunk = np.zeros((nc, 3), dtype=np.float64)
        gb_chunk = np.zeros(nc, dtype=np.float64)
        for (R, t) in ops:
            Rarr = np.ascontiguousarray(np.asarray(R, dtype=np.float64).reshape(3, 3),
                                        dtype=np.float64)
            tarr = np.ascontiguousarray(np.asarray(t, dtype=np.float64).reshape(3),
                                       dtype=np.float64)
            lib.simd_fc_grad_panel(
                hklT.ctypes.data_as(_P), AG_re.ctypes.data_as(_P), AG_im.ctypes.data_as(_P),
                Rarr.ctypes.data_as(_P), tarr.ctypes.data_as(_P),
                xc.ctypes.data_as(_P), Wt.ctypes.data_as(_P), stol2.ctypes.data_as(_P),
                _I(n), _I(nc),
                fc_re.ctypes.data_as(_P), fc_im.ctypes.data_as(_P),
                gx_chunk.ctypes.data_as(_P), gb_chunk.ctypes.data_as(_P))
        grad_xyz[c0:c1] = -_TWO_PI * gx_chunk
        grad_b[c0:c1] = gb_chunk

    fc = fc_re + 1j * fc_im
    return fc, grad_xyz, grad_b
