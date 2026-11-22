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
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np

try:
    from . import sfcalc
except ImportError:  # pragma: no cover
    import sfcalc


__all__ = ["simd_fc_and_gradients", "ensure_simd_lib", "HAVE_SIMD",
           "probe", "cpu_has_avx512"]

_TWO_PI = 2.0 * np.pi
_P = ctypes.c_void_p
_I = ctypes.c_int

_lib = None
HAVE_SIMD = False


def _lib_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_simd.so")


def probe():
    """Can the AVX-512 kernel be used here?  Returns ``(ok, reason)``.

    Never raises and never compiles: a stale ``_simd.so`` from another OS/CPU
    is reported as rebuildable rather than fatal.
    """
    if not cpu_has_avx512():
        return False, "CPU lacks AVX-512F/DQ/VL and FMA"
    if HAVE_SIMD or _read_library(_lib_path()) is not None:
        return True, "prebuilt _simd.so"
    cc = shutil.which("clang") or shutil.which("gcc")
    if cc is None:
        return False, "no compatible _simd.so and no C compiler (clang/gcc)"
    return True, "compiles _simd.c with %s" % os.path.basename(cc)


def cpu_has_avx512():
    """Check the required CPU flags on all processors visible to Linux."""
    required = {"avx512f", "avx512dq", "avx512vl", "fma"}
    try:
        with open("/proc/cpuinfo", "r") as fh:
            flags = [line.partition(":")[2].split()
                     for line in fh if line.partition(":")[0].strip() == "flags"]
    except OSError:
        return False
    return bool(flags) and all(required.issubset(cpu) for cpu in flags)


def _read_library(path):
    try:
        lib = ctypes.CDLL(path)
        panel = lib.simd_fc_grad_panel_v2
        panel.argtypes = [_P] * 8 + [_I] * 2 + [_P] * 4
        panel.restype = None
    except (OSError, AttributeError):
        return None
    return lib


def ensure_simd_lib():
    """Load a compatible SIMD library, compiling and publishing it if needed."""
    global _lib, HAVE_SIMD
    if _lib is not None:
        return True
    if not cpu_has_avx512():
        return False
    so = _lib_path()
    lib = _read_library(so)
    if lib is None:
        cc = shutil.which("clang") or shutil.which("gcc")
        if cc is None:
            return False
        src = os.path.join(os.path.dirname(so), "_simd.c")
        try:
            with tempfile.TemporaryDirectory(prefix=".simd-", dir=os.path.dirname(so)) as tmp:
                built = os.path.join(tmp, "_simd.so")
                result = subprocess.run(
                    [cc, "-O3", "-mavx512f", "-mavx512dq", "-mavx512vl",
                     "-mfma", "-funroll-loops", "-shared", "-fPIC",
                     "-o", built, src], capture_output=True, text=True)
                if result.returncode:
                    print("simdkernel: build failed:\n", result.stderr)
                    return False
                lib = _read_library(built)
                if lib is None:
                    return False
                os.replace(built, so)
        except OSError as exc:
            print("simdkernel: build error:", exc)
            return False
    _lib = lib
    HAVE_SIMD = True
    return True


def _load():
    if not ensure_simd_lib():
        raise RuntimeError("AVX-512 SIMD library unavailable on this machine")
    return _lib


def simd_fc_and_gradients(data, xyz, b_iso, occ, G, fc_atom_factor,
                          chunk=48, exp_mode=None, *, want_fc=True, want_grad=True,
                          threads=1):
    """Return (fc, grad_xyz, grad_b) with the AVX-512 inner loop.

    Same contract as fastkernel.fused_fc_and_gradients: for a GIVEN ``G`` the
    returned arrays match the numpy reference to ~1e-13 (SIMD sincos accuracy);
    pass the SAME G to both to compare.  exp_mode is accepted only for API
    compatibility with the numpy kernel and is ignored. Disabled outputs are
    returned as None. With multiple workers, Fc-only calls partition
    reflections, preserving the atom summation order for each reflection.
    Gradient calls partition atoms.
    """
    n, m = data.n_ref, data.n_atoms
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    b_iso = np.ascontiguousarray(b_iso, dtype=np.float64)
    if occ is None:
        occ = data.occ
    occ = np.ascontiguousarray(occ, dtype=np.float64)
    stol2 = np.ascontiguousarray(data.stol2, dtype=np.float64)
    ops = sfcalc._sym_ops(data)
    lib = _load()

    ag_re = ag_im = None
    if want_grad:
        ag = np.conj(np.asarray(G, dtype=np.complex128)) * np.real(fc_atom_factor)
        ag_re = np.ascontiguousarray(ag.real).ctypes.data_as(_P)
        ag_im = np.ascontiguousarray(ag.imag).ctypes.data_as(_P)
    # hkl must be COLUMN-MAJOR (3 x n_ref) for the SIMD kernel; each Miller
    # index column hklT[k*n_ref+n] = hkl[n,k] is contiguous.
    hklT = np.ascontiguousarray(data.hkl.T, dtype=np.float64)

    grad_xyz = np.zeros((m, 3), dtype=np.float64) if want_grad else None
    grad_b = np.zeros(m, dtype=np.float64) if want_grad else None
    op_ptrs = [(R.ctypes.data_as(_P), t.ctypes.data_as(_P)) for R, t in ops]

    chunk = max(1, int(chunk))
    n_chunks = (m + chunk - 1) // chunk
    n_tiles = (n + 63) // 64
    split_reflections = want_fc and not want_grad
    workers = min(max(1, int(threads)), max(1, n_tiles if split_reflections else n_chunks))

    def panels(first, last, ref_start=0, ref_stop=n):
        n_local = ref_stop - ref_start
        local_hkl = np.ascontiguousarray(hklT[:, ref_start:ref_stop])
        local_stol2 = stol2[ref_start:ref_stop]
        hkl_ptr = local_hkl.ctypes.data_as(_P)
        stol2_ptr = local_stol2.ctypes.data_as(_P)
        fc_re = np.zeros(n_local, dtype=np.float64) if want_fc else None
        fc_im = np.zeros(n_local, dtype=np.float64) if want_fc else None
        fc_re_ptr = fc_re.ctypes.data_as(_P) if want_fc else None
        fc_im_ptr = fc_im.ctypes.data_as(_P) if want_fc else None
        for c0 in range(first * chunk, last * chunk, chunk):
            c1 = min(c0 + chunk, m)
            nc = c1 - c0
            xc = xyz[c0:c1]
            bc = b_iso[c0:c1]
            oc = occ[c0:c1]
            if n_local == n:
                sfc = data.sf_block(c0, c1)
            else:
                sfc = data.sf_block(c0, c1, slice(ref_start, ref_stop))
            W = (oc[None, :] * sfc) * np.exp(-bc[None, :] * local_stol2[:, None])
            Wt = np.ascontiguousarray(W.T, dtype=np.float64)
            # The gradient slices belong exclusively to this worker.
            gx_chunk = grad_xyz[c0:c1].ctypes.data_as(_P) if want_grad else None
            gb_chunk = grad_b[c0:c1].ctypes.data_as(_P) if want_grad else None
            xc_ptr = xc.ctypes.data_as(_P)
            wt_ptr = Wt.ctypes.data_as(_P)
            for R_ptr, t_ptr in op_ptrs:
                lib.simd_fc_grad_panel_v2(
                    hkl_ptr, ag_re, ag_im, R_ptr, t_ptr,
                    xc_ptr, wt_ptr, stol2_ptr, n_local, nc, fc_re_ptr, fc_im_ptr,
                    gx_chunk, gb_chunk)
        return fc_re, fc_im

    if workers == 1:
        fc_re, fc_im = panels(0, n_chunks)
    elif split_reflections:
        edges = [min(n, (i * n_tiles // workers) * 64) for i in range(workers + 1)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(panels, 0, n_chunks, start, stop)
                       for start, stop in zip(edges, edges[1:])]
            real, imag = zip(*(f.result() for f in futures))
        fc_re, fc_im = np.concatenate(real), np.concatenate(imag)
    else:
        boundaries = [i * n_chunks // workers for i in range(workers + 1)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            partials = pool.map(panels, boundaries[:-1], boundaries[1:])
            fc_re, fc_im = next(partials)
            for real, imag in partials:
                if want_fc:
                    fc_re += real
                    fc_im += imag

    if want_grad:
        grad_xyz *= -_TWO_PI
    fc = fc_re + 1j * fc_im if want_fc else None
    return fc, grad_xyz, grad_b
