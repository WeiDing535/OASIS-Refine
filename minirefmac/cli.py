"""
minirefmac.cli -- command line entry point.

Example
-------
python -m minirefmac.cli \
    --mtz input.mtz --pdb input.pdb --out-dir run1 \
    --ncyc 5 --maxiter 10 --solvent mask --weight-scale 0.05 \
    --map-scale absolute
"""

import argparse
import os
import sys
import time

try:
    from . import dataio, refine
except ImportError:  # pragma: no cover
    import dataio, refine


def build_parser():
    p = argparse.ArgumentParser(
        prog="minirefmac",
        description="From-scratch REFMAC5-like ML crystallographic refinement")
    p.add_argument("--mtz", required=True, help="input MTZ (F/SIGF/FreeR_flag)")
    p.add_argument("--pdb", required=True, help="input PDB or mmCIF model")
    p.add_argument("--out-dir", default="minirefmac_out",
                   help="directory for refined.pdb / refined.mtz / report.json / refine.log")
    p.add_argument("--tag", default="", help="prefix for output file names")

    # input columns
    p.add_argument("--fp", default=None, help="observed amplitude column (auto)")
    p.add_argument("--sigfp", default=None, help="sigma column (auto)")
    p.add_argument("--free", default=None, help="FreeR flag column (auto)")
    p.add_argument("--free-value", type=int, default=0,
                   help="flag value of the excluded/free set (convention: 0)")
    p.add_argument("--phib", default=None, help="experimental phase column (deg)")
    p.add_argument("--fom", default=None, help="figure-of-merit column")
    p.add_argument("--hla", default=None)
    p.add_argument("--hlb", default=None)
    p.add_argument("--hlc", default=None)
    p.add_argument("--hld", default=None)
    p.add_argument("--dmin", type=float, default=0.0)
    p.add_argument("--dmax", type=float, default=0.0)
    p.add_argument("--keep-hydrogens", action="store_true")

    # refinement protocol
    p.add_argument("--ncyc", type=int, default=5, help="outer cycles (REFMAC NCYC)")
    p.add_argument("--maxiter", type=int, default=10, help="L-BFGS iterations per cycle")
    p.add_argument("--bins", type=int, default=20, help="resolution bins for D/S")
    p.add_argument("--solvent", choices=["mask", "babinet", "none"], default="mask")
    p.add_argument("--no-aniso", action="store_true", help="disable anisotropic scaling")
    p.add_argument("--phase-prior", action="store_true",
                   help="use MLHL experimental phase prior (needs PHIB/FOM or HLA-HLD)")
    p.add_argument("--no-monlib", action="store_true",
                   help="use jelly-body fallback restraints instead of the "
                        "bundled monomer dictionary")

    # weights
    p.add_argument("--weight", type=float, default=None,
                   help="explicit w_xray (overrides automatic gradient balancing)")
    p.add_argument("--weight-scale", type=float, default=0.05,
                   help="multiplier on the automatic gradient-norm balance "
                        "(0.05 calibrated to REFMAC-like geometry; 1.0 over-fits)")
    p.add_argument("--no-auto-weight", action="store_true")
    p.add_argument("--w-bond", type=float, default=1.0)
    p.add_argument("--w-angle", type=float, default=1.0)
    p.add_argument("--w-b", type=float, default=1.0)
    p.add_argument("--w-vdw", type=float, default=1.0,
                   help="weight of the one-sided non-bonded VDW repulsion "
                        "(0 disables clashes control; sigma=0.2 A)")
    p.add_argument("--no-sym-vdw", action="store_true",
                   help="restrict VDW repulsion to the asymmetric unit only "
                        "(skip symmetry mates / neighbouring unit cells)")
    p.add_argument("--max-shift", type=float, default=0.5,
                   help="maximum coordinate shift in Angstrom per cycle")
    p.add_argument("--bmin", type=float, default=2.0)
    p.add_argument("--bmax", type=float, default=200.0)
    p.add_argument("--chunk", type=int, default=48)
    p.add_argument("--map-scale", choices=["absolute", "observed"],
                   default="absolute",
                   help="output MTZ scale: 'absolute' = REFMAC5 convention "
                        "(F/SIGF and map coefficients divided by k_total); "
                        "'observed' = legacy input scale")
    p.add_argument("--fused-kernel", action="store_true",
                   help="EXPERIMENTAL: compute the phase E once per target "
                        "evaluation and share it between Fc and gradients "
                        "(1.46-1.63x faster). Warm-started G lags one step, "
                        "so results differ slightly from the default path.")
    p.add_argument("--simd-kernel", action="store_true",
                   help="STEP 2: AVX-512 SIMD C inner loop for Fc+gradients "
                        "(simdkernel.py / _simd.c). ~6-11x faster than the "
                        "numpy fused kernel; same fused math, custom SIMD "
                        "sincos (~1e-14). Requires an AVX-512 CPU and a C "
                        "compiler; builds _simd.so from _simd.c on demand.")
    p.add_argument("--fft-kernel", action="store_true",
                   help="STEP 3: O(G log G) FFT structure factors (fftcalc.py): "
                        "grid electron density -> one FFT -> sample at HKL, "
                        "plus adjoint-FFT gradients. Faster than the direct sum "
                        "for large reflection sets; accuracy improves with a "
                        "finer --fft-spacing (default auto ~0.3 A).")
    p.add_argument("--fft-spacing", type=float, default=None,
                   help="FFT grid spacing in Angstrom (default: auto, "
                        "memory-aware ~0.3 A). Smaller = more accurate, slower.")
    p.add_argument("--fft-radius", type=float, default=None,
                   help="FFT density cut-off radius in Angstrom. Default: "
                        "per-atom adaptive (2.6 sigma of the widest Gaussian, "
                        "floored at 2.5 A and capped at 5 A). A number forces "
                        "a fixed radius.")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""

    o = refine.Options(
        ncyc=args.ncyc, maxiter=args.maxiter, n_bins=args.bins,
        solvent=args.solvent, fit_aniso=not args.no_aniso,
        use_phase_prior=args.phase_prior,
        use_monlib=not args.no_monlib,
        weight=args.weight, weight_scale=args.weight_scale,
        auto_weight=not args.no_auto_weight,
        w_bond=args.w_bond, w_angle=args.w_angle, w_b=args.w_b,
        w_vdw=args.w_vdw, sym_vdw=not args.no_sym_vdw,
        max_shift_a=args.max_shift, b_min=args.bmin, b_max=args.bmax,
        chunk=args.chunk, map_scale=args.map_scale,
        fused_kernel=args.fused_kernel, simd_kernel=args.simd_kernel,
        fft_kernel=args.fft_kernel, fft_spacing=args.fft_spacing,
        fft_radius=args.fft_radius,
        verbose=not args.quiet,
        out_pdb=os.path.join(args.out_dir, tag + "refined.pdb"),
        out_mtz=os.path.join(args.out_dir, tag + "refined.mtz"),
        out_log=os.path.join(args.out_dir, tag + "refine.log"),
        out_json=os.path.join(args.out_dir, tag + "report.json"),
    )

    print("Loading %s + %s ..." % (args.mtz, args.pdb), flush=True)
    t0 = time.time()
    data = dataio.load_data(
        args.mtz, args.pdb,
        fp_label=args.fp, sigfp_label=args.sigfp, free_label=args.free,
        free_value=args.free_value,
        phib_label=args.phib, fom_label=args.fom,
        hla_label=args.hla, hlb_label=args.hlb,
        hlc_label=args.hlc, hld_label=args.hld,
        d_min=args.dmin, d_max=args.dmax,
        keep_hydrogens=args.keep_hydrogens)
    print("  %d reflections, %d atoms, %s, %.2f-%.2f A, load %.2f s"
          % (data.n_ref, data.n_atoms, data.spacegroup_name,
             data.d_min, data.d_max, time.time() - t0), flush=True)
    if args.phase_prior and data.phib is None and data.hl is None:
        print("  WARNING: --phase-prior requested but no phase columns found; "
              "MLHL falls back to plain ML", flush=True)

    engine = refine.Refiner(data, o)
    result = engine.run()
    print("\nRESULT  R %.4f -> %.4f   Rfree %.4f -> %.4f   -LL %.1f -> %.1f   %.1f s"
          % (result["initial"]["r_work"], result["final"]["r_work"],
             result["initial"]["r_free"], result["final"]["r_free"],
             result["initial"]["negll"], result["final"]["negll"],
             result["wall_seconds"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
