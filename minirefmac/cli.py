"""
minirefmac.cli -- command line entry point.

Examples
--------
python -m minirefmac \
    --mtz input.mtz --pdb input.pdb --out-dir run1 \
    --ncyc 5 --maxiter 10 --solvent mask --weight-scale 0.05 \
    --map-scale absolute

python -m minirefmac --version
"""

import argparse
import os
import sys
import time

try:
    from . import __version__, dataio, refine
except ImportError:  # pragma: no cover - direct script use
    import dataio, refine
    __version__ = "0.1.0"

PROG = "minirefmac"


def build_parser():
    p = argparse.ArgumentParser(
        prog=PROG,
        description="From-scratch REFMAC5-like maximum-likelihood "
                    "crystallographic refinement (self-contained).",
        epilog="Kernels: 'auto' picks the AVX-512 SIMD path when the CPU and a "
               "C compiler allow it; numpy is always available.  All kernels "
               "share the same ML coefficients, so the trajectory is identical "
               "(see --kernel).")
    p.add_argument("--version", action="version",
                   version="%s %s" % (PROG, __version__))
    p.add_argument("--mtz", required=True, help="input MTZ (F/SIGF/FreeR_flag)")
    p.add_argument("--pdb", required=True, help="input PDB or mmCIF model")
    p.add_argument("--out-dir", default="minirefmac_out",
                   help="directory for refined.pdb / refined.mtz / "
                        "report.json / refine.log")
    p.add_argument("--tag", default="", help="prefix for output file names")

    # ---- input columns -------------------------------------------------
    g = p.add_argument_group("input columns (all auto-detected)")
    g.add_argument("--fp", default=None, help="observed amplitude column")
    g.add_argument("--sigfp", default=None, help="sigma column")
    g.add_argument("--free", default=None, help="FreeR flag column")
    g.add_argument("--free-value", default="auto",
                   help="flag value of the excluded (free) set.  'auto' "
                        "(default) honours the CCP4 convention (0 = free) when "
                        "flag 0 is a 1-25 %% minority, otherwise takes the "
                        "minority flag closest to 5 %%; an integer forces it")
    g.add_argument("--phib", default=None, help="experimental phase column (deg)")
    g.add_argument("--fom", default=None, help="figure-of-merit column")
    g.add_argument("--hla", default=None)
    g.add_argument("--hlb", default=None)
    g.add_argument("--hlc", default=None)
    g.add_argument("--hld", default=None)
    g.add_argument("--dmin", type=float, default=0.0,
                   help="high-resolution limit in A (0 = use everything)")
    g.add_argument("--dmax", type=float, default=0.0,
                   help="low-resolution limit in A (0 = use everything)")
    g.add_argument("--keep-hydrogens", action="store_true",
                   help="keep hydrogens instead of dropping them")

    # ---- protocol ------------------------------------------------------
    g = p.add_argument_group("refinement protocol")
    g.add_argument("--ncyc", type=int, default=5, help="outer cycles (REFMAC NCYC)")
    g.add_argument("--maxiter", type=int, default=10,
                   help="L-BFGS iterations per cycle")
    g.add_argument("--bins", type=int, default=20,
                   help="resolution bins for the ML parameters D/S")
    g.add_argument("--solvent", choices=["mask", "babinet", "none"],
                   default="mask", help="bulk-solvent model")
    g.add_argument("--no-aniso", action="store_true",
                   help="disable anisotropic scaling")
    g.add_argument("--phase-prior", action="store_true",
                   help="use the MLHL experimental phase prior "
                        "(needs PHIB/FOM or HLA-HLD)")
    g.add_argument("--no-monlib", action="store_true",
                   help="use jelly-body fallback restraints instead of the "
                        "bundled monomer dictionary")
    g.add_argument("--max-shift", type=float, default=0.5,
                   help="maximum coordinate shift in A per cycle")
    g.add_argument("--bmin", type=float, default=2.0, help="lower B bound")
    g.add_argument("--bmax", type=float, default=200.0, help="upper B bound")

    # ---- weights -------------------------------------------------------
    g = p.add_argument_group("weights")
    g.add_argument("--weight", type=float, default=None,
                   help="explicit w_xray (overrides automatic balancing)")
    g.add_argument("--weight-scale", type=float, default=0.05,
                   help="multiplier on the automatic gradient-norm balance "
                        "(0.05 calibrated to REFMAC-like geometry; 1.0 "
                        "over-fits the data)")
    g.add_argument("--no-auto-weight", action="store_true")
    g.add_argument("--w-bond", type=float, default=1.0)
    g.add_argument("--w-angle", type=float, default=1.0)
    g.add_argument("--w-b", type=float, default=1.0)
    g.add_argument("--w-vdw", type=float, default=1.0,
                   help="weight of the one-sided non-bonded VDW repulsion "
                        "(0 disables clash control; sigma=0.2 A)")
    g.add_argument("--no-sym-vdw", action="store_true",
                   help="restrict VDW repulsion to the asymmetric unit "
                        "(skip symmetry mates / neighbouring cells)")

    # ---- performance ---------------------------------------------------
    g = p.add_argument_group("performance")
    g.add_argument("--kernel", choices=["auto", "numpy", "simd", "fused", "fft"],
                   default="auto",
                   help="structure-factor/gradient kernel.  'auto' (default) "
                        "uses the AVX-512 SIMD kernel when this machine has the "
                        "CPU flags and a C compiler, else numpy.  All kernels "
                        "use the same ML coefficients, so they give the same "
                        "refinement trajectory (verified by the selftest)")
    g.add_argument("--chunk", type=int, default=48,
                   help="reflections per block for the direct-sum kernels")
    g.add_argument("--fft-spacing", type=float, default=None,
                   help="FFT grid spacing in A (default: auto = adaptive, "
                        "0.25 A for typical protein B-factors).  Smaller = more "
                        "accurate, slower")
    g.add_argument("--fft-radius", type=float, default=None,
                   help="FFT density cut-off radius in A (default: per-atom "
                        "adaptive, 2.6 sigma of the widest Gaussian, floored at "
                        "2.5 A and capped at 5 A)")

    # ---- deprecated aliases --------------------------------------------
    g = p.add_argument_group("deprecated aliases (use --kernel)")
    g.add_argument("--fused-kernel", action="store_true",
                   help="alias for --kernel fused")
    g.add_argument("--simd-kernel", action="store_true",
                   help="alias for --kernel simd")
    g.add_argument("--fft-kernel", action="store_true",
                   help="alias for --kernel fft")

    # ---- output --------------------------------------------------------
    g = p.add_argument_group("output")
    g.add_argument("--map-scale", choices=["absolute", "observed"],
                   default="absolute",
                   help="output MTZ scale: 'absolute' = REFMAC5 convention "
                        "(F/SIGF and map coefficients divided by k_total); "
                        "'observed' = legacy input scale")
    g.add_argument("--quiet", action="store_true",
                   help="only warnings and the final summary")
    g.add_argument("--no-json", action="store_true",
                   help="skip report.json")
    return p


def _print_data_summary(data, seconds):
    print("  %d reflections, %d atoms, %s, %.2f-%.2f A, load %.2f s"
          % (data.n_ref, data.n_atoms, data.spacegroup_name,
             data.d_min, data.d_max, seconds), flush=True)
    st = getattr(data, "data_stats", {}) or {}
    if st.get("n_duplicate_hkl"):
        print("  note: %d duplicate HKLs were merged" % st["n_duplicate_hkl"],
              flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        os.makedirs(args.out_dir, exist_ok=True)
    except OSError as exc:
        print("ERROR: cannot create --out-dir %s: %s" % (args.out_dir, exc),
              file=sys.stderr)
        return 2
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
        kernel=args.kernel,
        fused_kernel=args.fused_kernel, simd_kernel=args.simd_kernel,
        fft_kernel=args.fft_kernel, fft_spacing=args.fft_spacing,
        fft_radius=args.fft_radius,
        verbose=not args.quiet,
        out_pdb=os.path.join(args.out_dir, tag + "refined.pdb"),
        out_mtz=os.path.join(args.out_dir, tag + "refined.mtz"),
        out_log=os.path.join(args.out_dir, tag + "refine.log"),
        out_json=None if args.no_json else os.path.join(
            args.out_dir, tag + "report.json"),
    )

    print("Loading %s + %s ..." % (args.mtz, args.pdb), flush=True)
    t0 = time.time()
    try:
        free_value = int(args.free_value)
    except (TypeError, ValueError):
        free_value = "auto"
    try:
        data = dataio.load_data(
            args.mtz, args.pdb,
            fp_label=args.fp, sigfp_label=args.sigfp, free_label=args.free,
            free_value=free_value,
            phib_label=args.phib, fom_label=args.fom,
            hla_label=args.hla, hlb_label=args.hlb,
            hlc_label=args.hlc, hld_label=args.hld,
            d_min=args.dmin, d_max=args.dmax,
            keep_hydrogens=args.keep_hydrogens)
    except (OSError, ValueError, RuntimeError) as exc:
        print("ERROR: cannot load input data: %s" % exc, file=sys.stderr)
        return 2
    _print_data_summary(data, time.time() - t0)

    fi = getattr(data, "free_info", {}) or {}
    if fi.get("mode") == "absent":
        print("  WARNING: no FreeR_flag column -> all reflections are work set "
              "and Rfree is unavailable", flush=True)
    elif fi.get("suspicious"):
        print("  WARNING: free set = %.1f %% of the observed data (flag %s=%d); "
              "Rfree is not meaningful -- re-run with --free-value"
              % (100.0 * float(fi.get("free_fraction", 0.0)), data.free_label,
                 data.free_value), flush=True)
    if args.phase_prior and data.phib is None and data.hl is None:
        print("  WARNING: --phase-prior requested but no phase columns found; "
              "MLHL falls back to plain ML", flush=True)

    try:
        engine = refine.Refiner(data, o)
        result = engine.run()
    except (ValueError, RuntimeError) as exc:
        print("ERROR: refinement failed: %s" % exc, file=sys.stderr)
        return 3

    print("\nRESULT  R %.4f -> %.4f   Rfree %.4f -> %.4f   -LL %.1f -> %.1f   "
          "%.1f s" % (result["initial"]["r_work"], result["final"]["r_work"],
                      result["initial"]["r_free"], result["final"]["r_free"],
                      result["initial"]["negll"], result["final"]["negll"],
                      result["wall_seconds"]), flush=True)
    print("  kernel=%s   outputs: %s, %s%s"
          % (result.get("kernel", "?"), o.out_pdb, o.out_mtz,
             "" if not o.out_json else ", " + o.out_json), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
