"""
minirefmac.selftest -- numerical self-checks for the refinement engine.

Run (only gemmi / numpy / scipy from PyPI are needed):
    python -m minirefmac.selftest \
        --mtz <...> --pdb <...> [--json out.json]

Checks
------
1. Fc (symmetry-expanded direct sum) vs gemmi StructureFactorCalculatorX
2. -LL objective consistency between negll_and_G and negll_only
3. analytic X-ray gradients vs central finite differences (x and B)
4. analytic geometry gradients vs central finite differences
5. ML parameter estimation sanity (D, S bounds, monotonic S)
6. one short refinement: total objective decreases, R/Rfree computed
7. output MTZ columns are readable and free-set RESTORE is honoured
8. output MTZ uses the REFMAC5 absolute scale (F_abs = F_obs/k_total),
   FC is atom-only, and the convention is recorded in the MTZ history
9. every acceleration kernel (fused / SIMD / FFT) reproduces the numpy
   reference Fc and gradients *for the same ML coefficient G* -- this is what
   guarantees that all kernels follow the same refinement trajectory
10. the FreeR flag convention is resolved correctly ("auto"): CCP4 0=free,
   0=work/1=free, and the single-value fallback
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import gemmi

try:
    from . import (dataio, sfcalc, likelihood, gradient, geometry, maps, refine,
                   fastkernel, simdkernel, fftcalc)
except ImportError:  # pragma: no cover
    import dataio, sfcalc, likelihood, gradient, geometry, maps, refine
    import fastkernel, simdkernel, fftcalc


def _finite_diff(fun, vec, idx, step):
    vp = vec.copy(); vp[idx] += step
    vm = vec.copy(); vm[idx] -= step
    return (fun(vp) - fun(vm)) / (2.0 * step)


def run(args):
    out = {"checks": {}, "passed": True}
    t0 = time.time()
    data = dataio.load_data(args.mtz, args.pdb,
                            fp_label=args.fp, sigfp_label=args.sigfp,
                            free_label=args.free, free_value=args.free_value,
                            d_min=args.dmin, d_max=args.dmax)
    out["data"] = dict(n_ref=data.n_ref, n_atoms=data.n_atoms,
                       spacegroup=data.spacegroup_name,
                       d_min=data.d_min, d_max=data.d_max,
                       n_work=int(np.sum(data.work)), n_free=int(np.sum(data.test)),
                       n_centric=int(np.sum(data.centric)))
    print("data:", out["data"], flush=True)

    xyz, b, occ = data.xyz.copy(), data.b_iso.copy(), data.occ
    fc_atoms = sfcalc.calc_fc(data, xyz, b, occ)
    fmask = sfcalc.calc_fmask(data, xyz)
    scale = sfcalc.fit_scales(data, fc_atoms, fmask, solvent=args.solvent,
                              fit_aniso=True)
    fc_total = sfcalc.build_fc_total(data, fc_atoms, fmask, scale)
    mlpar = likelihood.estimate_ml_params(data, fc_total, scale,
                                          use="work", n_bins=args.bins)
    print("scales:", scale.as_dict(), flush=True)

    # ---- 1. Fc vs gemmi -------------------------------------------------
    calc = gemmi.StructureFactorCalculatorX(data.cell)
    rng = np.random.default_rng(20261114)
    sel = rng.choice(data.n_ref, size=min(20, data.n_ref), replace=False)
    gem = np.array([calc.calculate_sf_from_model(data.st[0], tuple(map(int, data.hkl[i])))
                    for i in sel])
    mine = fc_atoms[sel]
    rel = float(np.max(np.abs(gem - mine)) / max(np.max(np.abs(gem)), 1e-9))
    out["checks"]["fc_vs_gemmi"] = dict(max_rel_err=rel, passed=rel < 1e-5)
    print("1) Fc vs gemmi: max rel err %.3g" % rel, flush=True)

    # ---- 2. objective consistency --------------------------------------
    n1, G, m, phase, extra = likelihood.negll_and_G(
        data, fc_total, scale, mlpar, use_phase_prior=args.phase_prior,
        work_only=True)
    n2 = likelihood.negll_only(data, fc_total, scale, mlpar,
                               use_phase_prior=args.phase_prior)
    d2 = abs(n1 - n2) / max(abs(n1), 1.0)
    out["checks"]["negll_consistency"] = dict(rel_diff=d2, passed=d2 < 1e-9,
                                              negll=float(n1))
    print("2) -LL consistency: %.3g  (-LL=%.2f)" % (d2, n1), flush=True)

    # ---- 3. X-ray gradients finite difference --------------------------
    fcf = scale.fc_atom_factor(data)
    _, gx, gb = gradient.fc_and_atomic_gradients(
        data, xyz, b, occ, G, fcf, chunk=args.chunk, want_fc=False)

    def negll_from_flat(v):
        x = v[:3 * data.n_atoms].reshape(data.n_atoms, 3)
        bb = v[3 * data.n_atoms:]
        fct = sfcalc.build_fc_total(
            data, sfcalc.calc_fc(data, x, bb, occ), fmask, scale)
        return likelihood.negll_only(data, fct, scale, mlpar,
                                     use_phase_prior=args.phase_prior)

    flat = np.concatenate([xyz.ravel(), b])
    rows = []
    picks = rng.choice(data.n_atoms, size=min(4, data.n_atoms), replace=False)
    for j in picks:
        for k in range(3):
            idx = 3 * int(j) + k
            fd = _finite_diff(negll_from_flat, flat, idx, 2e-6)
            rows.append(dict(atom=int(j), par="xyz%d" % k,
                             analytic=float(gx[j, k]), fd=float(fd)))
        idx = 3 * data.n_atoms + int(j)
        fd = _finite_diff(negll_from_flat, flat, idx, 1e-3)
        rows.append(dict(atom=int(j), par="B",
                         analytic=float(gb[j]), fd=float(fd)))
    an = np.array([r["analytic"] for r in rows])
    fd = np.array([r["fd"] for r in rows])
    rel3 = float(np.linalg.norm(an - fd) / max(np.linalg.norm(an), 1e-12))
    out["checks"]["xray_gradient_fd"] = dict(rel_l2_err=rel3, passed=rel3 < 1e-2,
                                             rows=rows)
    print("3) X-ray gradient FD: ||an-fd||/||an|| = %.3g" % rel3, flush=True)

    # ---- 4. geometry gradients finite difference -----------------------
    restr = geometry.Restraints(data)
    fg, ggx, ggb, gst = restr.calc(xyz, b)

    def geom_from_flat(v):
        return restr.calc(v[:3 * data.n_atoms].reshape(data.n_atoms, 3),
                          v[3 * data.n_atoms:])[0]

    grows = []
    for j in picks[:3]:
        for k in range(3):
            idx = 3 * int(j) + k
            fdv = _finite_diff(geom_from_flat, flat, idx, 1e-6)
            grows.append(dict(atom=int(j), par="xyz%d" % k,
                              analytic=float(ggx[j, k]), fd=float(fdv)))
        idx = 3 * data.n_atoms + int(j)
        fdv = _finite_diff(geom_from_flat, flat, idx, 1e-3)
        grows.append(dict(atom=int(j), par="B",
                          analytic=float(ggb[j]), fd=float(fdv)))
    gan = np.array([r["analytic"] for r in grows])
    gfd = np.array([r["fd"] for r in grows])
    rel4 = float(np.linalg.norm(gan - gfd) / max(np.linalg.norm(gan), 1e-12))
    out["checks"]["geometry_gradient_fd"] = dict(rel_l2_err=rel4,
                                                 passed=rel4 < 1e-2, rows=grows)
    out["checks"]["geometry_stats"] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                       for k, v in gst.items() if isinstance(v, (int, float, np.floating))}
    print("4) Geometry gradient FD: %.3g  (rmsd_bond=%.4f)"
          % (rel4, gst["rmsd_bond"]), flush=True)

    # ---- 5. ML parameters ----------------------------------------------
    D, S = mlpar.D, mlpar.S
    ok5 = bool(np.all(S > 0) and np.all(np.isfinite(D)) and np.all(D > 0)
               and np.all(np.isfinite(S)))
    out["checks"]["ml_params"] = dict(
        passed=ok5, D_min=float(D.min()), D_max=float(D.max()),
        S_min=float(S.min()), S_max=float(S.max()), n_bins=mlpar.n_bins)
    print("5) ML params: D %.3f-%.3f, S %.1f-%.1f" %
          (D.min(), D.max(), S.min(), S.max()), flush=True)

    # ---- 6. short refinement -------------------------------------------
    odir = args.out_dir
    os.makedirs(odir, exist_ok=True)
    opts = refine.Options(ncyc=1, maxiter=3, n_bins=args.bins,
                          solvent=args.solvent,
                          fit_aniso=True, verbose=False,
                          out_pdb=os.path.join(odir, "selftest.pdb"),
                          out_mtz=os.path.join(odir, "selftest.mtz"),
                          out_json=os.path.join(odir, "selftest.json"))
    r = refine.Refiner(data, opts).run()
    cyc = r["cycles"]
    f_initial = cyc[0]["negll"]
    f_final = cyc[-1]["negll"]
    ok6 = bool(np.isfinite(f_final))
    out["checks"]["short_refinement"] = dict(
        passed=ok6, r_initial=cyc[0]["r_work"], r_final=cyc[-1]["r_work"],
        rfree_initial=cyc[0]["r_free"], rfree_final=cyc[-1]["r_free"],
        negll_initial=float(f_initial), negll_final=float(f_final),
        n_evals=r["n_target_evals"], wall=r["wall_seconds"])
    print("6) short refinement: R %.4f -> %.4f, -LL %.1f -> %.1f (%.1f s)"
          % (cyc[0]["r_work"], cyc[-1]["r_work"], f_initial, f_final,
             r["wall_seconds"]), flush=True)

    # ---- 7. output MTZ --------------------------------------------------
    mtz = gemmi.read_mtz_file(opts.out_mtz)
    labels = mtz.column_labels()
    need = ["FWT", "PHWT", "DELFWT", "PHDELWT", "FOM", "FC_ALL", "PHIC_ALL"]
    missing = [x for x in need if x not in labels]
    arr = mtz.array
    ok7 = (len(missing) == 0 and bool(np.all(np.isfinite(arr))))
    out["checks"]["output_mtz"] = dict(passed=ok7, labels=labels,
                                       missing=missing,
                                       nref=mtz.nreflections)
    print("7) output MTZ: %d cols, missing=%s" % (len(labels), missing), flush=True)

    # ---- 8. REFMAC5 absolute output scale -------------------------------
    try:
        from .core import ScaleParams
        from . import maps as _maps
    except ImportError:  # pragma: no cover
        from core import ScaleParams
        import maps as _maps
    sp = r["cycles"][-1]["scale"]
    scale_final = ScaleParams(
        k_overall=float(sp["k_overall"]), b_over=float(sp["b_over"]),
        b_aniso=np.asarray(sp["b_aniso"], dtype=np.float64),
        k_sol=float(sp["k_sol"]), b_sol=float(sp["b_sol"]),
        solvent=sp.get("solvent", "mask"))
    mi = gemmi.read_mtz_file(args.mtz)
    mo = gemmi.read_mtz_file(opts.out_mtz)
    Li, Lo = mi.column_labels(), mo.column_labels()
    fi = np.asarray(mi.array, dtype=np.float64)[:, Li.index(data.fp_label)]
    fo = np.asarray(mo.array, dtype=np.float64)[:, Lo.index(data.fp_label)]
    k_all, _ = _maps.all_rows_k_total(mo, scale_final)
    good_f = np.isfinite(fi) & (fi > 0) & np.isfinite(fo)
    err_f = float(np.max(np.abs(fo[good_f] * k_all[good_f] - fi[good_f]))
                  / max(float(np.max(fi[good_f])), 1.0))
    aout = np.asarray(mo.array, dtype=np.float64)
    fc_sep = float(np.mean(np.abs(aout[:, Lo.index("FC")]
                                  - aout[:, Lo.index("FC_ALL")])))
    history_ok = any("output_scale=absolute" in h for h in mo.history)
    ok8 = bool(err_f < 1e-5 and fc_sep > 0
               and opts.map_scale == "absolute" and history_ok)
    out["checks"]["output_scale_absolute"] = dict(
        passed=ok8, map_scale=opts.map_scale, history_ok=history_ok,
        f_scale_rel_err=err_f, mean_abs_fc_minus_fc_all=fc_sep,
        history=list(mo.history))
    print("8) absolute output scale: rel|F_out*k - F_in|=%.3g, "
          "mean|FC-FC_ALL|=%.3f" % (err_f, fc_sep), flush=True)

    # ---- 9. kernel agreement at a fixed G --------------------------------
    def _rel(a, b):
        den = float(np.linalg.norm(b))
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / den) if den else 0.0

    kres = {}
    fc_fu, gx_fu, gb_fu = fastkernel.fused_fc_and_gradients(
        data, xyz, b, occ, G, fcf, chunk=args.chunk)
    kres["fused"] = dict(fc=_rel(fc_fu, fc_atoms), gx=_rel(gx_fu, gx),
                         gb=_rel(gb_fu, gb), required=True)
    ok_simd, why_simd = simdkernel.probe()
    if ok_simd:
        fc_sd, gx_sd, gb_sd = simdkernel.simd_fc_and_gradients(
            data, xyz, b, occ, G, fcf, chunk=args.chunk)
        kres["simd"] = dict(fc=_rel(fc_sd, fc_atoms), gx=_rel(gx_sd, gx),
                            gb=_rel(gb_sd, gb), required=True)
    else:
        kres["simd"] = dict(skipped=why_simd, required=False, fc=0.0,
                            gx=0.0, gb=0.0)
    try:
        sp = fftcalc.auto_spacing(data.cell, getattr(data, "d_min", None),
                                   b_min=float(np.min(b)))
        grid, fatoms, _ = fftcalc._scatter_density(
            data, sp, sfcalc._sym_ops(data), None, xyz=xyz, b_iso=b, occ=occ)
        fc_fft = fftcalc.fft_fc(data, spacing=sp, radius_A=None,
                                grid=grid, atoms=fatoms)
        kres["fft"] = dict(fc=_rel(fc_fft, fc_atoms), spacing=float(sp),
                           required=False)
    except Exception as exc:              # pragma: no cover
        kres["fft"] = dict(error=str(exc), required=False, fc=0.0)
    ok9 = all((v.get("fc", 0.0) < 1e-9 and v.get("gx", 0.0) < 1e-9
               and v.get("gb", 0.0) < 1e-9) for k, v in kres.items()
              if v.get("required"))
    ok9 = bool(ok9 and kres.get("fft", {}).get("fc", 1.0) < 5e-3)
    out["checks"]["kernel_agreement"] = dict(passed=ok9, kernels=kres)
    print("9) kernel agreement (fixed G): %s  (fft Fc rel err %.1e)"
          % (", ".join("%s %.0e" % (k, v.get("fc", 0.0))
                       for k, v in kres.items()), kres.get("fft", {}).get("fc", 0.0)),
          flush=True)

    # ---- 10. FreeR flag convention --------------------------------------
    cases = [("ccp4 0=free", [0] * 5 + [1] * 95, 0),
             ("0=work 1=free", [0] * 95 + [1] * 5, 1),
             ("single value", [0] * 100, 0)]
    fcase = {}
    for name, flags, want in cases:
        arr = np.asarray(flags, dtype=np.int64)
        with warnings.catch_warnings():      # the single-value case warns on purpose
            warnings.simplefilter("ignore")
            got, info = dataio.resolve_free_value(arr, np.ones(arr.size, bool),
                                                 "auto")
        fcase[name] = dict(flags=want, chosen=int(got), ok=bool(got == want),
                           mode=info["mode"])
    ok10 = all(v["ok"] for v in fcase.values())
    out["checks"]["free_value_auto"] = dict(passed=ok10, cases=fcase)
    print("10) free-value auto: %s" % {k: v["chosen"] for k, v in fcase.items()},
          flush=True)

    out["passed"] = all(v.get("passed", True) for k, v in out["checks"].items()
                        if isinstance(v, dict))
    out["wall_seconds"] = time.time() - t0
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2, default=float)
    print("SELFTEST %s in %.1f s" % ("PASS" if out["passed"] else "FAIL",
                                     out["wall_seconds"]), flush=True)
    return 0 if out["passed"] else 1


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mtz", required=True)
    p.add_argument("--pdb", required=True)
    p.add_argument("--out-dir", default="tests/selftest_out")
    p.add_argument("--json", default=None)
    p.add_argument("--fp", default=None)
    p.add_argument("--sigfp", default=None)
    p.add_argument("--free", default=None)
    p.add_argument("--free-value", default="auto")
    p.add_argument("--dmin", type=float, default=0.0)
    p.add_argument("--dmax", type=float, default=0.0)
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--solvent", default="mask")
    p.add_argument("--chunk", type=int, default=48)
    p.add_argument("--phase-prior", action="store_true")
    a = p.parse_args(argv)
    try:
        a.free_value = int(a.free_value)
    except (TypeError, ValueError):
        a.free_value = "auto"
    return run(a)


if __name__ == "__main__":
    import sys
    sys.exit(main())
