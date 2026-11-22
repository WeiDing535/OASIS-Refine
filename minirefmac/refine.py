"""
minirefmac.refine -- the refinement driver.

Outer loop (one "cycle", REFMAC's NCYC):
    1. recompute the solvent mask, overall/anisotropic scale, bulk solvent
    2. re-estimate the ML parameters D and S in resolution bins (sigmaA step)
    3. build the ML gradient coefficient G on the WORKING set only
    4. minimise  f = w_xray * (-LL) + f_geometry  with L-BFGS-B over
       fractional coordinates and isotropic B factors
    5. report Rwork / Rfree / -LL / geometry and adapt w_xray

The X-ray gradient is recomputed exactly at every L-BFGS evaluation: the ML
coefficients G depend on the current Fc, so this is a genuine maximum
likelihood minimisation (not a frozen-map linearisation).  Each evaluation
costs one structure-factor pass plus one gradient pass.

Author: hand-written re-implementation, algorithm after
Murshudov et al. (1997, 2011), Pannu & Read (1996), Pannu et al. (1998).
"""

import copy
import json
import math
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import minimize

try:
    from . import __version__
    from . import dataio, sfcalc, likelihood, gradient, geometry, maps, fastkernel
    from . import occupancy, simdkernel
    from .core import ScaleParams, CycleStats
except ImportError:  # pragma: no cover - allow flat-module execution
    import dataio, sfcalc, likelihood, gradient, geometry, maps, fastkernel
    from core import ScaleParams, CycleStats
    import occupancy, simdkernel
    __version__ = "0.2.0"


@dataclass
class Options:
    ncyc: int = 5
    maxiter: int = 10
    n_bins: int = 20
    solvent: str = "mask"           # mask | babinet | none
    fit_aniso: bool = True
    use_phase_prior: bool = False  # MLHL
    use_monlib: bool = True
    restraint_cifs: tuple = ()
    refine_occupancies: bool = False
    freeze_overlapping_occupancies: bool = False
    occupancy_prior_sigma: float = None
    weight: float = None            # explicit w_xray, overrides auto balance
    # Calibrated on the real SAD cases: 1.0 (pure gradient-norm balance)
    # over-fits (Rfree rises, geometry loosens).  0.05 reproduces REFMAC's
    # strong geometry restraints and lowers Rfree.
    weight_scale: float = 0.05
    auto_weight: bool = True
    # How w_xray is updated between cycles.
    #   "auto"      : bounded search on Rfree, starting from the gradient
    #                 balance (see _update_weight_auto and
    #                 METHODS_WEIGHT_AND_OVERFITTING.md).
    #   "heuristic" : the pre-0.3 ad-hoc feedback (rmsd bands + Rfree drift).
    #   "fixed"     : keep the cycle-0 gradient-balance value.
    # A "target rmsd" controller was implemented and measured (0.3): the bond
    # rmsd of this engine is nearly insensitive to w_xray (0.0040 -> 0.0069 for
    # a 27x weight change), so such a controller cannot hit a REFMAC-like 0.02
    # target and only drives w to its clip.  It was removed rather than shipped.
    weight_mode: str = "auto"
    weight_step: float = 1.5        # multiplicative step of the Rfree search
    weight_min_scale: float = 0.5   # lower clip, as a multiple of w_balance
    weight_max_scale: float = 3.0   # upper clip, as a multiple of w_balance
    # Early stopping.  "paired" (default) stops when the current model is
    # significantly worse than the best-Rfree model (paired test, t >= 2);
    # an integer N stops after N cycles without a new best; "off" disables.
    early_stop: str = "paired"
    # Write the best-Rfree model rather than the last cycle's model.  With an
    # honest free set this is the model you want; the last cycle is often
    # slightly over-fitted.
    keep_best: bool = True
    w_bond: float = 1.0
    w_angle: float = 1.0
    w_b: float = 1.0
    # non-bonded VDW repulsion (one-sided harmonic, ASU + symmetry mates)
    w_vdw: float = 1.0
    sym_vdw: bool = True
    max_shift_a: float = 0.5        # max coordinate shift in Angstrom
    b_min: float = 2.0
    b_max: float = 200.0
    chunk: int = 48
    # Structure-factor / gradient kernel.  "auto" (default) picks the fastest
    # exact kernel available on this machine: the AVX-512 SIMD kernel when the
    # CPU supports it and a C compiler (or a prebuilt _simd.so) is present,
    # otherwise the pure-numpy reference path.  Every kernel is fed the SAME
    # ML coefficient G, so they all trace the identical refinement trajectory.
    #   numpy : reference implementation (gradient.py)
    #   simd  : AVX-512 C kernel, ~8x faster per pass (simdkernel.py/_simd.c)
    #   fused : numpy kernel fusion of Fc+gradients (fastkernel.py)
    kernel: str = "auto"
    # Deprecated aliases kept for backwards compatibility with old scripts;
    # they only take effect when kernel == "auto".
    fused_kernel: bool = False
    simd_kernel: bool = False
    simd_threads: int = 1
    # REFMAC5 writes F/SIGF and all map coefficients on the ABSOLUTE scale
    # (F_abs = F_obs / k_total).  "observed" keeps the legacy input scale.
    map_scale: str = "absolute"
    verbose: bool = True
    out_pdb: str = None
    out_mtz: str = None
    out_log: str = None
    out_json: str = None
    b_scale: float = 5.0            # Å² per unit of the L-BFGS-B B variable


class Refiner:
    """Maximum-likelihood restrained refinement engine."""

    def __init__(self, data, opts=None):
        self.data = data
        self.opts = opts or Options()
        if not math.isfinite(self.opts.b_scale) or self.opts.b_scale <= 0:
            raise ValueError("B-factor optimizer scale must be positive and finite")
        self.kernel, self.kernel_reason = self._resolve_kernel()
        self.log_lines = []
        self.cycles = []
        self.restr = None
        self.scale = None
        self.mlpar = None
        self.fmask = None
        self.occ_groups = []
        self.occ_skipped = 0
        self.occ_updates = []
        self.w_xray = 1.0
        self.n_eval = 0
        self.last_fc_total = None
        self.last_m = None
        self.last_phase = None
        # frozen cycle-0 ML parameters / scale, used for a comparable -LL
        self.mlpar0 = None
        self.scale0 = None
        self.best_rfree = float("inf")
        self.best_rfree_cycle = None
        self.best_xyz = None
        self.best_b = None
        self.best_occ = None
        self.best_fc_total = None
        self.best_scale = None
        self.early_stopped = False
        # state of the Rfree weight search
        self._w_ref_rfree = None
        self._prev_fc_total = None
        self.w_balance = None

        # Parameter scaling uses ~0.1 A for coordinates and b_scale A^2 for B factors.
        a_mean = float(np.mean(data.cell_parameters[:3])) if data.cell_parameters else 50.0
        self.sx = 0.1 / max(a_mean, 1.0)
        self.sb = self.opts.b_scale

    # ------------------------------------------------------------------
    def log(self, msg):
        self.log_lines.append(msg)
        if self.opts.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------
    def _resolve_kernel(self):
        """Decide which structure-factor/gradient kernel to use.

        Returns ``(name, reason)``.  An explicit ``Options.kernel`` wins; the
        deprecated boolean flags act as explicit choices too; ``"auto"``
        probes the machine and never raises (a missing compiler or a CPU
        without AVX-512 simply means the numpy reference path).
        """
        o = self.opts
        k = str(getattr(o, "kernel", "auto") or "auto").strip().lower()
        valid = ("auto", "numpy", "fused", "simd")
        if k not in valid:
            raise ValueError("Options.kernel must be one of %s (got %r)"
                             % (", ".join(valid), o.kernel))
        legacy = [name for name, flag in (("fused", o.fused_kernel),
                                          ("simd", o.simd_kernel)) if flag]
        if k == "auto" and legacy:
            k = legacy[0]
        if k == "auto":
            ok, why = simdkernel.probe()
            if ok and simdkernel.ensure_simd_lib():
                return "simd", why
            return "numpy", "SIMD build/load failed" if ok else why
        if k == "simd":
            ok, why = simdkernel.probe()
            if ok and not simdkernel.ensure_simd_lib():
                ok, why = False, "SIMD build/load failed"
            if not ok:
                msg = ("SIMD kernel requested but unavailable (%s); "
                       "falling back to numpy" % why)
                if os.environ.get("MINIREFMAC_STRICT_KERNEL"):
                    raise RuntimeError(msg)
                return "numpy", msg
            return "simd", why
        return k, "explicit --kernel %s" % k

    # ------------------------------------------------------------------
    def _calc_fc_atoms(self, xyz, b_iso, occ):
        """Atom-only structure factors, computed with the selected kernel."""
        d, o = self.data, self.opts
        if self.kernel == "simd":
            fc, _, _ = simdkernel.simd_fc_and_gradients(
                d, xyz, b_iso, occ, None, None, chunk=o.chunk, want_grad=False,
                threads=o.simd_threads)
            return fc
        return sfcalc.calc_fc(d, xyz, b_iso, occ)

    # ------------------------------------------------------------------
    def _kernel_eval(self, xyz, b_iso, occ, fcf):
        """One X-ray evaluation: Fc -> ML coefficients G -> exact gradients.

        All kernels receive the *current* G, i.e. the gradient returned here is
        the exact derivative of the ML target at ``(xyz, b_iso)``.  That is
        what makes every kernel trace the same trajectory (see
        ``tests/test_kernels.py``-style check 9 in selftest.py).
        """
        d, o = self.data, self.opts
        kernel = self.kernel
        fc_atoms = self._calc_fc_atoms(xyz, b_iso, occ)
        fc_total = sfcalc.build_fc_total(d, fc_atoms, self.fmask, self.scale)
        negll, G, mm, phase, extra = likelihood.negll_and_G(
            d, fc_total, self.scale, self.mlpar,
            use_phase_prior=o.use_phase_prior, work_only=True)
        if kernel == "simd":
            _, gx, gb = simdkernel.simd_fc_and_gradients(
                d, xyz, b_iso, occ, G, fcf, chunk=o.chunk, want_fc=False,
                threads=o.simd_threads)
        elif kernel == "fused":
            _, gx, gb = fastkernel.fused_fc_and_gradients(
                d, xyz, b_iso, occ, G, fcf, chunk=o.chunk)
        else:
            _, gx, gb = gradient.fc_and_atomic_gradients(
                d, xyz, b_iso, occ, G, fcf, chunk=o.chunk, want_fc=False)
        return fc_atoms, gx, gb, fc_total, negll, G, mm, phase, extra

    # ------------------------------------------------------------------
    def _prepare(self):
        o = self.opts
        self.restr = geometry.Restraints(
            self.data,
            w_bond=o.w_bond, w_angle=o.w_angle, w_b=o.w_b,
            w_vdw=o.w_vdw, sym_vdw=o.sym_vdw,
            use_monlib=o.use_monlib, restraint_cifs=o.restraint_cifs)
        self.log("Geometry restraints: %d bonds (%d disulfides, %d explicit), %d angles (%d cross-peptide), "
                 "%d planar groups (%d peptide), %d torsions, %d chirs, "
                 "%d VDW pairs (%d symmetry) (%s)"
                 % (self.restr.n_bonds, self.restr.info["n_disulfide"],
                    self.restr.info["n_disulfide_explicit"], self.restr.n_angles,
                    self.restr.info["n_cross_angle"],
                    self.restr.info["n_plane_groups"],
                    self.restr.info["n_link_plane"],
                    self.restr.info["n_torsion"],
                    self.restr.info["n_chir"],
                    self.restr.info["n_vdw"], self.restr.info["n_vdw_sym"],
                    ("jelly-body fallback (dictionary disabled)"
                     if not o.use_monlib else "bundled monomer dictionary")))

        if o.restraint_cifs:
            self.log("  external component restraints applied: %s"
                     % (", ".join(self.restr.info["external_components"]) or "none matched"))

    # ------------------------------------------------------------------
    def _outer_update(self, xyz, b_iso, first=False):
        """Refresh mask, scales and ML parameters with the current model."""
        d, o = self.data, self.opts
        fc_atoms = self._calc_fc_atoms(xyz, b_iso, d.occ)
        try:
            if o.solvent == "mask":
                self.fmask = (sfcalc.calc_fmask(d, xyz, occ=d.occ) if o.refine_occupancies
                              else sfcalc.calc_fmask(d, xyz))
            else:
                self.fmask = None
        except Exception as exc:      # pragma: no cover
            self.log("  ! solvent mask failed (%s); falling back to babinet" % exc)
            o.solvent = "babinet"
            self.fmask = None
        self.scale = sfcalc.fit_scales(
            d, fc_atoms, self.fmask, solvent=o.solvent,
            fit_aniso=o.fit_aniso,
            params0=None if first else self.scale)
        fc_total = sfcalc.build_fc_total(d, fc_atoms, self.fmask, self.scale)
        self.mlpar = likelihood.estimate_ml_params(
            d, fc_total, self.scale, use="work", n_bins=o.n_bins,
            params0=None if first else self.mlpar)
        if first:
            # frozen reference for a cross-cycle comparable -LL
            self.mlpar0 = copy.deepcopy(self.mlpar)
            self.scale0 = copy.deepcopy(self.scale)
        self._prev_fc_total = self.last_fc_total
        self.last_fc_total = fc_total
        return fc_total

    # ------------------------------------------------------------------
    def _balance_weight(self, xyz, b_iso, fc_total):
        """Set w_xray so that X-ray and geometry gradient norms are comparable."""
        o = self.opts
        if o.weight is not None:
            self.w_xray = float(o.weight)
            self.w_balance = self.w_xray
            self.w_base = self.w_xray
            return
        d = self.data
        negll, G, m, phase, extra = likelihood.negll_and_G(
            d, fc_total, self.scale, self.mlpar,
            use_phase_prior=o.use_phase_prior, work_only=True)
        fcf = self.scale.fc_atom_factor(d)
        if self.kernel == "simd":
            _, gx, gb = simdkernel.simd_fc_and_gradients(
                d, xyz, b_iso, d.occ, G, fcf, chunk=o.chunk, want_fc=False,
                threads=o.simd_threads)
        elif self.kernel == "fused":
            _, gx, gb = fastkernel.fused_fc_and_gradients(
                d, xyz, b_iso, d.occ, G, fcf, chunk=o.chunk)
        else:
            _, gx, gb = gradient.fc_and_atomic_gradients(
                d, xyz, b_iso, d.occ, G, fcf, chunk=o.chunk, want_fc=False)
        fg, ggx, ggb, stats = self.restr.calc(xyz, b_iso)
        n_x = math.sqrt(float(np.sum(gx ** 2) + np.sum(gb ** 2)))
        n_g = math.sqrt(float(np.sum(ggx ** 2) + np.sum(ggb ** 2)))
        self.w_xray = o.weight_scale * n_g / max(n_x, 1e-12)
        self.w_base = self.w_xray
        self.w_balance = self.w_xray
        self.log("  weight balance: |g_xray|=%.4g |g_geom|=%.4g -> w_xray=%.6g"
                 % (n_x, n_g, self.w_xray))

    # ------------------------------------------------------------------
    def _update_weight_auto(self, st):
        """Bounded search for w_xray on Rfree, starting from the gradient balance.

        Why not a geometry target: measured on 7enm (0.3), the achieved bond
        rmsd barely responds to the weight (0.0040 A at w=0.3, 0.0069 A at
        w=8), so a controller aiming at a REFMAC-like rmsd target cannot work in
        this engine -- it just saturates at its clip.

        Why the search is PAIRED: Rfree on a 5 % free set has a standard error
        of 0.01-0.02 (0.018 on 7enm, 487 reflections), i.e. several times the
        cycle-to-cycle changes one is tempted to act on.  The per-reflection
        residuals of two successive models are strongly correlated, so the
        paired difference dR = sum(|d_i^new| - |d_i^old|)/sum(Fo) has a standard
        error several times smaller (0.003 on the same data).  We therefore step
        the weight up only when the paired improvement is significant at 2 s.e.,
        and hold otherwise.

        The scan on 7enm (5 cycles each, fixed w) shows R falling monotonically
        with w (0.395 at w=0.3 -> 0.359 at w=8) while Rfree has a broad minimum
        around 2-4x the gradient balance, which is where this search lands.
        """
        o = self.opts
        base = float(getattr(self, "w_balance", self.w_xray) or self.w_xray)
        lo = base * float(o.weight_min_scale)
        hi = base * float(o.weight_max_scale)
        if len(self.cycles) < 2:
            return
        prev = self.cycles[-2]
        improved, why = False, "no paired test available"
        try:
            ref_sc = ScaleParams(**prev.scale)
            if self._prev_fc_total is not None and self.last_fc_total is not None:
                stats = likelihood.rfree_stats(
                    self.data, self.last_fc_total, self.scale,
                    ref_fc=self._prev_fc_total, ref_scale=ref_sc)
                t = stats.get("t", float("nan"))
                if np.isfinite(t):
                    improved = bool(t <= -2.0)
                    why = "paired dRfree=%+.4f +/- %.4f (t=%+.1f)" % (
                        stats["delta_vs_ref"], stats["se_delta"], t)
        except Exception as exc:            # pragma: no cover
            why = "paired test failed (%s)" % exc
        if improved:
            new = min(self.w_xray * float(o.weight_step), hi)
        else:
            new = self.w_xray              # hold
        new = float(np.clip(new, lo, hi))
        self.log("  weight: %s -> w_xray %.4g -> %.4g (band %.4g..%.4g)"
                 % (why, self.w_xray, new, lo, hi))
        self.w_xray = new

    def _update_weight_heuristic(self, st):
        """Legacy ad-hoc feedback kept for comparison with pre-0.3 results."""
        o = self.opts
        rb = st.rmsd_bond
        prev = self.cycles[-2].r_free if len(self.cycles) >= 2 else st.r_free
        drf = st.r_free - prev
        if np.isfinite(rb):
            if rb > 0.025:                       # geometry too loose
                self.w_xray *= 0.6
            elif rb < 0.015 and drf <= 0.001:    # room for more data
                self.w_xray *= 1.2
        if np.isfinite(drf):
            if drf > 0.010:                      # clear over-fitting
                self.w_xray *= 0.35
            elif drf > 0.003:
                self.w_xray *= 0.6
        base = getattr(self, "w_base", self.w_xray)
        self.w_xray = float(np.clip(self.w_xray, base / 20.0, base * 5.0))

    # ------------------------------------------------------------------
    def _make_target(self, xyz0, b0):
        """Return (fun, bounds) for scipy L-BFGS-B on the scaled variables."""
        d, o = self.data, self.opts
        m = d.n_atoms
        scale, mlpar = self.scale, self.mlpar
        fmask = self.fmask
        fcf = scale.fc_atom_factor(d)
        occ = d.occ
        sx, sb = self.sx, self.sb
        w = self.w_xray
        prior = o.use_phase_prior

        # sx*a_mean = 0.1 A, so v_limit translates max_shift_a into v units
        v_limit = o.max_shift_a / (sx * float(np.mean(d.cell_parameters[:3])))
        bounds = [(-v_limit, v_limit)] * (3 * m)
        bounds += [((o.b_min - b0[j]) / sb, (o.b_max - b0[j]) / sb) for j in range(m)]

        def fun(v):
            self.n_eval += 1
            xyz = xyz0 + v[:3 * m].reshape(m, 3) * sx
            b_iso = b0 + v[3 * m:] * sb

            # Fc -> ML coefficients G -> exact gradients.  Every kernel is fed
            # the SAME G (the one that belongs to the current coordinates), so
            # all kernels follow an identical refinement trajectory; see
            # _kernel_eval().  (Before 2026-11 the fused/SIMD paths reused a
            # one-step-stale G, which made their gradients inexact and their
            # R/Rfree drift by ~4e-4 from the numpy reference.)
            (fc_atoms, gx, gb, fc_total, negll, G, mm, phase, extra) = \
                self._kernel_eval(xyz, b_iso, occ, fcf)
            fg, ggx, ggb, stats = self.restr.calc(xyz, b_iso)

            f = w * negll + fg
            gx_tot = w * gx + ggx
            gb_tot = w * gb + ggb
            grad = np.concatenate([gx_tot.ravel() * sx, gb_tot * sb])
            self._last_stats = stats
            self._last_extra = extra
            return float(f), np.asarray(grad, dtype=np.float64)

        return fun, bounds

    # ------------------------------------------------------------------
    def _report(self, cyc, xyz, b_iso, fc_total, dt, rms_shift, rms_bshift):
        d, o = self.data, self.opts
        r_w, r_f, r_all = likelihood.r_factors(d, fc_total, self.scale)
        negll_free = likelihood.negll_only(d, fc_total, self.scale, self.mlpar,
                                           use_phase_prior=o.use_phase_prior,
                                           set_mask=d.test)
        negll, G, m, phase, extra = likelihood.negll_and_G(
            d, fc_total, self.scale, self.mlpar,
            use_phase_prior=o.use_phase_prior, work_only=False)
        self.last_m, self.last_phase = m, phase
        fg, _, _, stats = self.restr.calc(xyz, b_iso)
        # comparable -LL: freeze the cycle-0 sigmaA params AND overall scale
        negll_fixed = likelihood.negll_only(
            d, fc_total, self.scale0 or self.scale, self.mlpar0 or self.mlpar,
            use_phase_prior=o.use_phase_prior)

        st = CycleStats(cycle=cyc, r_work=r_w, r_free=r_f, negll=negll,
                        negll_fixed=negll_fixed,
                        negll_free=negll_free, geom=fg,
                        rmsd_bond=stats.get("rmsd_bond", float("nan")),
                        rmsd_angle=stats["rmsd_angle"],
                        rmsd_angle_deg=stats["rmsd_angle_deg"],
                        n_angle_outlier_4sigma=stats["n_angle_outlier_4sigma"],
                        rmsz_bond=stats.get("rmsz_bond", float("nan")),
                        w_xray=self.w_xray,
                        mean_fom=float(np.nanmean(m[d.work])) if np.any(d.work) else float("nan"),
                        rms_shift=rms_shift, rms_bshift=rms_bshift,
                        ll_time=dt, scale=self.scale.as_dict())
        self.cycles.append(st)
        if np.isfinite(r_f) and r_f < self.best_rfree:
            self.best_rfree = float(r_f)
            self.best_rfree_cycle = cyc
            self.best_xyz = xyz.copy()
            self.best_b = b_iso.copy()
            if o.refine_occupancies:
                self.best_occ = d.occ.copy()
            self.best_fc_total = fc_total.copy()
            self.best_scale = ScaleParams(**self.scale.as_dict())
        self.log("  cycle %2d  R=%7.4f  Rfree=%7.4f  -LL=%12.2f  -LLfix=%12.2f"
                 "  FOM=%.3f  rmsBOND=%.4f  rmsANGL=%.2fdeg  w=%.3g  %.1fs"
                 % (cyc, r_w, r_f, negll, negll_fixed, st.mean_fom,
                    st.rmsd_bond, st.rmsd_angle_deg, self.w_xray, dt))
        return st

    # ------------------------------------------------------------------
    def run(self):
        d, o = self.data, self.opts
        t_start = time.time()
        self.log("=" * 78)
        self.log("minirefmac -- restrained maximum-likelihood refinement")
        self.log("  reflections %d (work %d / free %d), atoms %d, %s, d=%.2f-%.2f A"
                 % (d.n_ref, int(np.sum(d.work)), int(np.sum(d.test)), d.n_atoms,
                    d.spacegroup_name, d.d_min, d.d_max))
        self.log("  solvent=%s  ML params=%d bins  MLHL=%s  ncyc=%d maxiter=%d"
                 % (o.solvent, o.n_bins, o.use_phase_prior, o.ncyc, o.maxiter))
        self.log("  kernel=%s  (%s)" % (self.kernel, self.kernel_reason))
        if self.kernel == "simd":
            self.log("  SIMD workers requested: %d" % o.simd_threads)
        fi = getattr(d, "free_info", {}) or {}
        if fi.get("mode") == "absent":
            self.log("  free set: NONE (no FreeR_flag column) -> Rfree unavailable")
        else:
            self.log("  free set: %s=%d -> %d of %d observed reflections (%.1f %%)"
                     % (d.free_label, d.free_value, int(np.sum(d.test)),
                        int(np.sum(d.observed)),
                        100.0 * float(fi.get("free_fraction", 0.0))))
        st = getattr(d, "data_stats", {}) or {}
        if st.get("n_missing_f"):
            self.log("  note: %d reflections with F<=0 / missing are excluded "
                     "from the target" % st["n_missing_f"])
        if st.get("n_bad_sigma"):
            self.log("  note: %d reflections have a missing/invalid sigma "
                     "(median substituted)" % st["n_bad_sigma"])
        if st.get("all_altlocs"):
            self.log("  alternate conformers: %d atoms selected, occupancies %s"
                     % (st["n_selected_altloc_atoms"],
                        "refined in eligible groups" if o.refine_occupancies else "fixed"))
        if st.get("n_altloc_atoms"):
            self.log("  note: %d atoms with altloc other than blank/A were skipped"
                     % st["n_altloc_atoms"])
        if st.get("n_duplicate_hkl"):
            self.log("  WARNING: %d duplicate HKL rows in the input MTZ"
                     % st["n_duplicate_hkl"])
        self.log("=" * 78)

        if (o.freeze_overlapping_occupancies or o.occupancy_prior_sigma is not None) and not o.refine_occupancies:
            raise ValueError("occupancy diagnostics controls require --refine-occupancies")
        if o.occupancy_prior_sigma is not None and (not np.isfinite(o.occupancy_prior_sigma)
                                                     or o.occupancy_prior_sigma <= 0):
            raise ValueError("occupancy prior sigma must be positive and finite")
        active_groups = []
        if o.refine_occupancies:
            if not getattr(d, "_all_altlocs", False):
                raise ValueError("occupancy refinement requires data loaded with all_altlocs=True")
            self.occ_groups, self.occ_skipped = occupancy.find_groups(d)
            self.log("  occupancy refinement: %d paired groups, %d alternate residues skipped"
                     % (len(self.occ_groups), self.occ_skipped))
            occ_input = d.occ.copy()

        self._prepare()
        xyz = d.xyz.copy()
        b_iso = d.b_iso.copy()
        xyz_start, b_start = xyz.copy(), b_iso.copy()
        if o.refine_occupancies:
            component_overrides = self.restr.info["external_components"]
            equiv_input = occupancy.diagnose_equivalence(
                d, self.occ_groups, xyz, component_overrides)
            for index, diagnostic in equiv_input.items():
                if diagnostic["near_overlap"]:
                    row = self.occ_groups[index].first[0]
                    self.log("  alternate overlap %s %d %s: %.3f A (%s)"
                             % (d.chain_ids[row], d.res_seq[row], d.res_names[row],
                                diagnostic["best_rmsd_A"], diagnostic["best_matching"]))

            frozen_groups = {i for i, diagnostic in equiv_input.items()
                             if o.freeze_overlapping_occupancies and diagnostic["near_overlap"]}
            active_groups = [g for i, g in enumerate(self.occ_groups) if i not in frozen_groups]
            prior_center = np.array([occ_input[g.first[0]] for g in active_groups])
            if frozen_groups:
                self.log("  overlapping occupancy groups kept at input: %d" % len(frozen_groups))

        fc_total = self._outer_update(xyz, b_iso, first=True)
        self._balance_weight(xyz, b_iso, fc_total)
        r0 = self._report(0, xyz, b_iso, fc_total, time.time() - t_start, 0.0, 0.0)

        for cyc in range(1, o.ncyc + 1):
            t0 = time.time()
            xyz0, b0 = xyz.copy(), b_iso.copy()
            fun, bounds = self._make_target(xyz0, b0)
            v0 = np.zeros(3 * d.n_atoms + d.n_atoms, dtype=np.float64)
            res = minimize(fun, v0, jac=True, method="L-BFGS-B", bounds=bounds,
                           options=dict(maxiter=o.maxiter, ftol=1e-9,
                                        gtol=1e-6, maxcor=12))
            v = res.x
            xyz = xyz0 + v[:3 * d.n_atoms].reshape(d.n_atoms, 3) * self.sx
            b_iso = b0 + v[3 * d.n_atoms:] * self.sb
            b_iso = np.clip(b_iso, o.b_min, o.b_max)

            if active_groups:
                fc_atoms = self._calc_fc_atoms(xyz, b_iso, d.occ)
                d.occ, scores = occupancy.refine_groups(
                    d, active_groups, xyz, b_iso, fc_atoms, self.fmask, self.scale,
                    self.mlpar, use_phase_prior=o.use_phase_prior, chunk=o.chunk,
                    prior_center=prior_center, prior_sigma=o.occupancy_prior_sigma)
                if scores is not None:
                    before, after, target_before, target_after = scores
                    self.occ_updates.append(dict(cycle=cyc, work_negll_before=before,
                                                 work_negll_after=after,
                                                 target_before=target_before, target_after=target_after))
                    self.log("  paired occupancy work -LL: %.4f -> %.4f" % (before, after))
                    if o.occupancy_prior_sigma is not None:
                        self.log("  occupancy penalized target: %.4f -> %.4f"
                                 % (target_before, target_after))

            # recompute everything with the new model
            fc_total = self._outer_update(xyz, b_iso)
            shift_a = np.linalg.norm(
                (xyz - xyz0) @ np.asarray(d.cell.orth.mat.tolist()).T, axis=1)
            rms_shift = float(np.sqrt(np.mean(shift_a ** 2)))
            rms_bshift = float(np.sqrt(np.mean((b_iso - b0) ** 2)))
            dt = time.time() - t0
            self.log("  L-BFGS: %d evals, %s, f_final=%.4f, |shift|=%.4f A"
                     % (res.nfev, res.message, float(res.fun), rms_shift))
            st = self._report(cyc, xyz, b_iso, fc_total, dt, rms_shift, rms_bshift)

            if o.auto_weight and o.weight is None:
                if o.weight_mode == "auto":
                    self._update_weight_auto(st)
                elif o.weight_mode == "heuristic":
                    self._update_weight_heuristic(st)

            # ---- early stopping on Rfree ------------------------------
            es = str(o.early_stop).strip().lower()
            if es == "paired":
                # Is the current model SIGNIFICANTLY worse than the best one?
                # The paired test is 4-6x more sensitive than the raw Rfree
                # difference (see likelihood.rfree_stats), which is what makes
                # this usable on a 5 % free set.
                if (self.best_rfree_cycle is not None
                        and self.best_rfree_cycle != cyc
                        and self.best_fc_total is not None):
                    try:
                        ps = likelihood.rfree_stats(
                            d, self.last_fc_total, self.scale,
                            ref_fc=self.best_fc_total, ref_scale=self.best_scale)
                        t = ps.get("t", float("nan"))
                    except Exception:          # pragma: no cover
                        t = float("nan")
                    if np.isfinite(t) and t >= 2.0:
                        self.early_stopped = True
                        self.log("  early stop: current model is significantly "
                                 "worse than the best (paired dRfree=%+.4f +/- "
                                 "%.4f, t=%+.1f); best Rfree %.4f at cycle %d"
                                 % (ps["delta_vs_ref"], ps["se_delta"], t,
                                    self.best_rfree, self.best_rfree_cycle))
                        break
            elif es.isdigit() and int(es) > 0:
                n_pat = int(es)
                if (np.isfinite(st.r_free) and self.best_rfree_cycle is not None
                        and cyc - self.best_rfree_cycle >= n_pat):
                    self.early_stopped = True
                    self.log("  early stop: Rfree has not improved for %d "
                             "cycle(s); best %.4f at cycle %d (now %.4f)"
                             % (cyc - self.best_rfree_cycle, self.best_rfree,
                                self.best_rfree_cycle, st.r_free))
                    break

        # ---- final outputs -------------------------------------------
        # Rfree is the only unbiased guide: with an honest free set the last
        # cycle is often not the best model (Rfree turns up while R keeps
        # falling).  Write the best-Rfree model by default; --no-keep-best
        # restores the old "last cycle" behaviour.
        last = self.cycles[-1] if self.cycles else None
        fc_last_total = self.last_fc_total      # last cycle's Fc (before refit)
        use_best = bool(o.keep_best and self.best_xyz is not None
                        and self.best_rfree_cycle is not None)
        if use_best and last is not None and self.best_rfree_cycle != last.cycle:
            self.log("  writing the BEST-Rfree model (cycle %d, Rfree=%.4f) "
                     "instead of the last cycle (%d, Rfree=%.4f)"
                     % (self.best_rfree_cycle, self.best_rfree,
                        last.cycle, last.r_free))
            xyz_out, b_out = self.best_xyz.copy(), self.best_b.copy()
            if self.best_occ is not None:
                d.occ = self.best_occ.copy()
            fc_total = self._outer_update(xyz_out, b_out)
            # The map coefficients (m, phase) are produced inside the cycle from
            # that cycle's Fc, so they must be rebuilt here: otherwise the MTZ
            # would carry FWT/DELFWT from the last cycle and FC/PHIC from the
            # best model, i.e. two different models in one file.
            try:
                _, _, m_best, ph_best, _ = likelihood.negll_and_G(
                    d, fc_total, self.scale, self.mlpar,
                    use_phase_prior=o.use_phase_prior, work_only=True)
                self.last_m, self.last_phase = m_best, ph_best
            except Exception as exc:        # pragma: no cover
                self.log("  ! could not rebuild map coefficients for the best "
                         "model (%s); FWT/DELFWT come from the last cycle" % exc)
        else:
            use_best = False
            xyz_out, b_out = xyz, b_iso
        written = dict(cycle=(self.best_rfree_cycle if use_best
                              else (last.cycle if last is not None else 0)),
                       r_work=(next((c.r_work for c in self.cycles
                                     if c.cycle == self.best_rfree_cycle), float("nan"))
                               if use_best else (last.r_work if last else float("nan"))),
                       r_free=(self.best_rfree if use_best
                               else (last.r_free if last else float("nan"))),
                       negll=(next((c.negll for c in self.cycles
                                     if c.cycle == self.best_rfree_cycle),
                                    float("nan"))
                              if use_best else (last.negll if last else float("nan"))),
                       model="best" if use_best else "last")
        # ---- is the Rfree difference we are acting on real? -------------
        # Rfree is a mean over n_free reflections; report its standard error and,
        # when the written model is not the last cycle, test the difference
        # PAIRED (see likelihood.rfree_stats).
        rfree_diag = None
        if np.any(d.test):
            ref_fc = fc_last_total if use_best else None
            ref_sc = None
            if ref_fc is not None and last is not None:
                try:
                    ref_sc = ScaleParams(**last.scale)
                except Exception:
                    ref_fc = None
            try:
                rfree_diag = likelihood.rfree_stats(
                    d, fc_total, self.scale, ref_fc=ref_fc, ref_scale=ref_sc)
            except Exception as exc:      # pragma: no cover
                self.log("  ! Rfree statistics skipped (%s)" % exc)
                rfree_diag = None
            if rfree_diag and np.isfinite(rfree_diag.get("se", float("nan"))):
                self.log("  Rfree = %.4f +/- %.4f (1 s.e., n_free=%d)"
                         % (rfree_diag["r_free"], rfree_diag["se"],
                            rfree_diag["n_free"]))
            if rfree_diag and np.isfinite(rfree_diag.get("t", float("nan"))):
                self.log("  Rfree(written) - Rfree(last) = %+.4f +/- %.4f "
                         "(paired t=%+.1f) -> %s"
                         % (rfree_diag["delta_vs_ref"], rfree_diag["se_delta"],
                            rfree_diag["t"],
                            "significant" if rfree_diag.get("significant")
                            else "WITHIN NOISE"))

        d.xyz, d.b_iso = xyz_out, b_out
        if o.out_pdb:
            dataio.write_refined_pdb(d, xyz_out, b_out, o.out_pdb,
                                     occ=d.occ if o.refine_occupancies else None)
            self.log("  wrote %s" % o.out_pdb)
        if o.out_mtz:
            # one extra Fc evaluation so FC/PHIC can be the atom-only
            # contribution (REFMAC's FC column is atom-only as well; FC_ALL is
            # the total including the bulk solvent -- verified against
            # refmac5 5.8 on the 7enm test data).
            fc_atoms_final = self._calc_fc_atoms(xyz_out, b_out, d.occ)
            hist = [
                "MINIREFMAC version=%s kernel=%s" % (__version__, self.kernel),
                "MINIREFMAC ncyc=%d maxiter=%d bins=%d solvent=%s"
                % (o.ncyc, o.maxiter, o.n_bins, o.solvent),
                "MINIREFMAC free_column=%s free_value=%d n_free=%d n_work=%d"
                % (d.free_label or "none", d.free_value,
                   int(np.sum(d.test)), int(np.sum(d.work))),
            ]
            hist.append("MINIREFMAC written_model=%s cycle=%d R_work=%.4f "
                        "R_free=%.4f"
                        % (written["model"], written["cycle"],
                           written["r_work"], written["r_free"]))
            if last is not None:
                hist.append("MINIREFMAC last_cycle R_work=%.4f R_free=%.4f negll=%.2f"
                            % (last.r_work, last.r_free, last.negll))
                hist.append("MINIREFMAC NOTE printed R uses the observed scale "
                            "with the per-reflection scale k; this file is on "
                            "the absolute scale, so recompute as "
                            "sum|F_obs - k*FC_ALL|/sum F_obs (see README)")
            maps.write_output_mtz(d, o.out_mtz, fc_total, self.scale,
                                  self.mlpar, self.last_m, self.last_phase,
                                  xyz_out, b_out, extra_columns=True,
                                  output_scale=o.map_scale,
                                  fc_atoms=fc_atoms_final,
                                  history_extra=hist)
            self.log("  wrote %s (scale=%s)" % (o.out_mtz, o.map_scale))

        total = time.time() - t_start
        result = dict(
            version="minirefmac %s" % __version__,
            mtz=d.mtz_path, pdb=d.pdb_path,
            spacegroup=d.spacegroup_name, n_ref=d.n_ref,
            n_atoms=d.n_atoms, d_min=d.d_min, d_max=d.d_max,
            options=asdict(o),
            cycles=[c.as_dict() for c in self.cycles],
            initial=dict(r_work=r0.r_work, r_free=r0.r_free, negll=r0.negll),
            final=dict(cycle=self.cycles[-1].cycle,
                       r_work=self.cycles[-1].r_work,
                       r_free=self.cycles[-1].r_free,
                       negll=self.cycles[-1].negll,
                       rmsd_bond=self.cycles[-1].rmsd_bond,
                       rmsd_angle_deg=self.cycles[-1].rmsd_angle_deg,
                       n_angle_outlier_4sigma=self.cycles[-1].n_angle_outlier_4sigma,
                       rmsz_bond=self.cycles[-1].rmsz_bond),
            n_target_evals=self.n_eval,
            map_scale=o.map_scale,
            reflection_columns=dict(f=d.fp_label, sigma=d.sigfp_label),
            kernel=self.kernel,
            kernel_reason=self.kernel_reason,
            free=dict(label=d.free_label, value=int(d.free_value),
                      n_free=int(np.sum(d.test)), n_work=int(np.sum(d.work)),
                      fraction=float((getattr(d, "free_info", {}) or {}).get(
                          "free_fraction", 0.0)),
                      mode=(getattr(d, "free_info", {}) or {}).get("mode")),
            data_stats=dict(getattr(d, "data_stats", {}) or {}),
            restraints=dict(self.restr.info),
            r_definition=("R = sum|F_obs - k*|Fc|| / sum F_obs (observed scale, "
                          "per-reflection k).  The output MTZ is written on the "
                          "absolute scale, so recomputing R from it gives a "
                          "slightly different value; see README_LINUX.md."),
            final_scale_params=self.scale.as_dict(),
            best_rfree=float(self.best_rfree),
            best_rfree_cycle=int(self.best_rfree_cycle),
            written=written,
            rfree_stats=rfree_diag or {},
            early_stopped=bool(getattr(self, "early_stopped", False)),
            wall_seconds=total,
            log="\n".join(self.log_lines),
        )
        if o.refine_occupancies:
            equiv_written = occupancy.diagnose_equivalence(
                d, self.occ_groups, xyz_out, component_overrides)
            result["occupancy"] = dict(
                updates=self.occ_updates, skipped_residues=self.occ_skipped,
                equivalent_overlap_threshold_A=occupancy.EQUIVALENCE_RMSD_A,
                frozen_overlapping_groups=len(frozen_groups),
                groups=[dict(chain=d.chain_ids[g.first[0]],
                             residue=int(d.res_seq[g.first[0]]), labels=list(g.labels),
                             initial=occ_input[[g.first[0], g.second[0]]].tolist(),
                             written=d.occ[[g.first[0], g.second[0]]].tolist(),
                             frozen_overlapping=index in frozen_groups,
                             equivalence_input=equiv_input.get(index),
                             equivalence_written=equiv_written.get(index))
                        for index, g in enumerate(self.occ_groups)])
        if o.out_json:
            with open(o.out_json, "w") as fh:
                json.dump(result, fh, indent=2, default=float)
            self.log("  wrote %s" % o.out_json)
        if o.out_log:
            with open(o.out_log, "w") as fh:
                fh.write("\n".join(self.log_lines) + "\n")
        self.log("  total wall time %.1f s, %d target evaluations" % (total, self.n_eval))
        return result
