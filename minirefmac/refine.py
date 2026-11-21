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
import time
from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import minimize

try:
    from . import dataio, sfcalc, likelihood, gradient, geometry, maps, fastkernel
    from . import simdkernel, fftcalc
    from .core import ScaleParams, CycleStats
except ImportError:  # pragma: no cover - allow flat-module execution
    import dataio, sfcalc, likelihood, gradient, geometry, maps, fastkernel
    from core import ScaleParams, CycleStats
    import simdkernel, fftcalc


@dataclass
class Options:
    ncyc: int = 5
    maxiter: int = 10
    n_bins: int = 20
    solvent: str = "mask"           # mask | babinet | none
    fit_aniso: bool = True
    use_phase_prior: bool = False  # MLHL
    use_monlib: bool = True
    weight: float = None            # explicit w_xray, overrides auto balance
    # Calibrated on the real SAD cases: 1.0 (pure gradient-norm balance)
    # over-fits (Rfree rises, geometry loosens).  0.05 reproduces REFMAC's
    # strong geometry restraints and lowers Rfree.
    weight_scale: float = 0.05
    auto_weight: bool = True
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
    # Experimental kernel fusion: compute the phase E once per evaluation and
    # share it between Fc and the atomic gradients (see fastkernel.py).  False
    # (default) keeps the original two-pass path bit-for-bit unchanged.
    fused_kernel: bool = False
    # STEP 2: AVX-512 SIMD C kernel (simdkernel.py / _simd.c).  Same fused math
    # as fastkernel but the inner loop runs in hand-written AVX-512.  Requires
    # an AVX-512 capable CPU; falls back at import time if unavailable.
    simd_kernel: bool = False
    # STEP 3: O(G log G) FFT structure factors (fftcalc.py).  Builds the
    # electron density on a grid, one FFT, sample at HKL.  Fc ~0.1-0.2% of the
    # direct sum on a ~0.3 A grid; the adjoint-FFT gradient shares the same cost.
    fft_kernel: bool = False
    fft_spacing: float = None       # None -> auto (memory-aware, ~0.3 A)
    fft_radius: float = None        # density cut-off radius (A); None -> per-atom adaptive
    # REFMAC5 writes F/SIGF and all map coefficients on the ABSOLUTE scale
    # (F_abs = F_obs / k_total).  "observed" keeps the legacy input scale.
    map_scale: str = "absolute"
    verbose: bool = True
    out_pdb: str = None
    out_mtz: str = None
    out_log: str = None
    out_json: str = None


class Refiner:
    """Maximum-likelihood restrained refinement engine."""

    def __init__(self, data, opts=None):
        self.data = data
        self.opts = opts or Options()
        self.log_lines = []
        self.cycles = []
        self.restr = None
        self.scale = None
        self.mlpar = None
        self.fmask = None
        self.w_xray = 1.0
        self.n_eval = 0
        self.last_fc_total = None
        self.last_m = None
        self.last_phase = None
        # frozen cycle-0 ML parameters / scale, used for a comparable -LL
        self.mlpar0 = None
        self.scale0 = None
        self.best_rfree = float("inf")
        self.best_rfree_cycle = 0
        self.best_xyz = None
        self.best_b = None

        # parameter scaling: v = 1 corresponds to 0.1 A (xyz) / 5 A^2 (B)
        a_mean = float(np.mean(data.cell_parameters[:3])) if data.cell_parameters else 50.0
        self.sx = 0.1 / max(a_mean, 1.0)
        self.sb = 5.0

    # ------------------------------------------------------------------
    def log(self, msg):
        self.log_lines.append(msg)
        if self.opts.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------
    def _prepare(self):
        o = self.opts
        self.restr = geometry.Restraints(
            self.data,
            w_bond=o.w_bond, w_angle=o.w_angle, w_b=o.w_b,
            w_vdw=o.w_vdw, sym_vdw=o.sym_vdw,
            use_monlib=o.use_monlib)
        self.log("Geometry restraints: %d bonds, %d angles (%d cross-peptide), "
                 "%d planar groups (%d peptide), %d torsions, %d chirs, "
                 "%d VDW pairs (%d symmetry) (%s)"
                 % (self.restr.n_bonds, self.restr.n_angles,
                    self.restr.info["n_cross_angle"],
                    self.restr.info["n_plane_groups"],
                    self.restr.info["n_link_plane"],
                    self.restr.info["n_torsion"],
                    self.restr.info["n_chir"],
                    self.restr.info["n_vdw"], self.restr.info["n_vdw_sym"],
                    ("jelly-body fallback (dictionary disabled)"
                     if not o.use_monlib else "bundled monomer dictionary")))

    # ------------------------------------------------------------------
    def _outer_update(self, xyz, b_iso, first=False):
        """Refresh mask, scales and ML parameters with the current model."""
        d, o = self.data, self.opts
        fc_atoms = sfcalc.calc_fc(d, xyz, b_iso, d.occ)
        try:
            self.fmask = sfcalc.calc_fmask(d, xyz) if o.solvent == "mask" else None
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
        self.last_fc_total = fc_total
        return fc_total

    # ------------------------------------------------------------------
    def _balance_weight(self, xyz, b_iso, fc_total):
        """Set w_xray so that X-ray and geometry gradient norms are comparable."""
        o = self.opts
        if o.weight is not None:
            self.w_xray = float(o.weight)
            return
        d = self.data
        negll, G, m, phase, extra = likelihood.negll_and_G(
            d, fc_total, self.scale, self.mlpar,
            use_phase_prior=o.use_phase_prior, work_only=True)
        fcf = self.scale.fc_atom_factor(d)
        _, gx, gb = gradient.fc_and_atomic_gradients(
            d, xyz, b_iso, d.occ, G, fcf, chunk=o.chunk, want_fc=False)
        fg, ggx, ggb, stats = self.restr.calc(xyz, b_iso)
        n_x = math.sqrt(float(np.sum(gx ** 2) + np.sum(gb ** 2)))
        n_g = math.sqrt(float(np.sum(ggx ** 2) + np.sum(ggb ** 2)))
        self.w_xray = o.weight_scale * n_g / max(n_x, 1e-12)
        self.w_base = self.w_xray
        self.log("  weight balance: |g_xray|=%.4g |g_geom|=%.4g -> w_xray=%.6g"
                 % (n_x, n_g, self.w_xray))

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

        # ---- optional kernel-fusion warm start (see fastkernel.py) --------
        # The fused kernel needs G *before* it evaluates the shared phase E,
        # but G depends on the current Fc (chicken-and-egg).  We warm-start G
        # from the v=0 Fc (self.last_fc_total, produced by _outer_update) and
        # refresh it after every evaluation; the first L-BFGS evaluation (v=0)
        # is therefore bit-identical to the original path.  Default path
        # (fused_kernel=False) does not enter this block at all.
        g_warm = [None]
        if o.fused_kernel or o.simd_kernel:
            base_fc_total = self.last_fc_total
            if base_fc_total is None:  # pragma: no cover - direct _make_target call
                base_fc_total = sfcalc.build_fc_total(
                    d, sfcalc.calc_fc(d, xyz0, b0, occ), fmask, scale)
            _, G0, _, _, _ = likelihood.negll_and_G(
                d, base_fc_total, scale, mlpar,
                use_phase_prior=prior, work_only=True)
            g_warm[0] = G0

        def fun(v):
            self.n_eval += 1
            xyz = xyz0 + v[:3 * m].reshape(m, 3) * sx
            b_iso = b0 + v[3 * m:] * sb

            if o.fused_kernel:
                fc_atoms, gx, gb = fastkernel.fused_fc_and_gradients(
                    d, xyz, b_iso, occ, g_warm[0], fcf, chunk=o.chunk)
                fc_total = sfcalc.build_fc_total(d, fc_atoms, fmask, scale)
                negll, G, mm, phase, extra = likelihood.negll_and_G(
                    d, fc_total, scale, mlpar,
                    use_phase_prior=prior, work_only=True)
                g_warm[0] = G
            elif o.simd_kernel:
                fc_atoms, gx, gb = simdkernel.simd_fc_and_gradients(
                    d, xyz, b_iso, occ, g_warm[0], fcf, chunk=o.chunk)
                fc_total = sfcalc.build_fc_total(d, fc_atoms, fmask, scale)
                negll, G, mm, phase, extra = likelihood.negll_and_G(
                    d, fc_total, scale, mlpar,
                    use_phase_prior=prior, work_only=True)
                g_warm[0] = G
            elif o.fft_kernel:
                # O(G log G): grid density -> one FFT -> sample at HKL.
                # Scatter the density ONCE for the *trial* model and reuse the
                # grid for the adjoint gradients (a second scatter + FFT would
                # double the cost).
                _sp = o.fft_spacing if o.fft_spacing is not None else \
                    fftcalc.auto_spacing(d.cell, getattr(d, "d_min", None))
                _sym = sfcalc._sym_ops(d)
                _grid, _atoms, _orth = fftcalc._scatter_density(
                    d, _sp, _sym, o.fft_radius, xyz=xyz, b_iso=b_iso, occ=occ)
                fc_atoms = fftcalc.fft_fc(d, spacing=_sp, radius_A=o.fft_radius,
                                          grid=_grid, atoms=_atoms)
                fc_total = sfcalc.build_fc_total(d, fc_atoms, fmask, scale)
                negll, G, mm, phase, extra = likelihood.negll_and_G(
                    d, fc_total, scale, mlpar,
                    use_phase_prior=prior, work_only=True)
                _, gx, gb = fftcalc.fft_gradients(
                    d, xyz, b_iso, occ, G, fcf, spacing=_sp,
                    radius_A=o.fft_radius, want_fc=False, grid=_grid)
            else:
                fc_atoms = sfcalc.calc_fc(d, xyz, b_iso, occ)
                fc_total = sfcalc.build_fc_total(d, fc_atoms, fmask, scale)
                negll, G, mm, phase, extra = likelihood.negll_and_G(
                    d, fc_total, scale, mlpar, use_phase_prior=prior, work_only=True)

                _, gx, gb = gradient.fc_and_atomic_gradients(
                    d, xyz, b_iso, occ, G, fcf, chunk=o.chunk, want_fc=False)
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
                        rmsd_angle=stats.get("rmsd_angle", float("nan")),
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
        self.log("  cycle %2d  R=%7.4f  Rfree=%7.4f  -LL=%12.2f  -LLfix=%12.2f"
                 "  FOM=%.3f  rmsBOND=%.4f  rmsANGL=%.4f  w=%.3g  %.1fs"
                 % (cyc, r_w, r_f, negll, negll_fixed, st.mean_fom,
                    st.rmsd_bond, st.rmsd_angle, self.w_xray, dt))
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
        self.log("=" * 78)

        self._prepare()
        xyz = d.xyz.copy()
        b_iso = d.b_iso.copy()
        xyz_start, b_start = xyz.copy(), b_iso.copy()

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

        # ---- final outputs -------------------------------------------
        d.xyz, d.b_iso = xyz, b_iso
        if o.out_pdb:
            dataio.write_refined_pdb(d, xyz, b_iso, o.out_pdb)
            self.log("  wrote %s" % o.out_pdb)
        if o.out_mtz:
            # one extra Fc evaluation so FC/PHIC can be the atom-only
            # contribution (REFMAC's FC column excludes the bulk solvent).
            fc_atoms_final = sfcalc.calc_fc(d, xyz, b_iso, d.occ)
            maps.write_output_mtz(d, o.out_mtz, fc_total, self.scale,
                                  self.mlpar, self.last_m, self.last_phase,
                                  xyz, b_iso, extra_columns=True,
                                  output_scale=o.map_scale,
                                  fc_atoms=fc_atoms_final)
            self.log("  wrote %s (scale=%s)" % (o.out_mtz, o.map_scale))

        total = time.time() - t_start
        result = dict(
            version="minirefmac 0.1.0",
            mtz=d.mtz_path, pdb=d.pdb_path,
            spacegroup=d.spacegroup_name, n_ref=d.n_ref,
            n_atoms=d.n_atoms, d_min=d.d_min, d_max=d.d_max,
            options=asdict(o),
            cycles=[c.as_dict() for c in self.cycles],
            initial=dict(r_work=r0.r_work, r_free=r0.r_free, negll=r0.negll),
            final=dict(r_work=self.cycles[-1].r_work,
                       r_free=self.cycles[-1].r_free,
                       negll=self.cycles[-1].negll,
                       rmsd_bond=self.cycles[-1].rmsd_bond,
                       rmsz_bond=self.cycles[-1].rmsz_bond),
            n_target_evals=self.n_eval,
            map_scale=o.map_scale,
            final_scale_params=self.scale.as_dict(),
            best_rfree=float(self.best_rfree),
            best_rfree_cycle=int(self.best_rfree_cycle),
            wall_seconds=total,
            log="\n".join(self.log_lines),
        )
        if o.out_json:
            with open(o.out_json, "w") as fh:
                json.dump(result, fh, indent=2, default=float)
            self.log("  wrote %s" % o.out_json)
        if o.out_log:
            with open(o.out_log, "w") as fh:
                fh.write("\n".join(self.log_lines) + "\n")
        self.log("  total wall time %.1f s, %d target evaluations" % (total, self.n_eval))
        return result
