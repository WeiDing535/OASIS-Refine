"""
minirefmac.likelihood -- maximum-likelihood target, its Fc-gradient and the
per-resolution-bin ML (sigmaA/Luzzati) parameter estimation.

Implements the interface contract declared at the bottom of core.py:
    estimate_ml_params(data, fc_total, scale, use="work", n_bins=20,
                       params0=None) -> MLParams
    negll_and_G(data, fc_total, scale, mlpar, use_phase_prior=False,
                work_only=True) -> (negll, G, m, phase, extra)
    negll_only(data, fc_total, scale, mlpar, use_phase_prior=False,
               set_mask=None) -> float
    r_factors(data, fc_total, scale) -> (r_work, r_free, r_all)

CONVENTIONS (must match core.py / sfcalc.py)
-------------------------------------------
  * fc_total is the ABSOLUTE-scale complex total structure factor (N,).
    The observations are brought onto the same scale by k_total:
        Fo_abs    = fo    / k_total
        SigFo_abs = sigfo / k_total
  * c = 0 acentric / 1 centric, eps = epsilon.
    Sigma = (2-c)*SigFo_abs**2 + eps*S        (clip to a small positive floor)
    DFc   = D * fc_total,  DFc_abs = D*|fc_total|
    X     = (2-c)*Fo_abs*DFc_abs / Sigma
  * Only OBSERVED reflections (finite, positive fo) are evaluated; the
    per-reflection results are scattered back into full (N,) arrays with zeros
    on missing reflections.

NEGATIVE LOG-LIKELIHOOD (per reflection; the +ln Sigma terms are kept because
they depend on the ML parameter S and are required by estimate_ml_params)
    acentric: -LL = (Fo^2 + DFc^2)/Sigma     - ln I0(X)   + ln Sigma
    centric : -LL = (Fo^2 + DFc^2)/(2 Sigma) - ln cosh(X) + 0.5 ln Sigma
with the numerically stable identities
    ln I0(X)   = X + log(i0e(X))                     (i0e = exp(-|X|) I0(X))
    I1(X)/I0(X)= i1e(X)/i0e(X)
    ln cosh(X) = logaddexp(X, -X) - log 2

G = 2 * d(-LL)/d fc_total*  (Wirtinger derivative).  With the chain rule
df/dp = Re sum_h conj(G_h) * d fc_total_h/dp  (p a real atomic parameter),
this G is exactly the coefficient consumed by gradient.py.  For no phase prior
    G = D*(2-c)*(D*|fc_total| - m*Fo_abs)/Sigma * exp(i arg fc_total),
    m = I1(X)/I0(X) (acentric) | tanh(X) (centric),
which is identical to Servalcat xtal.py::LL_Xtal.calc_grad / dll_dab.

MLHL EXPERIMENTAL PHASE PRIOR (use_phase_prior=True)
    A single-peaked von Mises prior  P(phi) ~ exp(kappa cos(phi - phib)) is
    represented by the complex Hendrickson-Lattman coefficient HL = kappa e^{i phib}
    (kappa = |HL|, obtained from FOM by inverting I1(kappa)/I0(kappa)=FOM).
    Combining it with the Rice term gives the complex field
        Z = HL + beta*fc_total,  beta = (2-c)*Fo_abs*D/Sigma,
        kappa_comb = |Z|,  m_comb = I1(kcomb)/I0(kcomb),  phase_comb = arg(Z).
    acentric: -LL = (Fo^2+DFc^2)/Sigma - ln I0(kcomb) + ln I0(kappa_pr) + ln Sigma
              G    = (2-c)*D^2*fc/Sigma - m_comb*beta*(Z/kcomb)
    centric : APPROXIMATION (marked below): we keep the centric ln cosh and
              replace X by the combined field magnitude kcomb:
              -LL = (Fo^2+DFc^2)/(2 Sigma) - ln cosh(kcomb) + ln I0(kappa_pr)
                    + 0.5 ln Sigma
              G    = (2-c)*D^2*fc/Sigma - tanh(kcomb)*beta*(Z/kcomb)
              This is an approximation because a rigorous centric treatment
              must SUM the prior over the two discrete allowed phases (phi0,
              phi0+pi) instead of using the complex von Mises combination Z.
              It is however exactly continuous with the no-prior formula in
              the limit HL -> 0 (kcomb -> X, ln I0(kappa_pr) -> 0), and the
              gradient reduces to the standard centric dll_dab there.
    Reflections WITHOUT a usable prior fall back to the no-prior formula.
"""
import math

import numpy as np
from scipy.special import i0e, i1e
from scipy.optimize import minimize

try:
    from .core import MLParams
except ImportError:  # pragma: no cover - allow running as a flat module
    from core import MLParams


__all__ = ["estimate_ml_params", "negll_and_G", "negll_only", "r_factors"]

_LOG2 = np.log(2.0)
_TINY = 1e-12


# ---------------------------------------------------------------------
# FOM -> von Mises concentration kappa (monotone lookup table, vectorised)
# ---------------------------------------------------------------------
_KAPPA_GRID = np.linspace(0.0, 50.0, 4001)
_RATIO_GRID = i1e(_KAPPA_GRID) / i0e(_KAPPA_GRID)   # I1/I0, monotone 0 -> ~0.99


def _fom_to_kappa(fom):
    """Invert  I1(kappa)/I0(kappa) = FOM  by table interpolation, kappa in [0,50]."""
    f = np.clip(np.asarray(fom, dtype=np.float64), 0.0, _RATIO_GRID[-1])
    return np.interp(f, _RATIO_GRID, _KAPPA_GRID)


def _build_prior(data, idx, use_phase_prior):
    """Return (HL (n,) complex128, has_prior (n,) bool) for the given subset.

    Priority: Hendrickson-Lattman A,B (only where C=D=0, i.e. a single-peaked
    von Mises distribution); otherwise (PHIB, FOM) columns.  If neither is
    available (or use_phase_prior is False) every reflection falls back to the
    no-prior likelihood.
    """
    n = len(idx)
    HL = np.zeros(n, dtype=np.complex128)
    has = np.zeros(n, dtype=bool)
    if not use_phase_prior:
        return HL, has

    if data.hl is not None:
        hl = np.asarray(data.hl, dtype=np.float64)[idx]        # (n,4) A,B,C,D
        single = (np.abs(hl[:, 2]) < 1e-8) & (np.abs(hl[:, 3]) < 1e-8)
        ab = hl[:, 0] + 1j * hl[:, 1]
        ok = single & np.isfinite(ab.real) & np.isfinite(ab.imag) & (np.abs(ab) > 0)
        HL[ok] = ab[ok]
        has[ok] = True
        return HL, has

    if (data.phib is not None) and (data.fom is not None):
        phib = np.asarray(data.phib, dtype=np.float64)[idx]
        fom = np.asarray(data.fom, dtype=np.float64)[idx]
        kap = _fom_to_kappa(fom)
        HL = kap * np.exp(1j * phib)
        has = np.isfinite(kap) & np.isfinite(phib) & (fom > 0.0) & (kap > 0.0)
        HL[~has] = 0.0
        return HL, has

    return HL, has


# ---------------------------------------------------------------------
# Core per-reflection evaluator (all inputs are (n,) arrays / broadcastable
# scalars).  Returns (ll, G, m, phase) each of shape (n,).
# ---------------------------------------------------------------------
def _negll_core(Fo, SigFo, fc, D, S, c, eps, HL, has_prior):
    Fo = np.asarray(Fo, dtype=np.float64)
    SigFo = np.asarray(SigFo, dtype=np.float64)
    fc = np.asarray(fc, dtype=np.complex128)
    c = np.asarray(c, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64)
    n = fc.shape[0]

    fc_abs = np.abs(fc)
    unit = np.divide(fc, fc_abs, out=np.zeros(n, dtype=np.complex128),
                     where=(fc_abs > 0.0))

    Sigma = np.maximum((2.0 - c) * SigFo ** 2 + eps * S, _TINY)
    lnSigma = np.log(Sigma)
    DFc_abs = D * fc_abs
    fo2 = Fo ** 2
    dfc2 = DFc_abs ** 2
    X = (2.0 - c) * Fo * DFc_abs / Sigma

    # ---- no-phase-prior branch (also the HL->0 limit of the prior branch) ----
    lnI0 = X + np.log(i0e(X))
    lncosh = np.logaddexp(X, -X) - _LOG2
    m_a = i1e(X) / i0e(X)
    m_c = np.tanh(X)
    m0 = np.where(c == 0.0, m_a, m_c)

    ll_a = (fo2 + dfc2) / Sigma - lnI0 + lnSigma
    ll_c = (fo2 + dfc2) / (2.0 * Sigma) - lncosh + 0.5 * lnSigma
    ll0 = np.where(c == 0.0, ll_a, ll_c)
    G0 = D * (2.0 - c) * (D * fc_abs - m0 * Fo) / Sigma * unit
    phase0 = np.angle(fc)

    if not bool(np.any(has_prior)):
        return ll0, G0, m0, phase0

    # ---- MLHL combined field -------------------------------------------
    beta = (2.0 - c) * Fo * D / Sigma                      # real (n,)
    Z = HL + beta * fc                                      # complex (n,)
    kcomb = np.abs(Z)
    kpr = np.abs(HL)
    Zu = np.divide(Z, kcomb, out=np.zeros(n, dtype=np.complex128),
                   where=(kcomb > _TINY))                  # unit vector Z/|Z|

    lnI0c = kcomb + np.log(i0e(kcomb))
    lncosh_c = np.logaddexp(kcomb, -kcomb) - _LOG2
    lnI0pr = kpr + np.log(i0e(kpr))
    mcomb_a = i1e(kcomb) / i0e(kcomb)
    mcomb_c = np.tanh(kcomb)
    mcomb = np.where(c == 0.0, mcomb_a, mcomb_c)

    llp_a = (fo2 + dfc2) / Sigma - lnI0c + lnI0pr + lnSigma
    # centric MLHL: APPROXIMATION (see module docstring).  ln cosh(kcomb)
    # rather than the rigorous two-point phase sum; continuous as HL->0.
    llp_c = ((fo2 + dfc2) / (2.0 * Sigma) - lncosh_c + lnI0pr
             + 0.5 * lnSigma)
    llp = np.where(c == 0.0, llp_a, llp_c)
    Gp = (2.0 - c) * D ** 2 * fc / Sigma - mcomb * beta * Zu
    phasep = np.angle(Z)

    hp = np.asarray(has_prior, dtype=bool)
    ll = np.where(hp, llp, ll0)
    G = np.where(hp, Gp, G0)
    m = np.where(hp, mcomb, m0)
    phase = np.where(hp, phasep, phase0)
    return ll, G, m, phase


# ---------------------------------------------------------------------
# Shared subset evaluator
# ---------------------------------------------------------------------
def _eval_at(data, fc_total, scale, mlpar, idx, use_phase_prior):
    k = np.asarray(scale.k_total(data), dtype=np.float64)
    fc = np.asarray(fc_total, dtype=np.complex128)
    Fo = np.asarray(data.fo, dtype=np.float64)[idx] / k[idx]
    SigFo = np.asarray(data.sigfo, dtype=np.float64)[idx] / k[idx]
    fcs = fc[idx]
    D = np.asarray(mlpar.D, dtype=np.float64)[idx]
    S = np.asarray(mlpar.S, dtype=np.float64)[idx]
    c = np.asarray(data.centric, dtype=np.float64)[idx]
    eps = np.asarray(data.epsilon, dtype=np.float64)[idx]
    HL, has = _build_prior(data, idx, use_phase_prior)
    return _negll_core(Fo, SigFo, fcs, D, S, c, eps, HL, has)


# ---------------------------------------------------------------------
# Public: negll_and_G
# ---------------------------------------------------------------------
def negll_and_G(data, fc_total, scale, mlpar,
                use_phase_prior=False, work_only=True):
    """Return (negll, G, m, phase, extra).

    G is (N,) complex128; it is ZERO outside the gradient mask
    (work_only=True -> data.work, work_only=False -> data.observed).
    `negll` always accumulates the WORKING set only (for monitoring Rfree the
    test-set -LL is reported separately in extra['negll_free']).
    m/phase are filled for every observed reflection (for map coefficients);
    missing reflections carry zeros.
    """
    n = data.n_ref
    obs = np.asarray(data.observed, dtype=bool)
    idx = np.flatnonzero(obs)
    ll = np.zeros(n, dtype=np.float64)
    G = np.zeros(n, dtype=np.complex128)
    m = np.zeros(n, dtype=np.float64)
    phase = np.zeros(n, dtype=np.float64)

    if idx.size:
        ll_s, G_s, m_s, ph_s = _eval_at(data, fc_total, scale, mlpar,
                                        idx, use_phase_prior)
        ll[idx] = ll_s
        m[idx] = m_s
        phase[idx] = ph_s
        gmask = np.asarray(data.work if work_only else data.observed,
                           dtype=bool)
        G[idx] = G_s * gmask[idx]

    work = np.asarray(data.work, dtype=bool)
    negll = float(np.sum(ll[work]))
    extra = {
        "negll_work": negll,
        "negll_free": negll_only(data, fc_total, scale, mlpar,
                                 use_phase_prior=use_phase_prior,
                                 set_mask=data.test),
        "mean_fom_work": (float(np.mean(m[work])) if work.any() else 0.0),
        "n_work": int(work.sum()),
    }
    return negll, G, m, phase, extra


# ---------------------------------------------------------------------
# Public: negll_only
# ---------------------------------------------------------------------
def negll_only(data, fc_total, scale, mlpar,
               use_phase_prior=False, set_mask=None):
    """Sum -LL over `set_mask` (default: work set).  Shares the exact same
    evaluator as negll_and_G, hence numerically identical (<1e-9 rel.)."""
    if set_mask is None:
        set_mask = data.work
    mask = (np.asarray(set_mask, dtype=bool)
            & np.asarray(data.observed, dtype=bool))
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0.0
    ll, _G, _m, _ph = _eval_at(data, fc_total, scale, mlpar,
                               idx, use_phase_prior)
    return float(np.sum(ll))


# ---------------------------------------------------------------------
# Public: estimate_ml_params
# ---------------------------------------------------------------------
def estimate_ml_params(data, fc_total, scale, use="work", n_bins=20,
                       params0=None):
    """Estimate per-resolution-bin D and S by minimising -sum LL with L-BFGS-B.

    Resolution bins are equal-count quantiles of s2 = 1/d^2 (rank based), so
    every bin holds ~N/n_bins reflections.  `use="work"` optimises on the work
    set (default); `use="free"` optimises on the free set (REFMAC's recipe for
    estimating Sigma_mod without over-fitting the working reflections).
    Small bins (<20 reflections) borrow reflections from neighbouring bins.
    D in [0.01, 1.5], S>0 are enforced by log-parameterisation + bounds.
    """
    n = data.n_ref
    n_bins = max(1, int(n_bins))
    fc = np.asarray(fc_total, dtype=np.complex128)
    fc_abs = np.abs(fc)
    k = np.asarray(scale.k_total(data), dtype=np.float64)
    Fo = np.asarray(data.fo, dtype=np.float64) / k
    SigFo = np.asarray(data.sigfo, dtype=np.float64) / k
    c = np.asarray(data.centric, dtype=np.float64)
    eps = np.asarray(data.epsilon, dtype=np.float64)
    obs = np.asarray(data.observed, dtype=bool)

    # ---- equal-count resolution bins on s2 ----
    s2 = np.asarray(data.s2, dtype=np.float64)
    order = np.argsort(s2, kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)
    bin_index = np.clip((ranks * n_bins) // max(n, 1), 0, n_bins - 1)
    bin_index = bin_index.astype(np.int64)

    if str(use).lower() == "free":
        est = np.asarray(data.test, dtype=bool)
    else:
        est = np.asarray(data.work, dtype=bool)

    # ---- initial D (optionally from params0) ----
    D_init = np.full(n_bins, 0.8, dtype=np.float64)
    if params0 is not None:
        try:
            if isinstance(params0, MLParams):
                D_init[:] = float(np.clip(np.mean(params0.D), 0.01, 1.5))
            elif isinstance(params0, dict):
                D_init[:] = float(np.clip(params0.get("D", 0.8), 0.01, 1.5))
            else:
                q = np.asarray(params0, dtype=np.float64).ravel()
                D_init[:] = float(np.clip(q[0], 0.01, 1.5))
        except Exception:
            pass

    D_bins = np.full(n_bins, 0.8, dtype=np.float64)
    S_bins = np.full(n_bins, 1.0, dtype=np.float64)
    n_ref_b = np.zeros(n_bins, dtype=np.int64)
    n_used_b = np.zeros(n_bins, dtype=np.int64)
    d_lo = np.zeros(n_bins, dtype=np.float64)
    d_hi = np.zeros(n_bins, dtype=np.float64)

    for b in range(n_bins):
        inb = (bin_index == b)
        sel0 = inb & est & obs
        nb = int(sel0.sum())
        n_ref_b[b] = nb
        if nb:
            d_lo[b] = float(np.min(data.d[sel0]))
            d_hi[b] = float(np.max(data.d[sel0]))
        if nb == 0:
            continue

        # borrow neighbours if this bin is too small
        sel = sel0
        if nb < 20:
            lo = max(0, b - 1)
            hi = min(n_bins, b + 2)
            wider = (bin_index >= lo) & (bin_index < hi) & est & obs
            if int(wider.sum()) > nb:
                sel = wider
        n_used_b[b] = int(sel.sum())

        fo_b = Fo[sel]
        sfo_b = SigFo[sel]
        fc_b = fc[sel]
        fc_abs_b = fc_abs[sel]
        c_b = c[sel]
        eps_b = eps[sel]
        m_b = int(sel.sum())

        # ---- robust initial S (absolute scale) ----
        D0 = float(D_init[b])
        var_fo = float(np.var(fo_b))
        mean_fc2 = float(np.mean(fc_abs_b ** 2))
        mean_fo2 = max(float(np.mean(fo_b ** 2)), 1.0)
        S0 = max(var_fo - D0 ** 2 * mean_fc2, 0.02 * mean_fo2)
        S0 = min(S0, 20.0 * mean_fo2)

        S_floor = 1e-4 * mean_fo2
        S_ceil = 100.0 * mean_fo2
        bounds = [(np.log(0.01), np.log(1.5)),
                  (np.log(S_floor), np.log(S_ceil))]
        p0 = np.array([np.log(np.clip(D0, 0.011, 1.49)),
                       np.log(np.clip(S0, S_floor * 1.001, S_ceil * 0.999))])

        HL_b = np.zeros(m_b, dtype=np.complex128)
        has_b = np.zeros(m_b, dtype=bool)

        def obj(p):
            D = np.exp(p[0])
            S = np.exp(p[1])
            ll, _G, _m, _ph = _negll_core(fo_b, sfo_b, fc_b, D, S,
                                          c_b, eps_b, HL_b, has_b)
            return float(np.sum(ll))

        try:
            res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 80, "ftol": 1e-10})
            p = res.x
        except Exception:
            p = p0
        D_bins[b] = float(np.exp(p[0]))
        S_bins[b] = float(np.exp(p[1]))

    # ---- light smoothing of the bin curves (stabilises D/S, avoids overfit) ----
    def _smooth(v):
        ker = np.array([1.0, 2.0, 1.0]) / 4.0
        padded = np.pad(v, (1, 1), mode="edge")
        s = np.convolve(padded, ker, mode="valid")
        return 0.5 * v + 0.5 * s

    D_bins = _smooth(D_bins)
    S_bins = np.maximum(_smooth(S_bins), 1e-6)

    # ---- expand to per-reflection arrays ----
    D = D_bins[bin_index]
    S = S_bins[bin_index]

    # ---- bin_stats ----
    bin_stats = []
    for b in range(n_bins):
        sel = (bin_index == b) & est & obs
        nb = int(sel.sum())
        if nb == 0:
            bin_stats.append(dict(n_ref=0, n_used=int(n_used_b[b]),
                                  d_range=[float(d_lo[b]), float(d_hi[b])],
                                  D=float(D_bins[b]), S=float(S_bins[b]),
                                  mean_fom=0.0, cc=0.0,
                                  mean_FOM=0.0, CC=0.0))
            continue
        fo_b = Fo[sel]
        sfo_b = SigFo[sel]
        fc_b = fc[sel]
        c_b = c[sel]
        eps_b = eps[sel]
        HL_b = np.zeros(nb, dtype=np.complex128)
        has_b = np.zeros(nb, dtype=bool)
        _ll, _G, mb, _ph = _negll_core(fo_b, sfo_b, fc_b,
                                       float(D_bins[b]), float(S_bins[b]),
                                       c_b, eps_b, HL_b, has_b)
        mean_fom = float(np.mean(mb))
        a = fo_b
        bb = np.abs(fc_b)
        if nb > 1 and np.std(a) > 0 and np.std(bb) > 0:
            cc = float(np.corrcoef(a, bb)[0, 1])
        else:
            cc = 0.0
        bin_stats.append(dict(n_ref=nb, n_used=int(n_used_b[b]),
                              d_range=[float(d_lo[b]), float(d_hi[b])],
                              D=float(D_bins[b]), S=float(S_bins[b]),
                              mean_fom=mean_fom, cc=cc,
                              mean_FOM=mean_fom, CC=cc))

    return MLParams(bin_index=bin_index, D=D, S=S, n_bins=n_bins,
                    bin_stats=bin_stats, use=str(use))


# ---------------------------------------------------------------------
# Public: r_factors
# ---------------------------------------------------------------------
def r_factors(data, fc_total, scale):
    """R = sum |fo - k_total*|fc_total|| / sum fo on work / test / observed."""
    k = np.asarray(scale.k_total(data), dtype=np.float64)
    fo = np.asarray(data.fo, dtype=np.float64)
    fcalc = k * np.abs(np.asarray(fc_total, dtype=np.complex128))
    obs = np.asarray(data.observed, dtype=bool)
    diff = np.abs(fo - fcalc)

    def R(mask):
        m = np.asarray(mask, dtype=bool) & obs
        den = float(np.sum(fo[m]))
        if den <= 0.0:
            return float("nan")
        return float(np.sum(diff[m]) / den)

    return R(data.work), R(data.test), R(obs)


def rfree_stats(data, fc_total, scale, ref_fc=None, ref_scale=None):
    """Rfree with its standard error, plus a paired test against a reference model.

    Rfree is a ratio of sums, ``R = sum|Fo - k|Fc|| / sum Fo`` over the free
    set, so the numerator ``A = sum|d_i|`` carries all the sampling noise while
    the denominator ``B = sum Fo`` is fixed by the data.  Hence

        SE(R)      = sd(|d_i|) * sqrt(n) / B
        dR (A vs B) = sum(|d_i^A| - |d_i^B|) / B      (paired, same reflections)
        SE(dR)     = sd(|d_i^A| - |d_i^B|) * sqrt(n) / B

    The paired form is far more sensitive than comparing two independent
    estimates, because two models of the same crystal share Fo and have highly
    correlated |d_i|.  We report ``t = dR/SE(dR)`` and call a difference real
    when ``|t| >= 2``.

    Note: the mean-of-ratios ``mean(|d_i|/Fo)`` is NOT Rfree -- it is dominated
    by the weakest reflections and reads much higher (0.50 vs 0.40 on 7enm).

    Returns a dict; keys are present only when they can be computed.
    """
    obs = np.asarray(data.observed, dtype=bool)
    test = np.asarray(data.test, dtype=bool) & obs
    n = int(np.sum(test))
    out = {"n_free": n}
    if n < 2:
        return out
    fo = np.asarray(data.fo, dtype=np.float64)
    good = test & (fo > 0.0)
    n = int(np.sum(good))
    if n < 2:
        return out
    fo_f = fo[good]
    B = float(np.sum(fo_f))
    k = np.asarray(scale.k_total(data), dtype=np.float64)
    d1 = np.abs(fo - k * np.abs(np.asarray(fc_total, dtype=np.complex128)))[good]
    A = float(np.sum(d1))
    out.update(n_free=n, r_free=A / B if B > 0 else float("nan"),
               se=(float(np.std(d1, ddof=1)) * math.sqrt(n) / B
                   if B > 0 else float("nan")),
               sum_fo=B)
    if ref_fc is not None and ref_scale is not None:
        k2 = np.asarray(ref_scale.k_total(data), dtype=np.float64)
        d2 = np.abs(fo - k2 * np.abs(np.asarray(ref_fc, dtype=np.complex128)))[good]
        diff = d1 - d2
        mean_d = float(np.sum(diff)) / B if B > 0 else float("nan")
        se_d = (float(np.std(diff, ddof=1)) * math.sqrt(n) / B
                if B > 0 else float("nan"))
        out.update(delta_vs_ref=mean_d, se_delta=se_d,
                   t=(mean_d / se_d if se_d and np.isfinite(se_d) and se_d > 0
                      else float("nan")),
                   ref_r_free=float(np.sum(d2)) / B if B > 0 else float("nan"))
        # identical models give mean_d = se_d = 0: not a difference
        out["significant"] = bool(se_d > 0.0 and abs(mean_d) >= 2.0 * se_d)
    return out
