"""
minirefmac.core -- shared data structures and numerical conventions.

This module is the *interface contract* for all other modules.  Every array is
numpy float64 unless stated otherwise.  All modules import from here and must
not redefine these structures.

FOURIER CONVENTION (must be used consistently everywhere)
-------------------------------------------------------
    Fc(h) = sum_j q_j f_j(stol2) exp(-B_j * stol2) exp(2*pi*i * h . x_j)

    stol2 = (sin(theta)/lambda)^2 = 1/(4 d^2)
    s2    = 1/d^2 = 4 * stol2
    f_j   = IT92 Cromer-Mann X-ray scattering factor evaluated at stol2.

SCALE CONVENTION
----------------
    fc_atoms = sum_j ...                       (absolute electron scale, complex)
    fc_total = fc_atoms + k_sol*exp(-b_sol*stol2)*fmask     (solvent="mask")
             = fc_atoms * (1 - k_sol*exp(-b_sol*stol2))     (solvent="babinet")
             = fc_atoms                                     (solvent="none")
    k_total  = k_overall * exp(-b_over*stol2) * exp(-0.25*aniso_quad(h))

    The observed amplitudes are converted to the absolute scale with
        Fo_abs = fo / k_total,  SigFo_abs = sigfo / k_total
    and the likelihood is evaluated between Fo_abs and D*fc_total.

    fc_atom_factor(h) = d fc_total / d fc_atoms
                      = 1.0                              (mask / none)
                      = 1 - k_sol*exp(-b_sol*stol2)     (babinet)
    Atomic-parameter gradients must be multiplied by fc_atom_factor(h).

LIKELIHOOD (per reflection, work set only)
-----------------------------------------
    c      = 0 acentric, 1 centric
    eps    = epsilon (multiplicity)
    Sigma  = (2-c)*SigFo_abs**2 + eps*S
    DFc    = D * fc_total
    X      = (2-c)*Fo_abs*|DFc|/Sigma
    m      = I1(X)/I0(X)  (acentric)  |  tanh(X)  (centric)
    -LL_a  = (Fo_abs**2+|DFc|**2)/Sigma - ln I0(X) + ln Sigma
    -LL_c  = (Fo_abs**2+|DFc|**2)/(2*Sigma) - ln cosh(X) + 0.5*ln Sigma

    G_h    = 2 * d(-LL)/d fc_total*  = D*(2-c)*(D*|fc_total| - m*Fo_abs)/Sigma
                                       * exp(i*arg(fc_total))
    (This is exactly Servalcat's `dll_dab` / REFMAC's ML difference-map
     coefficient.  See likelihood.py for the phase-prior (MLHL) variant.)

ATOMIC GRADIENTS (gradient.py)
------------------------------
    C_j(h) = conj(G_h) * fc_atom_factor(h) * q_j*f_j*exp(-B_j*stol2)*exp(2*pi*i*h.x_j)
    df/dx_j = -2*pi * sum_h h_k * Im[C_j(h)]
    df/dB_j = -0.25 * sum_h s2(h) * Re[C_j(h)]

WEIGHTS
-------
    f_total = w_xray * f_xray + f_geom
    f_xray  = sum_h (-LL_h) over working reflections
    w_xray  is adapted so that rmsd(bond) stays in [0.015, 0.025] A.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
import gemmi

__all__ = ["RefineData", "ScaleParams", "MLParams", "CycleStats", "FOURIER_NOTE"]

FOURIER_NOTE = "Fc(h) = sum_j q f_j(stol2) exp(-B stol2) exp(2*pi*i*h.x)"


@dataclass
class RefineData:
    """All reflection + model data needed by the refinement engine."""

    # ---- crystal ----
    cell: Any                       # gemmi.Cell
    sg: Any                         # gemmi.SpaceGroup
    spacegroup_name: str = ""
    cell_parameters: tuple = ()     # (a,b,c,alpha,beta,gamma)

    # ---- reflections, shape (N,) -------------------------------------
    hkl: np.ndarray = None          # int32 (N,3)
    d: np.ndarray = None            # d spacing in A
    stol2: np.ndarray = None        # 1/(4 d^2)
    s2: np.ndarray = None           # 1/d^2 = 4*stol2
    epsilon: np.ndarray = None      # multiplicity, float64 >= 1
    centric: np.ndarray = None      # int, 0 = acentric, 1 = centric
    recip_vec: np.ndarray = None    # (N,3) orthogonal reciprocal vector s (1/A)

    fo: np.ndarray = None           # observed amplitude, NaN if missing
    sigfo: np.ndarray = None
    free: np.ndarray = None         # integer FreeR flag
    is_free: np.ndarray = None      # bool, True for the excluded (test) set
    observed: np.ndarray = None     # bool, np.isfinite(fo) & fo > 0

    # optional experimental phase prior (MLHL); radians + weight
    phib: Optional[np.ndarray] = None   # radians
    fom: Optional[np.ndarray] = None    # 0..1
    hl: Optional[np.ndarray] = None     # (N,4) Hendrickson-Lattman A,B,C,D

    # ---- model, shape (M,) ------------------------------------------
    xyz: np.ndarray = None          # (M,3) fractional coordinates
    b_iso: np.ndarray = None        # (M,)
    occ: np.ndarray = None          # (M,)
    elements: List[str] = None      # (M,) element symbols
    atom_names: List[str] = None
    altloc: List[str] = None
    res_names: List[str] = None
    res_seq: np.ndarray = None
    chain_ids: List[str] = None
    is_hydrogen: np.ndarray = None  # bool (M,)

    # ---- originals for writing output -------------------------------
    st: Any = None                  # gemmi.Structure (kept in sync on write)
    mtz: Any = None                 # gemmi.Mtz
    mtz_path: str = ""
    pdb_path: str = ""
    fp_label: str = "F"
    sigfp_label: str = "SIGF"
    free_label: str = "FreeR_flag"
    free_value: int = 0
    d_min: float = 0.0
    d_max: float = 0.0

    # ---- diagnostics (filled by dataio.load_data) --------------------
    # free_info : how the FreeR flag value was chosen (explicit / auto),
    #             the flag histogram and the resulting free fraction.
    # data_stats: reflection-level sanity counters (missing F, bad sigma,
    #             duplicate HKL, altlocs, ...).  Reported in refine.log and
    #             report.json so a silently wrong input never goes unnoticed.
    free_info: Dict[str, Any] = field(default_factory=dict)
    data_stats: Dict[str, Any] = field(default_factory=dict)

    # precomputed scattering factors f_j(h), shape (N,M)
    sf_table: np.ndarray = None
    # Low-memory alternative: one IT92 curve per distinct element plus the
    # per-atom column map into it.  Both are ALWAYS populated; the full
    # (N,M) sf_table is kept only for small/medium systems (or FORCE_FULL).
    sf_unique: np.ndarray = None     # (N, n_elem) float64
    elem_index: np.ndarray = None    # (M,) int
    sf_table_nbytes: int = 0         # bytes actually used by the sf storage

    def sf_block(self, c0, c1):
        """Scattering factors for atoms [c0:c1], shape (N, c1-c0) float64.

        Small/medium systems slice the full (N,M) table bit-for-bit unchanged;
        large systems gather the per-element curves (numpy advanced indexing).
        """
        if self.sf_table is not None:
            return self.sf_table[:, c0:c1]
        return self.sf_unique[:, self.elem_index[c0:c1]]

    @property
    def n_ref(self):
        return len(self.hkl)

    @property
    def n_atoms(self):
        return len(self.xyz)

    @property
    def work(self):
        """Boolean mask: observed working reflections."""
        return self.observed & ~self.is_free

    @property
    def test(self):
        return self.observed & self.is_free


@dataclass
class ScaleParams:
    """Overall / anisotropic scaling and bulk-solvent parameters."""

    k_overall: float = 1.0
    b_over: float = 0.0
    # symmetric traceless tensor in the orthogonal reciprocal basis,
    # ordered (b11,b22,b33,b12,b13,b23); trace is removed by the fitter.
    b_aniso: np.ndarray = None
    k_sol: float = 0.35
    b_sol: float = 100.0
    solvent: str = "mask"        # "mask" | "babinet" | "none"

    def __post_init__(self):
        if self.b_aniso is None:
            self.b_aniso = np.zeros(6, dtype=np.float64)

    def aniso_quad(self, data):
        """Quadratic form b11*sx^2 + b22*sy^2 + ... using data.recip_vec."""
        s = data.recip_vec
        b11, b22, b33, b12, b13, b23 = self.b_aniso
        return (b11 * s[:, 0] ** 2 + b22 * s[:, 1] ** 2 + b33 * s[:, 2] ** 2
                + 2 * b12 * s[:, 0] * s[:, 1]
                + 2 * b13 * s[:, 0] * s[:, 2]
                + 2 * b23 * s[:, 1] * s[:, 2])

    def k_total(self, data):
        """Real positive scale from absolute model scale to observed scale."""
        return (self.k_overall * np.exp(-self.b_over * data.stol2)
                * np.exp(-0.25 * self.aniso_quad(data)))

    def fc_atom_factor(self, data):
        """d fc_total / d fc_atoms (real, per reflection)."""
        if self.solvent == "babinet":
            return 1.0 - self.k_sol * np.exp(-self.b_sol * data.stol2)
        return np.ones(data.n_ref, dtype=np.float64)

    def as_dict(self):
        return dict(k_overall=self.k_overall, b_over=self.b_over,
                    b_aniso=list(map(float, self.b_aniso)),
                    k_sol=self.k_sol, b_sol=self.b_sol, solvent=self.solvent)


@dataclass
class MLParams:
    """Resolution-bin ML parameters, expanded to per-reflection arrays."""

    bin_index: np.ndarray = None    # (N,) int
    D: np.ndarray = None            # (N,) Luzzati/sigmaA-like D
    S: np.ndarray = None            # (N,) model error variance (absolute scale)
    n_bins: int = 20
    bin_stats: List[Dict[str, Any]] = field(default_factory=list)
    use: str = "work"               # which set was used for estimation


@dataclass
class CycleStats:
    cycle: int = 0
    r_work: float = float("nan")
    r_free: float = float("nan")
    negll: float = float("nan")
    # -LL evaluated with the ML parameters (D, S) frozen at their cycle-0
    # values.  The plain `negll` is NOT comparable across cycles because
    # sigmaA is re-estimated every cycle; this one is.
    negll_fixed: float = float("nan")
    negll_free: float = float("nan")
    geom: float = float("nan")
    rmsd_bond: float = float("nan")
    rmsd_angle: float = float("nan")
    rmsz_bond: float = float("nan")
    w_xray: float = float("nan")
    mean_fom: float = float("nan")
    rms_shift: float = float("nan")
    rms_bshift: float = float("nan")
    ll_time: float = 0.0
    scale: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self):
        return {k: (float(v) if isinstance(v, (np.floating, float, int)) else v)
                for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------
# Module interface contract (implemented by the parallel work packages)
# ---------------------------------------------------------------------
#
# dataio.py
#   load_data(mtz_path, pdb_path, fp_label=None, sigfp_label=None,
#             free_label=None, free_value=0, phib_label=None, fom_label=None,
#             hla_label=None, hlb_label=None, hlc_label=None, hld_label=None,
#             d_min=0.0, d_max=0.0, keep_hydrogens=False) -> RefineData
#   make_sf_table(data) -> np.ndarray (N,M)   # cached in data.sf_table
#   write_refined_pdb(data, xyz, b_iso, path) -> None
#
# sfcalc.py
#   calc_fc(data, xyz, b_iso, occ=None, chunk=64) -> np.ndarray (N,) complex128
#   calc_fmask(data, xyz=None) -> np.ndarray (N,) complex128
#   fit_scales(data, fc_atoms, fmask=None, solvent="mask",
#              fit_aniso=True, params0=None) -> ScaleParams
#   build_fc_total(data, fc_atoms, fmask, scale) -> np.ndarray (N,) complex128
#
# likelihood.py
#   estimate_ml_params(data, fc_total, scale, use="work", n_bins=20,
#                      params0=None) -> MLParams
#   negll_and_G(data, fc_total, scale, mlpar, use_phase_prior=False,
#               work_only=True) -> (negll, G, m, phase, extra)
#       G     : (N,) complex128  -- 2*d(-LL)/d fc_total*, ZERO on free/missing
#       m     : (N,) float64     -- figure of merit
#       phase : (N,) float64     -- best/combined phase in radians
#       extra : dict (e.g. per-set -LL, FOM means)
#   negll_only(data, fc_total, scale, mlpar, use_phase_prior=False,
#              set_mask=None) -> float
#   r_factors(data, fc_total, scale) -> (r_work, r_free, r_all)
#
# gradient.py
#   fc_and_atomic_gradients(data, xyz, b_iso, occ, G, fc_atom_factor,
#                           chunk=48, want_fc=True)
#       -> (fc, grad_xyz (M,3), grad_b (M,))
#
# geometry.py
#   class Restraints:
#       __init__(self, data, w_bond=1.0, w_angle=1.0,
#                w_b=1.0, w_torsion=1.0, w_plane=1.0, w_chir=1.0,
#                use_monlib=True)
#       calc(self, xyz, b_iso) -> (f, grad_xyz, grad_b, stats:dict)
#       n_bonds / n_angles properties
#
# maps.py
#   write_output_mtz(data, path, fc_total, scale, mlpar, m, phase,
#                    xyz, b_iso, extra_columns=True,
#                    output_scale="absolute", fc_atoms=None) -> None
#   map_coefficients(data, fc_total, scale, mlpar, m, phase,
#                    output_scale="absolute", fc_atoms=None) -> dict[str, array]
#   OUTPUT SCALE: "absolute" (default) matches REFMAC5 -- F/SIGF and every
#     generated amplitude are on the electron scale of Fc, i.e. fo/k_total
#     and k_fc=1.  "observed" keeps the legacy input scale, k_fc=k_total.
#     FC/PHIC is atom-only when fc_atoms is supplied.
#
# refine.py / cli.py / selftest.py : written by the lead.
