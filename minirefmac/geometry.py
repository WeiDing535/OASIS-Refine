"""
minirefmac.geometry -- stereochemical restraints for maximum-likelihood
refinement (PROLSQ / REFMAC5 style).

Interface (declared at the bottom of core.py)::

    Restraints(data, w_bond=1.0, w_angle=1.0, w_b=1.0,
               w_torsion=1.0, w_plane=1.0, w_chir=1.0, use_monlib=True,
               w_vdw=1.0, sym_vdw=True, vdw_cutoff=5.0, peptide_plane=True)
    calc(xyz, b_iso) -> (f, grad_xyz, grad_b, stats: dict)
    n_bonds / n_angles : properties

COORDINATE CONVENTION (measured on gemmi 0.7.5 with a TRICLINIC cell)
---------------------------------------------------------------------
    M = np.asarray(cell.orth.mat.tolist())       # fractional -> Cartesian
    cell.orthogonalize(Fractional(f)) == M @ f == f @ M.T   (row form)

    Xcart      = Xfrac @ M.T
    grad_frac  = grad_cart @ M          (the contract's A^T grad_cart)

data.xyz holds FRACTIONAL coordinates.  All distances / chiral volumes are
evaluated in orthogonal Angstrom space; calc() returns the gradient WITH
RESPECT TO FRACTIONAL COORDINATES, obtained by the chain rule above.  The
finite-difference test in tests/test_geometry.py perturbs fractional coords
and compares f(+h)-f(-h) with grad_xyz.

IMPLEMENTATION NOTES
--------------------
* Bond / angle restraints come from the bundled monomer dictionary in
  ``minirefmac/data/monlib_bundle.zip`` (built once by
  ``tools/build_monlib_bundle.py``).  At runtime ``monlib.py`` rebuilds the
  same attribute surface (bonds, angles, torsions, chirs, planes) from
  compact per-residue JSON, so no external files and no CIF parsing are
  needed.
* Dictionary angles use explicit three-atom angles in radians. Unknown
  residues retain the initial-distance fallback.
* Planarity is a TRUE least-squares plane:  f = w lambda_min(C) with
  C = sum u_i u_i^T and u_i = X_i - mean(X).  lambda_min is rotation invariant
  (so the group may rotate freely) and the envelope theorem gives the exact,
  vectorised analytic gradient  2 w (n.u_i) n.  It actually flattens a
  puckered ring, unlike the old in-plane 1-3/1-4 distance hold.  Chirality
  uses a true oriented-volume restraint with an analytic vectorised gradient.
* Torsions use the rigorous periodic dihedral
  E = 2 w (1 - cos(n (tau - tau0))) / n^2,  w = w_torsion / sigma_rad^2,
  with a fully vectorised analytic gradient (verified by finite difference).
  Dictionary torsions are selected the way REFMAC5 does: only the rotamer
  "chi*" torsions with esd > 0 are harmonic restraints; esd = 0 marks a
  dictionary CONSTRAINT (handled by the planar-group term), and the
  "sp2_sp3_*" (O-C-CA-N backbone pseudo-torsion) / "sp3_sp3_*" (proline
  ring pucker) labels are not harmonic torsions at all.  chi* torsions whose
  first and fourth atoms are covalently connected through a short path that
  avoids the middle two atoms are ring torsions (PRO chi1 / chi2) and are
  skipped as well -- exactly the torsions REFMAC omits.  Adding back the
  trans-peptide omega (CA-C-N-CA, 180 deg, period 1, esd 5 deg) reproduces
  REFMAC's "Torsion angles, period 1/2/3/6" counts on all three test cases:
      3o2e 35/0/0/0,  7enm 80/2/123/37,  5hhk 274/19/418/110.
* Residues / atom pairs not covered by the dictionary use "initial-geometry
  preservation" (jelly-body) restraints with the dictionary median sigma, so
  restraints are never missing (UNK / MSE / non-standard residues).
* Intra-chain peptide C--N links come from the ``link_TRANS`` definition,
  also bundled (``LINK_TRANS.json`` inside the zip): the C--N bond, the three
  non-hydrogen cross-peptide bond angles (CA-C-N / O-C-N / C-N-CA), the
  trans omega torsion and the peptide-unit plane.
* Explicit same-ASU disulfides and unambiguous close cysteine SG pairs are
  restrained as bonds and excluded from non-bonded repulsion.
* Non-bonded VDW repulsion is a one-sided harmonic on ASU-internal pairs AND
  symmetry mates / neighbouring unit cells:  f = w sum max(0, d_min-d)^2 with
  d_min = r_vdw(i)+r_vdw(j)-0.3 A and sigma = 0.2 A (REFMAC's Av(Sigma)).
  Covalent neighbours up to 1-4 are excluded.  Each pair stores the 3x3
  Jacobian of the symmetry copy w.r.t. the source atom's FRACTIONAL coords,
  so the analytic gradient is exact even for symmetry mates (verified by FD).
* All restraint arrays are built ONCE in __init__; calc() is fully
  vectorised (numpy only, no Python loop over restraints) and uses
  np.bincount rather than np.add.at for speed (M ~ 670, restraints ~ 10^4).
"""

import os
import warnings
from collections import Counter, defaultdict
from itertools import combinations, product
import gemmi
import numpy as np
from scipy.spatial import cKDTree

try:
    from . import monlib
except (ImportError, ValueError):          # flat-module fallback
    import monlib
_PEPTIDE_CN = 1.337              # fallbacks if mon_lib_list.cif is unreadable
_PEPTIDE_CN_SIGMA = 0.011
# link_TRANS cross-peptide angles: (comp1, atom1, comp2, atom2, comp3, atom3,
#                                   value_deg, esd_deg).  comp '1'/'2' is the
# first / second residue of the link; H-containing entries are skipped.
_PEPTIDE_LINK_ANGLES = [
    ("1", "CA", "1", "C",  "2", "N",  115.917, 1.50),
    ("1", "O",  "1", "C",  "2", "N",  123.469, 1.50),
    ("1", "C",  "2", "N",  "2", "CA", 122.095, 1.76),
    ("1", "C",  "2", "N",  "2", "H",  119.176, 1.83),
]
_PEPTIDE_LINK_PLANE = ("1", "CA", "1", "C", "1", "O", "2", "N")
_PEPTIDE_PLANE_ESD = 0.020
_LINK_DEF_CACHE = {}
_BACKBONE = frozenset(("N", "CA", "C", "O", "OXT"))
_SIGMA_VOL = 0.5           # A^3, chiral-volume restraint sigma (dictionary
                           # gives only the sign, so this is a fixed choice)
_SIGMA_14 = 0.20           # A, (legacy 1-4 distance sigma, no longer used)
_SIGMA_FLOOR = 1e-3        # A, minimum distance sigma
_OMEGA_ESD = 5.0           # deg, trans-peptide omega torsion esd
# REFMAC5 applies harmonic torsion restraints only to side-chain rotamer
# torsions ("chi*" labels in the bundled dictionary).  "sp2_sp3_*" is
# O-C-CA-N backbone pseudo-torsion and "sp3_sp3_*" a proline ring-pucker
# pseudo-torsion; those are used for VDW-through-torsion, not as harmonic
# dihedral restraints.  See the module docstring for the count comparison.
_TORSION_LABEL_PREFIX = "chi"
_TORSION_RING_MAXBOND = 3  # max covalent path length a1 -> a4 (ring test)

# ---- non-bonded (VDW) repulsion ------------------------------------------
# REFMAC5 lists "VDW repulsions: refined_atoms" (and a separate symmetry
# class) with Av(Sigma) = 0.200 A.  We use the same one-sided harmonic:
#     f = w * sum max(0, d_min - d)^2 ,  d_min = r_i + r_j - slack
# so atoms further apart than d_min cost nothing.  Topological neighbours up
# to 1-4 are excluded (they are already covered by bond/angle/torsion terms);
# REFMAC treats 1-4 non-bonded contacts as a separate, weaker class.
_VDW_RADII = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
              "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98,
              "SE": 1.90, "FE": 1.80, "MG": 1.73, "ZN": 1.39, "CA": 2.31}
_VDW_DEFAULT_R = 1.70
_VDW_SLACK = 0.30          # A; C...C -> d_min 3.10 A
_VDW_SIGMA = 0.20          # A, matches REFMAC's Av(Sigma)
_VDW_CUTOFF = 5.0          # A, candidate-pair search radius (generous: a
                           # restraint can only *appear* if two atoms move
                           # close; max_shift_a is 0.5 A per cycle)

# distance-restraint kinds (used for stats grouping)
_BOND = 0
_ANGLE = 1
_PLANE = 2
_TORSION = 3
_N_KIND = 4


def _sigma_b(n1, n2):
    """B-factor restraint sigma: 1.5 (backbone), 3.0 (sidechain), else 2.0."""
    b1 = n1 in _BACKBONE
    b2 = n2 in _BACKBONE
    if b1 and b2:
        return 1.5
    if (not b1) and (not b2):
        return 3.0
    return 2.0


def _ring_torsion(adj, quad, maxbond=_TORSION_RING_MAXBOND):
    """True if a1 and a4 are covalently linked around a short covalent ring.

    adj maps an atom name to its bonded neighbours (within one residue).
    The search excludes the two middle atoms a2/a3, so an ordinary acyclic
    side-chain chi torsion (e.g. LYS N-CA-CB-CG) returns False while a
    proline ring chi (N-CA-CB-CG with N-CD-CG closing the ring) returns True.
    REFMAC does not apply harmonic chi restraints to ring torsions.
    """
    a1, a2, a3, a4 = quad
    seen = {a1, a2, a3}
    frontier = {a1}
    for _ in range(maxbond):
        nxt = set()
        for u in frontier:
            nxt.update(adj.get(u, ()))
        nxt -= seen
        if a4 in nxt:
            return True
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    return False


def _load_link_defs():
    """Peptide ``link_TRANS`` restraints, from the bundled dictionary (cached).

    Returns dict(bond=(d, esd), angles=[...], omega=(val, esd, period),
    plane=[(comp, atom), ...], plane_esd=float).  If the bundle itself is
    unavailable, the module constants above are used.
    """
    key = "bundled"
    if key in _LINK_DEF_CACHE:
        return _LINK_DEF_CACHE[key]
    out = dict(bond=(_PEPTIDE_CN, _PEPTIDE_CN_SIGMA),
               angles=list(_PEPTIDE_LINK_ANGLES),
               omega=(180.0, _OMEGA_ESD, 1),
               plane=[(_PEPTIDE_LINK_PLANE[i], _PEPTIDE_LINK_PLANE[i + 1])
                      for i in range(0, len(_PEPTIDE_LINK_PLANE), 2)],
               plane_esd=_PEPTIDE_PLANE_ESD)
    raw = monlib.bundled().link_trans()
    try:
        if raw.get("bond"):
            out["bond"] = (float(raw["bond"][0]), float(raw["bond"][1]))
        if raw.get("angles"):
            out["angles"] = [tuple(r) for r in raw["angles"]]
        for r in raw.get("torsions", ()):
            if r[0] == "omega":
                # period 0 in the library means "not periodic", i.e. a single
                # well at 180 deg; our periodic form needs n >= 1.
                out["omega"] = (float(r[1]), float(r[2]),
                                max(1, int(float(r[3]))))
        pl = [(r[1], r[2]) for r in raw.get("planes", ())
              if r[0] == "plan-1"]
        if pl:
            out["plane"] = pl
            out["plane_esd"] = float(next(r[3] for r in raw["planes"]
                                          if r[0] == "plan-1"))
    except (TypeError, ValueError, IndexError, KeyError, StopIteration):
        pass
    _LINK_DEF_CACHE[key] = out
    return out


def _accumulate(out, idx, val):
    """out[idx] += val  for a (R,3) val array, via fast bincount per column."""
    m = out.shape[0]
    for k in range(3):
        out[:, k] += np.bincount(idx, weights=val[:, k], minlength=m)


def _explicit_disulfides(data, sulfur):
    """Resolve PDB/mmCIF disulfide declarations to the selected atom rows."""
    st = getattr(data, "st", None)
    connections = [] if st is None else [
        c for c in st.connections if c.type == gemmi.ConnectionType.Disulf]
    if not connections:
        return [], set()

    selected = dict(zip(getattr(data, "_atom_sel", range(data.n_atoms)),
                        range(data.n_atoms)))
    atoms = defaultdict(list)
    sulfur = set(sulfur)
    original_atoms = ((chain, residue, atom) for model in st for chain in model
                      for residue in chain for atom in residue)
    for index, (chain, residue, atom) in enumerate(original_atoms):
        row = selected.get(index)
        if row in sulfur:
            key = (chain.name, str(residue.seqid), residue.name, atom.name)
            atoms[key].append((atom.altloc, row))

    occupied = set()
    pairs = set()
    for connection in connections:
        rows = []
        for address in (connection.partner1, connection.partner2):
            key = (address.chain_name, str(address.res_id.seqid),
                   address.res_id.name, address.atom_name)
            matches = [row for alt, row in atoms[key]
                       if address.altloc in ("", " ", "\x00") or alt == address.altloc]
            occupied.update(matches)
            if len(matches) == 1:
                rows.append(matches[0])
        if len(set(rows)) != 2 or connection.asu == gemmi.Asu.Different:
            warnings.warn("Disulfide %s is unresolved or crosses symmetry; "
                          "no direct bond restraint added" % connection.name)
            continue
        pairs.add(tuple(sorted(rows)))
    degree = Counter(row for pair in pairs for row in pair)
    if any(n > 1 for n in degree.values()):
        warnings.warn("Conflicting explicit disulfides have been skipped")
    return [pair for pair in sorted(pairs) if all(degree[row] == 1 for row in pair)], occupied


class Restraints:
    """Stereochemical restraint target and its analytic gradients."""

    def __init__(self, data, w_bond=1.0, w_angle=1.0,
                 w_b=1.0, w_torsion=1.0, w_plane=1.0, w_chir=1.0,
                 use_monlib=True, w_vdw=1.0, sym_vdw=True,
                 vdw_cutoff=_VDW_CUTOFF, peptide_plane=True, restraint_cifs=()):
        # Restraint parameters always come from the dictionary bundled in
        # ``minirefmac/data/monlib_bundle.zip``.
        self.monlib_path = monlib.BUNDLE_PATH
        self.use_monlib = bool(use_monlib)
        self.weights = dict(w_bond=float(w_bond), w_angle=float(w_angle),
                            w_b=float(w_b), w_torsion=float(w_torsion),
                            w_plane=float(w_plane), w_chir=float(w_chir),
                            w_vdw=float(w_vdw))
        w_bond = float(w_bond)
        w_angle = float(w_angle)
        w_b = float(w_b)
        w_torsion = float(w_torsion)
        w_plane = float(w_plane)
        w_chir = float(w_chir)
        self._w_vdw = float(w_vdw)
        self._sym_vdw = bool(sym_vdw)
        self._vdw_cutoff = float(vdw_cutoff)
        self._peptide_plane = bool(peptide_plane)

        # ---- fractional -> Cartesian matrix -----------------------------
        M = np.asarray(data.cell.orth.mat.tolist(), dtype=np.float64)  # (3,3)
        self._orth = M
        x0 = np.asarray(data.xyz, dtype=np.float64) @ M.T              # (M,3) A
        self._x0 = x0
        n_atoms = x0.shape[0]

        names = list(data.atom_names)
        res_names = list(data.res_names)
        chains = list(data.chain_ids)
        seq = np.asarray(data.res_seq)

        # ---- dictionary (only residue kinds actually present) -----------
        kinds = sorted(set(res_names))
        lib = self._load_library(kinds)
        external = monlib.load_component_cifs(restraint_cifs)
        used_external = sorted(set(external).intersection(kinds))
        lib.update({name: external[name] for name in used_external})
        linkdef = _load_link_defs()
        n_kinds_hit = len(lib)

        # dictionary bond target table (resname, unordered atom pair) -> d0
        bond_d0 = {}
        esd_pool = []
        for rn, rt in lib.items():
            for b in rt.bonds:
                key = (rn, frozenset((b.id1.atom, b.id2.atom)))
                try:
                    bond_d0[key] = max(float(b.value), 0.5)
                    e = float(b.esd)
                    if e > 0:
                        esd_pool.append(e)
                except (TypeError, ValueError):
                    pass
        med_sigma = float(np.median(esd_pool)) if esd_pool else 0.02
        med_sigma = max(med_sigma, _SIGMA_FLOOR)
        self._med_sigma = med_sigma

        # ---- residue blocks (consecutive atoms with same chain/seq/name) --
        blocks = []
        residue_rows = getattr(data, "_residue_rows", seq)
        start = 0
        for i in range(1, n_atoms + 1):
            if (i == n_atoms or chains[i] != chains[start]
                    or residue_rows[i] != residue_rows[start]
                    or res_names[i] != res_names[start]):
                blocks.append((start, i))
                start = i

        # ---- restraint builders (plain lists, converted to arrays later) --
        di, dj, d0, dsig, dw, dkind, dfrom = [], [], [], [], [], [], []
        bi, bj, bw = [], [], []
        ci1, ci2, ci3, cic, cv, cw = [], [], [], [], [], []
        plane_groups = []
        t1, t2, t3, t4, tval, tesd, tper = [], [], [], [], [], [], []
        dist_pairs = set()
        angles = []
        residues_hit = set()
        n_bond_from_dict = 0

        angle_keys, torsion_keys, plane_keys, chiral_keys = set(), set(), set(), set()

        def add_angle(ids, target, sigma):
            key = (*ids, float(target), max(float(sigma), 0.1))
            if key in angle_keys:
                return False
            angle_keys.add(key)
            angles.append(key)
            return True

        def add_plane(ids, weight):
            key = (tuple(ids), weight)
            if key in plane_keys:
                return False
            plane_keys.add(key)
            plane_groups.append((ids, weight))
            return True

        def add_torsion(a, b, c, d, v_deg, e_deg, per):
            """Dictionary torsion, or the peptide omega dihedral."""
            key = (a, b, c, d, v_deg, e_deg, per)
            if len({a, b, c, d}) < 4 or key in torsion_keys:
                return
            torsion_keys.add(key)
            t1.append(a); t2.append(b); t3.append(c); t4.append(d)
            tval.append(np.deg2rad(float(v_deg)))
            tesd.append(np.deg2rad(max(float(e_deg), 1.0)))
            tper.append(max(1, int(per)))

        def add_dist(i, j, target, sigma, weight, kind, from_dict):
            if i == j:
                return False
            key = (i, j) if i < j else (j, i)
            if key in dist_pairs:
                return False
            sigma = max(float(sigma), _SIGMA_FLOOR)
            dist_pairs.add(key)
            di.append(i)
            dj.append(j)
            d0.append(float(target))
            dsig.append(sigma)
            dw.append(float(weight))
            dkind.append(kind)
            dfrom.append(bool(from_dict))
            if kind == _BOND:
                bi.append(i)
                bj.append(j)
                sb = _sigma_b(names[i], names[j])
                bw.append(w_b / (sb * sb))
            return True

        # Shared atoms belong to every map; each alternate label gets its
        # own map. Restraints containing only shared atoms are added once.
        block_maps = []
        self._altloc_conflicts = set()
        all_altlocs = getattr(data, "_all_altlocs", False)
        labels = ([alt if alt not in ("", " ", "\x00") else "" for alt in data.altloc]
                  if all_altlocs else [""] * n_atoms)
        peptide_maps = []
        peptide_names = {atom for entry in linkdef["angles"] for atom in entry[1:6:2]}
        peptide_names.update(("N", "CA", "C", "O"))
        for s, e in blocks:
            shared = {}
            alternatives = defaultdict(dict)
            for i in range(s, e):
                alt = labels[i]
                if not alt:
                    shared.setdefault(names[i], i)
                else:
                    alternatives[alt].setdefault(names[i], i)
            block_maps.append({alt: shared | mapping
                               for alt, mapping in alternatives.items()} or {"": shared})
            backbone = {name: i for name, i in shared.items() if name in peptide_names}
            backbone_alts = {alt: {name: i for name, i in mapping.items() if name in peptide_names}
                             for alt, mapping in alternatives.items()}
            peptide_maps.append({(alt if mapping else ""): backbone | mapping
                                 for alt, mapping in backbone_alts.items()} or {"": backbone})
            for left, right in combinations(alternatives.values(), 2):
                self._altloc_conflicts.update(tuple(sorted(pair))
                                              for pair in product(left.values(), right.values()))

        torsion_adj_cache = {}
        conformers = ((k, *blocks[k], mapping) for k, maps in enumerate(block_maps)
                      for mapping in maps.values())
        for block_index, s, e, name2idx in conformers:
            rn = res_names[s]
            rt = lib.get(rn)
            n_dict_bonds = 0

            if rt is not None:
                # -- bonds --
                for b in rt.bonds:
                    i = name2idx.get(b.id1.atom)
                    j = name2idx.get(b.id2.atom)
                    if i is None or j is None:
                        continue            # H / dictionary-only atom
                    try:
                        v = max(float(b.value), 0.5)
                        sg = max(float(b.esd), _SIGMA_FLOOR)
                    except (TypeError, ValueError):
                        continue
                    n_dict_bonds += 1
                    if add_dist(i, j, v, sg, w_bond / (sg * sg),
                                _BOND, True):
                        n_bond_from_dict += 1

                for a in rt.angles:
                    ids = [name2idx.get(atom.atom) for atom in (a.id1, a.id2, a.id3)]
                    if None not in ids and len(set(ids)) == 3:
                        add_angle(ids, a.value, a.esd)

                # -- planarity: TRUE least-squares plane --------------------
                # Each group contributes  f = w * lambda_min(C)  where
                # C = sum_i u_i u_i^T, u_i = X_i - mean(X) and lambda_min is
                # the out-of-plane variance.  lambda_min is rotation invariant,
                # and by the envelope theorem its gradient is EXACTLY
                #   d f / d X_i = 2 w (n . u_i) n
                # with n the smallest-eigenvalue unit eigenvector (the n(x)
                # dependence drops out at the Rayleigh stationary point).
                # A group needs >= 4 atoms (3 points are always coplanar).
                for pl in rt.planes:
                    ids = [name2idx.get(a.atom) for a in pl.ids]
                    ids = [i for i in ids if i is not None]
                    if len(ids) < 4:
                        continue
                    try:
                        esd = max(float(pl.esd), 0.02)
                    except (TypeError, ValueError):
                        esd = 0.02
                    add_plane(ids, w_plane / (esd * esd))

                # -- chirality: true oriented-volume analytic gradient ----
                for ch in rt.chirs:
                    a = name2idx.get(ch.id1.atom)
                    b = name2idx.get(ch.id2.atom)
                    c = name2idx.get(ch.id3.atom)
                    ctr = name2idx.get(ch.id_ctr.atom)
                    if any(v is None for v in (a, b, c, ctr)):
                        continue
                    sgn = self._chir_sign(ch.sign)
                    key = (a, b, c, ctr, sgn)
                    if sgn == 0 or key in chiral_keys:
                        continue                # 'Both': no restraint
                    chiral_keys.add(key)
                    v1 = x0[a] - x0[ctr]
                    v2 = x0[b] - x0[ctr]
                    v3 = x0[c] - x0[ctr]
                    vol0 = float(np.dot(v1, np.cross(v2, v3)))
                    vtarget = sgn * abs(vol0)   # keep magnitude, fix sign
                    ci1.append(a)
                    ci2.append(b)
                    ci3.append(c)
                    cic.append(ctr)
                    cv.append(vtarget)
                    cw.append(w_chir / (_SIGMA_VOL * _SIGMA_VOL))

                # -- torsion: TRUE dihedral with periodicity ---------------
                # REFMAC-compatible selection (see module docstring):
                #   * only "chi*" rotamer labels,
                #   * esd == 0 is a dictionary CONSTRAINT -> planar groups,
                #   * skip torsions closed by a covalent ring (PRO chi1/chi2).
                for tt in rt.torsions:
                    label = str(tt.label)
                    if rn not in external and not label.startswith(_TORSION_LABEL_PREFIX):
                        continue
                    quad = tuple(getattr(tt, k).atom
                                 for k in ("id1", "id2", "id3", "id4"))
                    ids = [name2idx.get(a) for a in quad]
                    if any(v is None for v in ids):
                        continue            # H / dictionary-only atom
                    try:
                        val = float(tt.value)
                        esd = float(tt.esd)
                        per = int(tt.period)
                    except (TypeError, ValueError):
                        continue
                    # esd == 0 marks a dictionary CONSTRAINT (e.g. PHE CONST_1,
                    # the aromatic-ring torsions).  REFMAC does not turn those
                    # into harmonic restraints -- it uses the planar-group term
                    # instead, which we now implement properly.  Clamping a 0
                    # esd to 1 deg would give a weight ~3300 and swamp the
                    # whole geometry target.
                    if esd <= 0.0:
                        continue
                    if rn not in torsion_adj_cache:
                        adj = {}
                        for bb in rt.bonds:
                            u, v = bb.id1.atom, bb.id2.atom
                            adj.setdefault(u, set()).add(v)
                            adj.setdefault(v, set()).add(u)
                        torsion_adj_cache[rn] = adj
                    if rn not in external and _ring_torsion(torsion_adj_cache[rn], quad):
                        continue
                    add_torsion(*ids, val, esd, per)

            # -- FALLBACK: initial-geometry preservation ------------------
            # For residues with no usable dictionary bonds (unknown / MSE /
            # non-standard / H-only dictionary), hold the initial 1-2 and
            # 1-3 distances with the dictionary median sigma.
            if n_dict_bonds == 0:
                med13 = 0.04
                w13 = w_angle / (med13 * med13)
                residue_atoms = sorted(set(name2idx.values())) if all_altlocs else range(s, e)
                for p, q in combinations(residue_atoms, 2):
                    if (p, q) in dist_pairs:
                        continue
                    dv = float(np.linalg.norm(x0[p] - x0[q]))
                    if 0.9 < dv < 2.0:
                        add_dist(p, q, dv, med_sigma,
                                 w_bond / (med_sigma * med_sigma),
                                 _BOND, False)
                    elif 2.0 <= dv < 3.3:
                        add_dist(p, q, dv, med13, w13, _ANGLE, False)
            else:
                residues_hit.add(block_index)
        n_res_hit = len(residues_hit)

        # ---- peptide links: C-N bond, 3 cross-link angles, omega, plane --
        cn_d0, cn_sigma = linkdef["bond"]
        om_val, om_esd, om_per = linkdef["omega"]
        n_peptide = 0
        n_cross_angle = 0
        n_link_plane = 0
        for k in range(len(blocks) - 1):
            s0, e0 = blocks[k]
            s1, e1 = blocks[k + 1]
            if chains[s0] != chains[s1]:
                continue
            for (alt0, m0), (alt1, m1) in product(peptide_maps[k].items(),
                                                    peptide_maps[k + 1].items()):
                if alt0 and alt1 and alt0 != alt1:
                    continue
                ci = m0.get("C")
                ni = m1.get("N")
                if ci is None or ni is None:
                    continue
                dv = float(np.linalg.norm(x0[ci] - x0[ni]))
                if dv > 1.8:                       # chain break, not a peptide bond
                    continue
                lookup = {"1": m0, "2": m1}

                n_peptide += add_dist(ci, ni, cn_d0, cn_sigma,
                                      w_bond / (cn_sigma ** 2), _BOND, False)

                for c1, a1, c2, a2, c3, a3, th_deg, es_deg in linkdef["angles"]:
                    ids = [lookup[c].get(a) for c, a in ((c1, a1), (c2, a2), (c3, a3))]
                    if None not in ids and len(set(ids)) == 3:
                        n_cross_angle += add_angle(ids, th_deg, es_deg)

                # trans-peptide omega: CA_i - C_i - N_{i+1} - CA_{i+1} -> 180 deg,
                # period 1, esd 5 deg (REFMAC's "period 1" torsion class: exactly
                # one restraint per peptide link).
                ca0 = m0.get("CA")
                ca1 = m1.get("CA")
                if ca0 is not None and ca1 is not None:
                    add_torsion(ca0, ci, ni, ca1, om_val, om_esd, om_per)

                # peptide-unit plane (CA_i, C_i, O_i, N_{i+1}), sigma 0.020 A.
                # REFMAC lists one such planar group per peptide link; without it
                # our planar-group count was ~1/3 of REFMAC's.
                if self._peptide_plane:
                    ids = []
                    for comp, atom in linkdef["plane"]:
                        idx = lookup.get(comp, {}).get(atom)
                        if idx is None:
                            ids = []
                            break
                        ids.append(idx)
                    if len(set(ids)) >= 4:
                        esd = max(float(linkdef["plane_esd"]), 0.005)
                        n_link_plane += add_plane(ids, w_plane / (esd * esd))

        sulfur = [i for i, name in enumerate(names) if name == "SG" and res_names[i] == "CYS"]
        explicit, occupied = _explicit_disulfides(data, sulfur)
        sulfur = [i for i in sulfur if i not in occupied]
        inferred = []
        if sulfur:
            candidates = sorted((i, j) for i, j in cKDTree(x0[sulfur]).query_pairs(2.5)
                                if (sulfur[i], sulfur[j]) not in self._altloc_conflicts
                                and (not labels[sulfur[i]] or not labels[sulfur[j]]
                                     or labels[sulfur[i]] == labels[sulfur[j]]))
            degree = Counter(row for pair in candidates for row in pair)
            inferred = [(sulfur[i], sulfur[j]) for i, j in candidates
                        if degree[i] == degree[j] == 1]
        disulfide_sigma = 0.02
        n_disulfide = sum(add_dist(i, j, 2.031, disulfide_sigma,
                                   w_bond / disulfide_sigma ** 2, _BOND, False)
                          for i, j in explicit + inferred)

        angle_table = np.asarray(angles, dtype=float).reshape(-1, 5)
        self._ai, self._aj, self._ak = angle_table[:, :3].astype(np.int64).T
        self._a0, self._asig = np.deg2rad(angle_table[:, 3:]).T
        self._aw = w_angle / self._asig ** 2

        # ---- freeze into arrays -----------------------------------------
        self._di = np.asarray(di, dtype=np.int64)
        self._dj = np.asarray(dj, dtype=np.int64)
        self._d0 = np.asarray(d0, dtype=np.float64)
        self._dsig = np.asarray(dsig, dtype=np.float64)
        self._dw = np.asarray(dw, dtype=np.float64)
        self._dkind = np.asarray(dkind, dtype=np.int8)
        self._dfrom = np.asarray(dfrom, dtype=bool)
        self._bi = np.asarray(bi, dtype=np.int64)
        self._bj = np.asarray(bj, dtype=np.int64)
        self._bw = np.asarray(bw, dtype=np.float64)
        self._ci1 = np.asarray(ci1, dtype=np.int64)
        self._ci2 = np.asarray(ci2, dtype=np.int64)
        self._ci3 = np.asarray(ci3, dtype=np.int64)
        self._cic = np.asarray(cic, dtype=np.int64)
        self._cv = np.asarray(cv, dtype=np.float64)
        self._cw = np.asarray(cw, dtype=np.float64)

        # ---- torsion arrays ---------------------------------------------
        self._t1 = np.asarray(t1, dtype=np.int64)
        self._t2 = np.asarray(t2, dtype=np.int64)
        self._t3 = np.asarray(t3, dtype=np.int64)
        self._t4 = np.asarray(t4, dtype=np.int64)
        self._tval = np.asarray(tval, dtype=np.float64)     # target, rad
        self._tesd = np.asarray(tesd, dtype=np.float64)     # sigma, rad
        self._tper = np.asarray(tper, dtype=np.int64)
        self._tw = w_torsion / np.maximum(self._tesd, 1e-4) ** 2

        # ---- plane groups -> padded (P, L, 3) arrays --------------------
        if plane_groups:
            lmax = max(len(ids) for ids, _ in plane_groups)
            self._pl_idx = np.zeros((len(plane_groups), lmax), dtype=np.int64)
            self._pl_mask = np.zeros((len(plane_groups), lmax), dtype=bool)
            self._pl_w = np.zeros(len(plane_groups), dtype=np.float64)
            for p, (ids, wp) in enumerate(plane_groups):
                self._pl_idx[p, :len(ids)] = ids
                self._pl_mask[p, :len(ids)] = True
                self._pl_w[p] = wp
        else:
            self._pl_idx = np.zeros((0, 4), dtype=np.int64)
            self._pl_mask = np.zeros((0, 4), dtype=bool)
            self._pl_w = np.zeros(0, dtype=np.float64)

        # precomputed boolean masks for stats
        self._mask_bond = (self._dkind == _BOND)
        self._mask_angle = (self._dkind == _ANGLE)

        # ---- non-bonded (VDW) contact pairs -----------------------------
        self._build_vdw(data, x0)

        # ---- coverage / info for reporting ------------------------------
        n_bond = int(self._mask_bond.sum())
        self.info = dict(
            monlib_path=self.monlib_path,
            use_monlib=self.use_monlib,
            n_res=len(blocks),
            n_res_hit=n_res_hit,
            n_kinds=len(kinds),
            n_kinds_hit=n_kinds_hit,
            n_bond=n_bond,
            n_angle=self.n_angles,
            n_angle_distance=int(self._mask_angle.sum()),
            n_plane=int(self._pl_mask.sum()),   # atoms in planar groups
            n_plane_groups=int(self._pl_w.size),
            n_torsion=int(self._tw.size),
            n_chir=len(ci1),
            n_bfac=len(bi),
            n_peptide=n_peptide,
            n_disulfide=n_disulfide,
            n_disulfide_explicit=len(explicit),
            external_components=used_external,
            n_cross_angle=n_cross_angle,
            n_link_plane=n_link_plane,
            n_bond_from_dict=n_bond_from_dict,
            med_sigma=med_sigma,
            hit_rate_res=(n_res_hit / len(blocks) if blocks else 0.0),
            hit_rate_kind=(n_kinds_hit / len(kinds) if kinds else 0.0),
            hit_rate_bond=(n_bond_from_dict / n_bond if n_bond else 0.0),
            n_vdw=int(self._vi.size),
            n_vdw_sym=int(getattr(self, "_n_vdw_sym", 0)),
            n_restraints=(len(di) + self.n_angles + len(ci1) + len(bi) + int(self._vi.size)
                          + int(self._pl_mask.sum()) + int(self._tw.size)),
        )

    # ------------------------------------------------------------------
    def _build_vdw(self, data, x0):
        """One-sided harmonic non-bonded repulsion on ASU + symmetry mates.

        A contact is (ASU atom i, a *copy* of ASU atom j) where the copy is
            Xcopy = (R_s x_j + t_s + n) @ M.T = x_j @ (R_s.T @ M.T) + (t_s+n) @ M.T
        with s a space-group operation (identity included) and n a lattice
        translation in {-1,0,1}^3.  (s=identity, n=0) is the ASU itself and is
        enumerated separately.  Only i<j is kept for cross-image pairs: the
        inverse viewpoint (j, i, s^-1, -n) is the *same* physical contact, so
        this de-duplicates it.  Pairs whose atoms are <=3 covalent bonds apart
        in the ASU are dropped (bond/angle/torsion territory).

        Stored per pair: the base atom i, the copy source atom j, and the 3x3
        Jacobian B[frac, cart] of Xcopy w.r.t. FRACTIONAL x_j.  That makes the
        chain rule exact and fully vectorised:
            d f / d x_i (frac) = (-coef) @ M
            d f / d x_j (frac) = coef @ B.T
        with coef = 2 w (d_min-d) (X_i - Xcopy) / d.
        """
        M = self._orth
        xfrac = np.asarray(data.xyz, dtype=np.float64)
        m = xfrac.shape[0]
        cut = self._vdw_cutoff
        B_id = M.T                      # identity op: X = xfrac @ M.T
        zero = np.zeros(3, dtype=np.float64)

        # topological (covalent) neighbours up to 1-4
        adj = [[] for _ in range(m)]
        for i, j in zip(self._di[self._mask_bond], self._dj[self._mask_bond]):
            adj[int(i)].append(int(j))
            adj[int(j)].append(int(i))
        topo = set()
        for a in range(m):
            l1 = set(adj[a])
            l2 = set()
            for b in l1:
                l2.update(adj[b])
            l3 = set()
            for c in l2:
                l3.update(adj[c])
            for b in (l1 | l2 | l3):
                if b != a:
                    topo.add((min(a, b), max(a, b)))

        vi, vj, vB, vc = [], [], [], []

        # (1) ASU-internal pairs (both atoms in the base cell)
        tree_au = cKDTree(x0)
        for i, j in sorted(tree_au.query_pairs(cut)):
            if (i, j) in topo or (i, j) in self._altloc_conflicts:
                continue
            vi.append(i); vj.append(j); vB.append(B_id); vc.append(zero)
        n_au = len(vi)

        # (2) cross-image pairs: symmetry mates and neighbouring unit cells
        if self._sym_vdw:
            try:
                from .sfcalc import _sym_ops     # package form
            except (ImportError, ValueError):   # flat-module fallback
                from sfcalc import _sym_ops
            ops = _sym_ops(data)
            ops = [(np.asarray(R, dtype=np.float64),
                    np.asarray(t, dtype=np.float64)) for (R, t) in ops]
            id_idx = None
            for s, (R, t) in enumerate(ops):
                if np.allclose(R, np.eye(3), atol=1e-8) and np.allclose(t, 0, atol=1e-8):
                    id_idx = s
                    break

            groups = []
            for s, (R, t) in enumerate(ops):
                B = R.T @ M.T
                for n in product((-1, 0, 1), repeat=3):
                    if s == id_idx and n == (0, 0, 0):
                        continue
                    c = (t + np.asarray(n, dtype=np.float64)) @ M.T
                    groups.append((s, B, c, xfrac @ B + c))
            if groups:
                allc = np.concatenate([g[3] for g in groups], axis=0)
                tree = cKDTree(allc)
                hits = tree.query_ball_point(x0, r=cut)
                for i, row in enumerate(hits):
                    for k in row:
                        gidx = k // m
                        j = k % m
                        if j <= i:               # de-duplicate inverse contact
                            continue
                        s, B, c, _ = groups[gidx]
                        if s == id_idx and (i, j) in topo:
                            continue
                        vi.append(i); vj.append(j); vB.append(B); vc.append(c)
        self._n_vdw_sym = len(vi) - n_au
        self._n_vdw_au = n_au

        self._vi = np.asarray(vi, dtype=np.int64)
        self._vj = np.asarray(vj, dtype=np.int64)
        self._vB = np.asarray(vB, dtype=np.float64).reshape(-1, 3, 3)
        self._vc = np.asarray(vc, dtype=np.float64).reshape(-1, 3)
        radii = np.array([_VDW_RADII.get(str(e).upper(), _VDW_DEFAULT_R)
                          for e in data.elements], dtype=np.float64)
        self._vdmin = radii[self._vi] + radii[self._vj] - _VDW_SLACK
        self._vw = self._w_vdw / (_VDW_SIGMA ** 2)

    # ------------------------------------------------------------------
    @staticmethod
    def _chir_sign(s):
        """+1 / -1 / 0 from a chirality sign (Positive/Negative/Both).

        Works for both gemmi's ChiralityType and the bundled dictionary's
        ``SimpleNamespace(name=...)`` representation.
        """
        nm = getattr(s, "name", str(s).split(".")[-1])
        nm = str(nm).lower()
        if nm.startswith("pos"):
            return 1
        if nm.startswith("neg"):
            return -1
        return 0                     # 'Both' / unknown -> no restraint

    def _load_library(self, kinds):
        """Restraints for the residue kinds present in the model.

        Reads the default dictionary bundled in ``minirefmac/data``.
        """
        if not self.use_monlib:
            return {}
        return monlib.bundled().load(kinds)

    # ------------------------------------------------------------------
    @property
    def n_bonds(self):
        return int(self._mask_bond.sum())

    @property
    def n_angles(self):
        return self._a0.size

    # ------------------------------------------------------------------
    def calc(self, xyz, b_iso):
        """Return (f, grad_xyz (M,3) wrt FRACTIONAL, grad_b (M,), stats)."""
        xyz = np.asarray(xyz, dtype=np.float64)
        b_iso = np.asarray(b_iso, dtype=np.float64)
        m = xyz.shape[0]
        X = xyz @ self._orth.T                     # (M,3) Cartesian, Angstrom
        gcart = np.zeros((m, 3), dtype=np.float64)
        f = 0.0

        # ---- all distance restraints (bonds / 1-3 angles / planes / 1-4) --
        rij = X[self._di] - X[self._dj]            # (R,3)
        d = np.sqrt(np.einsum("ij,ij->i", rij, rij))
        d = np.maximum(d, 1e-9)
        delta = d - self._d0
        f += float(np.sum(self._dw * delta * delta))
        coef = (2.0 * self._dw * delta / d)[:, None] * rij
        _accumulate(gcart, self._di, coef)
        _accumulate(gcart, self._dj, -coef)

        u = X[self._ai] - X[self._aj]
        v = X[self._ak] - X[self._aj]
        un = np.maximum(np.linalg.norm(u, axis=1), 1e-9)
        vn = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
        u /= un[:, None]
        v /= vn[:, None]
        dot = np.einsum("ij,ij->i", u, v)
        sine = np.linalg.norm(np.cross(u, v), axis=1)
        angle_delta = np.arctan2(sine, dot) - self._a0
        f += float(np.sum(self._aw * angle_delta ** 2))
        dc = 2 * self._aw * angle_delta / np.maximum(sine, 1e-8)
        gi = (dot[:, None] * u - v) * (dc / un)[:, None]
        gk = (dot[:, None] * v - u) * (dc / vn)[:, None]
        _accumulate(gcart, self._ai, gi)
        _accumulate(gcart, self._ak, gk)
        _accumulate(gcart, self._aj, -gi - gk)

        # ---- B-factor restraints (on the bonds) -------------------------
        dB = b_iso[self._bi] - b_iso[self._bj]
        f += float(np.sum(self._bw * dB * dB))
        cb = 2.0 * self._bw * dB
        gb = (np.bincount(self._bi, weights=cb, minlength=m)
              - np.bincount(self._bj, weights=cb, minlength=m))

        # ---- chirality: oriented volume --------------------------------
        if self._ci1.size:
            v1 = X[self._ci1] - X[self._cic]
            v2 = X[self._ci2] - X[self._cic]
            v3 = X[self._ci3] - X[self._cic]
            vol = np.einsum("ij,ij->i", v1, np.cross(v2, v3))
            dvol = vol - self._cv
            f += float(np.sum(self._cw * dvol * dvol))
            c = (2.0 * self._cw * dvol)[:, None]
            g1 = c * np.cross(v2, v3)             # dV/dv1
            g2 = c * np.cross(v3, v1)             # dV/dv2
            g3 = c * np.cross(v1, v2)             # dV/dv3
            _accumulate(gcart, self._ci1, g1)
            _accumulate(gcart, self._ci2, g2)
            _accumulate(gcart, self._ci3, g3)
            _accumulate(gcart, self._cic, -(g1 + g2 + g3))

        # ---- true dihedral torsions -------------------------------------
        # tau = atan2( (n1 x n2).b2 / |b2| , n1.n2 )  with
        # b1=p2-p1, b2=p3-p2, b3=p4-p3, n1=b1 x b2, n2=b2 x b3.
        # E = 2w(1-cos(n(tau-tau0)))/n^2  (so E ~ w (tau-tau0)^2 near the
        # minimum, w = w_torsion/sigma_rad^2).  Gradle: dE/dtau times the
        # exact vectorised chain rule for atan2 (verified by FD).
        torsion_dev = np.zeros(0, dtype=np.float64)
        if self._t1.size:
            p1, p2 = X[self._t1], X[self._t2]
            p3, p4 = X[self._t3], X[self._t4]
            b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
            n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
            l2 = np.sqrt(np.einsum("ij,ij->i", b2, b2))
            l2 = np.maximum(l2, 1e-8)
            mvec = np.cross(n1, n2)
            xx = np.einsum("ij,ij->i", n1, n2)
            yy = np.einsum("ij,ij->i", mvec, b2) / l2
            tau = np.arctan2(yy, xx)
            rr = np.maximum(xx * xx + yy * yy, 1e-12)
            A = xx / rr                 # d tau / d y
            Bc = -yy / rr               # d tau / d x
            V = np.cross(n2, b2)
            W = np.cross(b2, n1)
            mb2 = np.einsum("ij,ij->i", mvec, b2)
            Al = (A / l2)[:, None]
            Bl = Bc[:, None]
            G1 = Al * np.cross(b2, V) + Bl * np.cross(b2, n2)
            G3 = Al * np.cross(W, b2) + Bl * np.cross(n1, b2)
            G2 = Al * (np.cross(V, b1) + np.cross(b3, W) + mvec
                       - (mb2 / l2 ** 2)[:, None] * b2) \
                + Bl * (np.cross(n2, b1) + np.cross(b3, n1))
            per = self._tper.astype(np.float64)
            dtau = tau - self._tval
            f += float(np.sum(self._tw * 2.0 * (1.0 - np.cos(per * dtau))
                              / per ** 2))
            dE = (2.0 * self._tw * np.sin(per * dtau) / per)[:, None]
            _accumulate(gcart, self._t1, -dE * G1)
            _accumulate(gcart, self._t2, dE * (G1 - G2))
            _accumulate(gcart, self._t3, dE * (G2 - G3))
            _accumulate(gcart, self._t4, dE * G3)
            wrapped = np.mod(dtau + np.pi / per, 2.0 * np.pi / per) - np.pi / per
            torsion_dev = np.rad2deg(np.abs(wrapped))

        # ---- least-squares planarity ------------------------------------
        plane_z = np.zeros(0, dtype=np.float64)
        if self._pl_w.size:
            Xg = X[self._pl_idx]                             # (P,L,3)
            mk = self._pl_mask[..., None]                    # (P,L,1)
            Xg = np.where(mk, Xg, 0.0)
            cnt = self._pl_mask.sum(1).astype(np.float64)    # (P,)
            xbar = Xg.sum(1) / cnt[:, None]                  # (P,3)
            u = (Xg - xbar[:, None, :]) * mk                 # (P,L,3)
            C = np.einsum("pli,plj->pij", u, u)              # (P,3,3)
            _, vecs = np.linalg.eigh(C)                      # ascending
            nvec = vecs[:, :, 0]                             # (P,3) plane normal
            z = np.einsum("pli,pi->pl", u, nvec)             # (P,L) signed dist
            zm = np.where(self._pl_mask, z, 0.0)
            lam = np.einsum("pl,pl->p", zm, zm)              # = lambda_min
            f += float(np.sum(self._pl_w * lam))
            coef = (2.0 * self._pl_w)[:, None, None] * zm[:, :, None] \
                * nvec[:, None, :]                           # (P,L,3)
            _accumulate(gcart, self._pl_idx.ravel(), coef.reshape(-1, 3))
            plane_z = zm[self._pl_mask]

        # ---- non-bonded VDW repulsion (ASU + symmetry mates) -----------
        grad_extra = np.zeros((m, 3), dtype=np.float64)
        vdw_pen = np.zeros(0, dtype=np.float64)
        if self._vi.size:
            # Xcopy = x_j(frac) @ B[frac,cart] + c   (c = lattice/sym shift in A)
            Xc = np.einsum("pk,pkq->pq", xyz[self._vj], self._vB) + self._vc
            rv = X[self._vi] - Xc
            dv = np.sqrt(np.einsum("ij,ij->i", rv, rv))
            dv = np.maximum(dv, 1e-9)
            vdw_pen = np.maximum(self._vdmin - dv, 0.0)
            f += float(np.sum(self._vw * vdw_pen * vdw_pen))
            coef = (2.0 * self._vw * vdw_pen / dv)[:, None] * rv
            _accumulate(gcart, self._vi, -coef)
            _accumulate(grad_extra, self._vj,
                        np.einsum("pq,pkq->pk", coef, self._vB))

        # ---- Cartesian -> FRACTIONAL gradient (chain rule) -------------
        grad_xyz = gcart @ self._orth + grad_extra

        stats = self._stats(delta, angle_delta, dB, vdw_pen, plane_z, torsion_dev)
        return float(f), grad_xyz, gb, stats

    # ------------------------------------------------------------------
    def _stats(self, delta, angle_delta, dB, vdw_pen=None, plane_z=None,
               torsion_dev=None):
        kb = self._mask_bond
        ka = self._mask_angle
        sig = self._dsig

        def rms(a):
            return float(np.sqrt(np.mean(a))) if len(a) else 0.0

        db = delta[kb]
        da = delta[ka]
        rmsd_bond = rms(db * db)
        rmsd_angle = rms(angle_delta ** 2)
        rmsz_bond = rms(db * db / (sig[kb] ** 2))
        rmsz_angle = rms((angle_delta / self._asig) ** 2)
        rmsd_b = rms(dB * dB)
        max_bond_dev = float(np.max(np.abs(db))) if len(db) else 0.0

        stats = dict(
            rmsd_bond=rmsd_bond,
            rmsd_angle=rmsd_angle,  # radians
            rmsd_angle_deg=float(np.rad2deg(rmsd_angle)),
            rmsd_angle_distance_A=rms(da ** 2),
            n_angle_outlier_4sigma=int(np.count_nonzero(abs(angle_delta) > 4 * self._asig)),
            rmsz_bond=rmsz_bond,
            rmsz_angle=rmsz_angle,
            rmsd_b=rmsd_b,
            n_bond=int(kb.sum()),
            n_angle=self.n_angles,
            max_bond_dev=max_bond_dev,
        )
        # extra diagnostics (do not interfere with the required keys)
        vdw_pen = np.zeros(0) if vdw_pen is None else vdw_pen
        plane_z = np.zeros(0) if plane_z is None else plane_z
        torsion_dev = np.zeros(0) if torsion_dev is None else torsion_dev
        stats.update(
            n_plane=int(self._pl_mask.sum()),
            n_chir=int(self._ci1.size),
            n_bfac=int(self._bi.size),
            n_restraints=int(self._di.size + self.n_angles + self._ci1.size + self._bi.size
                             + self._vi.size + self._pl_mask.sum()
                             + self._tw.size),
            dict_hit_rate=self.info["hit_rate_res"],
            n_plane_groups=int(self._pl_w.size),
            rmsd_plane=rms(plane_z * plane_z),
            max_plane_dev=(float(np.max(np.abs(plane_z))) if plane_z.size else 0.0),
            n_torsion=int(self._tw.size),
            rmsd_torsion=rms(torsion_dev * torsion_dev),
            max_torsion_dev=(float(np.max(torsion_dev)) if torsion_dev.size else 0.0),
            n_vdw=int(self._vi.size),
            n_vdw_sym=int(getattr(self, "_n_vdw_sym", 0)),
            n_vdw_active=int(np.count_nonzero(vdw_pen)),
            rmsd_vdw=rms(vdw_pen * vdw_pen),
            max_vdw_dev=(float(np.max(vdw_pen)) if vdw_pen.size else 0.0),
        )
        return stats
