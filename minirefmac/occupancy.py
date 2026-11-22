"""Work-set occupancy updates for paired alternative conformers."""

from collections import defaultdict
from copy import copy
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

try:
    from . import likelihood, monlib, sfcalc
except ImportError:  # flat-module compatibility
    import likelihood
    import monlib
    import sfcalc


@dataclass(frozen=True)
class Group:
    first: np.ndarray
    second: np.ndarray
    total_percent: int
    labels: tuple


def find_groups(data):
    """Use uniform, matching two-conformer residues with PDB-precision occupancies."""
    residues = defaultdict(lambda: defaultdict(list))
    for row, (residue, alt) in enumerate(zip(data._residue_rows, data.altloc)):
        if alt not in ("", " ", "\x00"):
            residues[residue][alt].append(row)
    groups = []
    for conformers in residues.values():
        if len(conformers) != 2:
            continue
        labels = tuple(sorted(conformers))
        a, b = (np.asarray(conformers[name], dtype=np.intp) for name in labels)
        names_a = sorted(data.atom_names[i] for i in a)
        names_b = sorted(data.atom_names[i] for i in b)
        if names_a != names_b or len(set(names_a)) != len(names_a):
            continue
        units = data.occ[[a[0], b[0]]] * 100
        if not np.isfinite(units).all() or not np.allclose(units, np.rint(units), rtol=0, atol=1e-5):
            continue
        if not all(np.allclose(data.occ[rows], units[i] / 100, rtol=0, atol=1e-7)
                   for i, rows in enumerate((a, b))):
            continue
        p, q = np.rint(units).astype(int)
        if p < 0 or q < 0 or not 0 < p + q <= 100:
            continue
        groups.append(Group(a, b, int(p + q), labels))
    return groups, len(residues) - len(groups)


# Only chemically established aromatic exchanges are audited. Connectivity
# alone cannot distinguish bond order, protonation, or stereochemistry.
_RING_EXCHANGE = {a: b for prefix in ("CD", "CE", "HD", "HE")
                  for a, b in ((prefix + "1", prefix + "2"),
                               (prefix + "2", prefix + "1"))}
EQUIVALENCE_RMSD_A = 0.25


def diagnose_equivalence(data, groups, xyz, external_components=()):
    """Compare approved equivalent labels without alignment or coordinate changes."""
    orth = np.asarray(data.cell.orth.mat.tolist())
    diagnostics = {}
    for index, group in enumerate(groups):
        component = data.res_names[group.first[0]]
        if component not in ("PHE", "TYR") or component in external_components:
            continue
        restraints = monlib.bundled().get(component)
        if restraints is None:
            continue
        bonds = {frozenset((b.id1.atom, b.id2.atom)) for b in restraints.bonds}
        exchanged = {frozenset(_RING_EXCHANGE.get(name, name) for name in bond)
                     for bond in bonds}
        if exchanged != bonds:
            continue
        first = {data.atom_names[i]: i for i in group.first}
        second = {data.atom_names[i]: i for i in group.second}
        if not {"CD1", "CD2", "CE1", "CE2"} <= first.keys():
            continue
        names = sorted(first)
        swapped = [_RING_EXCHANGE.get(name, name) for name in names]
        if any(name not in second or data.elements[first[original]] != data.elements[second[name]]
               for original, name in zip(names, swapped)):
            continue
        x1 = xyz[[first[name] for name in names]]
        deltas = []
        for matching in (names, swapped):
            x2 = xyz[[second[name] for name in matching]]
            distance = np.linalg.norm((x1 - x2) @ orth.T, axis=1)
            deltas.append((float(np.sqrt(np.mean(distance ** 2))), float(distance.max())))
        (direct, direct_max), (exchange, exchange_max) = deltas
        use_exchange = exchange < direct
        rmsd = exchange if use_exchange else direct
        diagnostics[index] = dict(
            component=component, direct_rmsd_A=direct, exchange_rmsd_A=exchange,
            best_matching="aromatic_exchange" if use_exchange else "same_labels",
            best_rmsd_A=rmsd, max_displacement_A=exchange_max if use_exchange else direct_max,
            near_overlap=bool(rmsd <= EQUIVALENCE_RMSD_A))
    return diagnostics


def _unit_fc(data, rows, xyz, b_iso, chunk):
    subset = copy(data)
    subset.xyz = xyz[rows]
    if data.sf_table is not None:
        subset.sf_table = data.sf_table[:, rows]
    else:
        subset.elem_index = data.elem_index[rows]
    return sfcalc.calc_fc(subset, subset.xyz, b_iso[rows], np.ones(rows.size), chunk=chunk)


def _target(data, base, basis, x0, idx, scale, mlpar, use_phase_prior):
    gradient_basis = basis[idx].conj().T

    def evaluate(values):
        fc = base + basis @ (values - x0)
        ll, g, _, _ = likelihood._eval_at(data, fc, scale, mlpar, idx, use_phase_prior)
        gradient = np.real(gradient_basis @ g)
        return float(ll.sum()), gradient
    return evaluate


def refine_groups(data, groups, xyz, b_iso, fc_atoms, fmask, scale, mlpar,
                  use_phase_prior=False, chunk=48, prior_center=None, prior_sigma=None):
    """Return occupancies plus before/after work and penalized target scores."""
    idx = np.flatnonzero(data.work)
    if not groups or not idx.size:
        return data.occ, None
    x0 = np.array([data.occ[g.first[0]] for g in groups])
    total_percent = np.array([g.total_percent for g in groups])
    upper = total_percent / 100
    basis = np.column_stack([
        _unit_fc(data, g.first, xyz, b_iso, chunk)
        - _unit_fc(data, g.second, xyz, b_iso, chunk) for g in groups])
    basis *= scale.fc_atom_factor(data)[:, None]
    base = sfcalc.build_fc_total(data, fc_atoms, fmask, scale)
    work_target = _target(data, base, basis, x0, idx, scale, mlpar, use_phase_prior)

    def evaluate(values):
        loss, grad = work_target(values)
        if prior_sigma is not None:
            shift = (values - prior_center) / prior_sigma
            loss += 0.5 * float(np.dot(shift, shift))
            grad += shift / prior_sigma
        return loss, grad

    initial, _ = evaluate(x0)
    work_initial, _ = work_target(x0)
    result = minimize(evaluate, x0, jac=True, method="L-BFGS-B",
                      bounds=list(zip(np.zeros_like(upper), upper)),
                      options={"maxiter": 30, "ftol": 1e-12})
    if not np.isfinite(result.x).all():
        return data.occ, None
    first_percent = np.clip(np.rint(result.x * 100).astype(int), 0, total_percent)
    final, _ = evaluate(first_percent / 100)
    if not np.isfinite(final) or final >= initial - 1e-8:
        return data.occ, (work_initial, work_initial, initial, initial)
    result_occ = data.occ.copy()
    for g, p in zip(groups, first_percent):
        result_occ[g.first] = p / 100
        result_occ[g.second] = (g.total_percent - p) / 100
    work_final, _ = work_target(first_percent / 100)
    return result_occ, (work_initial, work_final, initial, final)
