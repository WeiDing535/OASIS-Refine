"""Fast unit tests for minirefmac (stdlib unittest, no pytest needed).

    python -m unittest discover -s tests -v
    # or
    python tests/test_units.py

These complement ``python -m minirefmac.selftest`` (which is the heavy
numerical acceptance test): here we pin the small decisions that are easy to
break and hard to notice -- FreeR flag conventions, FFT grid sizing, kernel
agreement, R-factor algebra and MTZ column detection.
"""

import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace

import gemmi
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from minirefmac import dataio, likelihood, sfcalc  # noqa: E402
from minirefmac import core, fastkernel, geometry, occupancy, simdkernel  # noqa: E402

MTZ = os.path.join(ROOT, "test_data", "7enm", "input.mtz")
PDB = os.path.join(ROOT, "test_data", "7enm", "input.pdb")
HAVE_DATA = os.path.exists(MTZ) and os.path.exists(PDB)


def _rel(a, b):
    den = float(np.linalg.norm(b))
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / den) if den else 0.0


class TestFreeValue(unittest.TestCase):
    def test_ccp4_zero_is_free(self):
        flags = np.array([0] * 5 + [1] * 95)
        got, info = dataio.resolve_free_value(flags, np.ones(flags.size, bool), "auto")
        self.assertEqual(got, 0)
        self.assertAlmostEqual(info["free_fraction"], 0.05, places=6)
        self.assertNotIn("suspicious", info)

    def test_inverted_zero_is_work(self):
        flags = np.array([0] * 95 + [1] * 5)
        got, info = dataio.resolve_free_value(flags, np.ones(flags.size, bool), "auto")
        self.assertEqual(got, 1)
        self.assertAlmostEqual(info["free_fraction"], 0.05, places=6)

    def test_explicit_override_wins(self):
        flags = np.array([0] * 5 + [1] * 95)
        got, info = dataio.resolve_free_value(flags, np.ones(flags.size, bool), 1)
        self.assertEqual((got, info["mode"]), (1, "explicit"))
        self.assertIn("suspicious", info)          # 95 % free -> meaningless Rfree

    def test_single_value_flags_fall_back(self):
        flags = np.zeros(100, dtype=np.int64)
        got, info = dataio.resolve_free_value(flags, np.ones(100, bool), "auto")
        self.assertEqual(got, 0)
        self.assertTrue(info.get("fallback"))
        self.assertIn("suspicious", info)

    def test_string_auto(self):
        flags = np.array([0] * 6 + [1] * 94)
        got, _ = dataio.resolve_free_value(flags, np.ones(flags.size, bool), "AUTO")
        self.assertEqual(got, 0)


class TestDisulfide(unittest.TestCase):
    def test_short_sulfur_pair_is_bonded_and_excluded_from_vdw(self):
        xyz = np.array([[0.1, 0.1, 0.1], [0.3031, 0.1, 0.1]])
        d = SimpleNamespace(
            cell=gemmi.UnitCell(10, 10, 10, 90, 90, 90), xyz=xyz,
            atom_names=["SG", "SG"], res_names=["CYS", "CYS"],
            chain_ids=["A", "B"], res_seq=[1, 1], elements=["S", "S"],
        )
        restraints = geometry.Restraints(d, sym_vdw=False)
        self.assertEqual(restraints.info["n_disulfide"], 1)
        self.assertEqual(restraints.info["n_vdw"], 0)
        f, _, _, _ = restraints.calc(xyz, np.zeros(2))
        self.assertAlmostEqual(f, 0.0)

    def test_explicit_disulfide_overrides_nearby_inferred_partner(self):
        st = gemmi.read_pdb_string(
            "SSBOND   1 CYS A    6    CYS B    7                          1555   1555  4.59  \n"
            "ATOM      1  SG  CYS A   6       1.000   1.000   1.000  1.00 20.00           S  \n"
            "ATOM      2  SG  CYS B   7       5.590   1.000   1.000  1.00 20.00           S  \n"
            "ATOM      3  SG  CYS B   8       3.100   1.000   1.000  1.00 20.00           S  \n"
            "END\n")
        xyz = np.array([[0.1, 0.1, 0.1], [0.559, 0.1, 0.1], [0.31, 0.1, 0.1]])
        data = SimpleNamespace(
            cell=gemmi.UnitCell(10, 10, 10, 90, 90, 90), xyz=xyz,
            atom_names=["SG"] * 3, res_names=["CYS"] * 3,
            chain_ids=["A", "B", "B"], res_seq=[6, 7, 8], elements=["S"] * 3,
            st=st, n_atoms=3,
        )
        restraints = geometry.Restraints(data, sym_vdw=False)
        self.assertEqual(restraints.info["n_disulfide"], 1)
        self.assertEqual(restraints.info["n_disulfide_explicit"], 1)
        _, gradient, _, _ = restraints.calc(xyz, np.zeros(3))
        self.assertLess(gradient[0, 0], 0)
        self.assertGreater(gradient[1, 0], 0)


class TestComponentCif(unittest.TestCase):
    def test_external_component_changes_restraint_target(self):
        xyz = np.array([[0.1, 0.1, 0.1], [0.23, 0.1, 0.1]])
        data = SimpleNamespace(
            cell=gemmi.UnitCell(10, 10, 10, 90, 90, 90), xyz=xyz,
            atom_names=["C1", "O1"], res_names=["XGL"] * 2,
            chain_ids=["A"] * 2, res_seq=[1] * 2, elements=["C", "O"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "component.cif")
            with open(path, "w") as fh:
                fh.write("""data_comp_XGL
_chem_comp.id XGL
_chem_comp.type NON-POLYMER
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
C1 C
O1 O
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.type
_chem_comp_bond.value_dist
_chem_comp_bond.value_dist_esd
C1 O1 single 1.50 0.02
""")
            restraints = geometry.Restraints(data, sym_vdw=False,
                                             restraint_cifs=[path])
        self.assertEqual(restraints.info["external_components"], ["XGL"])
        _, gradient, _, stats = restraints.calc(xyz, np.zeros(2))
        self.assertAlmostEqual(stats["rmsd_bond"], 0.2)
        self.assertGreater(gradient[0, 0], 0)


class TestOccupancyEquivalence(unittest.TestCase):
    def test_aromatic_exchange_matches_only_overlapping_sites(self):
        names = ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"]
        first = np.array([[1, 2, 2], [2, 2, 2], [3, 1, 2], [3, 3, 2],
                          [4, 1, 2], [4, 3, 2], [5, 2, 2]], dtype=float)
        second = first[[0, 1, 3, 2, 5, 4, 6]]
        xyz = np.vstack((first, second)) / 10
        data = SimpleNamespace(cell=gemmi.UnitCell(10, 10, 10, 90, 90, 90),
                               atom_names=names * 2, elements=["C"] * 14,
                               res_names=["PHE"] * 14)
        group = occupancy.Group(np.arange(7), np.arange(7, 14), 100, ("A", "B"))
        result = occupancy.diagnose_equivalence(data, [group], xyz)[0]
        self.assertEqual(result["best_matching"], "aromatic_exchange")
        self.assertTrue(result["near_overlap"])
        self.assertAlmostEqual(result["best_rmsd_A"], 0.0)
        xyz[7:, 0] += 0.05
        self.assertFalse(occupancy.diagnose_equivalence(data, [group], xyz)[0]["near_overlap"])
        self.assertEqual(occupancy.diagnose_equivalence(data, [group], xyz, ["PHE"]), {})


@unittest.skipUnless(HAVE_DATA, "test_data/7enm missing")
class TestOnRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = dataio.load_data(MTZ, PDB)
        cls.fc = sfcalc.calc_fc(cls.d, cls.d.xyz, cls.d.b_iso, cls.d.occ)
        cls.fmask = sfcalc.calc_fmask(cls.d, cls.d.xyz)
        cls.scale = sfcalc.fit_scales(cls.d, cls.fc, cls.fmask, solvent="mask",
                                      fit_aniso=True)
        cls.fc_total = sfcalc.build_fc_total(cls.d, cls.fc, cls.fmask, cls.scale)

    def test_free_set_is_five_percent(self):
        fi = self.d.free_info
        self.assertEqual(fi["mode"], "auto")
        self.assertEqual(int(self.d.free_value), 0)
        self.assertAlmostEqual(fi["free_fraction"], 0.048, places=3)
        self.assertEqual(int(self.d.test.sum()) + int(self.d.work.sum()),
                         int(self.d.observed.sum()))

    def test_r_factors_are_sane(self):
        r_work, r_free, r_all = likelihood.r_factors(self.d, self.fc_total,
                                                    self.scale)
        for r in (r_work, r_free, r_all):
            self.assertTrue(0.0 < r < 1.0)
        self.assertAlmostEqual(r_all, 0.383, delta=0.02)

    def test_rfree_stats_matches_r_factors(self):
        r_work, r_free, _ = likelihood.r_factors(self.d, self.fc_total,
                                                self.scale)
        st = likelihood.rfree_stats(self.d, self.fc_total, self.scale)
        self.assertEqual(st["n_free"], int(self.d.test.sum()))
        self.assertAlmostEqual(st["r_free"], r_free, places=12)
        # Rfree on a 5 % free set is noisy: s.e. of order 1e-2, not 1e-3
        self.assertTrue(0.005 < st["se"] < 0.05)

    def test_paired_test_identical_models(self):
        st = likelihood.rfree_stats(self.d, self.fc_total, self.scale,
                                    ref_fc=self.fc_total, ref_scale=self.scale)
        self.assertAlmostEqual(st["delta_vs_ref"], 0.0, places=12)
        self.assertFalse(st["significant"])

    def test_paired_test_detects_worse_model(self):
        # a 25 % scale error on Fc must show up as a clearly worse Rfree.
        # (5-15 % errors are NOT significant on 487 free reflections -- that is
        # the point of the statistic.)
        st = likelihood.rfree_stats(self.d, self.fc_total * 1.25, self.scale,
                                    ref_fc=self.fc_total, ref_scale=self.scale)
        self.assertGreater(st["delta_vs_ref"], 0.03)
        self.assertTrue(st["significant"])
        self.assertGreater(abs(st["t"]), 3.0)
        # the paired s.e. must be smaller than the unpaired one
        self.assertLess(st["se_delta"], st["se"])

    def test_perfect_fc_gives_zero_r(self):
        k = np.asarray(self.scale.k_total(self.d))
        fc = np.where(k > 0, self.d.fo / np.where(k == 0, 1.0, k), 0.0).astype(
            np.complex128)
        r_work, _, _ = likelihood.r_factors(self.d, fc, self.scale)
        self.assertLess(r_work, 1e-9)

    def test_alternate_atoms_share_backbone_and_have_separate_restraints(self):
        st = self.d.st.clone()
        res = next(r for ch in st[0] for r in ch if r.name == "SER")
        side = res.find_atom("CB", "*")
        side.altloc = "A"
        side.occ = 0.6
        other = side.clone()
        other.altloc = "B"
        other.occ = 0.4
        other.pos.x += 0.2
        res.add_atom(other)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "alternate.pdb")
            st.write_pdb(path)
            default = dataio.load_data(MTZ, path)
            full = dataio.load_data(MTZ, path, all_altlocs=True)
            self.assertEqual(default.n_atoms + 1, full.n_atoms)
            rows = [i for i, alt in enumerate(full.altloc) if alt in ("A", "B")]
            self.assertEqual(len(rows), 2)
            np.testing.assert_allclose(sorted(full.occ[rows]), [0.4, 0.6])
            r = geometry.Restraints(full)
            r0 = geometry.Restraints(default)
            bonded = set(zip(r._di[r._mask_bond], r._dj[r._mask_bond]))
            self.assertTrue(all(any(i in pair for pair in bonded) for i in rows))
            self.assertIn(tuple(rows), r._altloc_conflicts)
            self.assertNotIn(tuple(rows), set(zip(r._vi[:r._n_vdw_au], r._vj[:r._n_vdw_au])))
            self.assertEqual(r.info["n_peptide"], r0.info["n_peptide"])
            moved = full.xyz.copy()
            moved[rows[-1], 0] += 0.01
            self.assertFalse(np.allclose(
                sfcalc.calc_fc(full, full.xyz, full.b_iso),
                sfcalc.calc_fc(full, moved, full.b_iso)))
            dataio.write_refined_pdb(full, moved, full.b_iso, path)
            readback = dataio.load_data(MTZ, path, all_altlocs=True)
            np.testing.assert_array_equal(full.occ, readback.occ)
            np.testing.assert_allclose(readback.xyz[rows[-1]], moved[rows[-1]], atol=2e-5)

    def test_grouped_occupancy_target_gradient_and_conservation(self):
        st = self.d.st.clone()
        res = next(r for ch in st[0] for r in ch if r.name == "SER")
        side = res.find_atom("CB", "*")
        side.altloc, side.occ = "A", 0.6
        other = side.clone()
        other.altloc, other.occ = "B", 0.25
        other.pos.x += 0.4
        res.add_atom(other)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "occupancy.pdb")
            st.write_pdb(path)
            d = dataio.load_data(MTZ, path, all_altlocs=True)
        groups, skipped = occupancy.find_groups(d)
        self.assertEqual((len(groups), skipped), (1, 0))
        g = groups[0]
        self.assertEqual(g.total_percent, 85)
        fc = sfcalc.calc_fc(d, d.xyz, d.b_iso)
        total = sfcalc.build_fc_total(d, fc, self.fmask, self.scale)
        ml = likelihood.estimate_ml_params(d, total, self.scale, use="work")
        response = occupancy._unit_fc(d, g.first, d.xyz, d.b_iso, 48)
        response -= occupancy._unit_fc(d, g.second, d.xyz, d.b_iso, 48)
        response *= self.scale.fc_atom_factor(d)
        x = d.occ[g.first].copy()
        evaluate = occupancy._target(d, total, response[:, None], x,
                                     np.flatnonzero(d.work), self.scale, ml, False)
        _, grad = evaluate(x)
        step = 1e-5
        fd = (evaluate(x + step)[0] - evaluate(x - step)[0]) / (2 * step)
        self.assertAlmostEqual(grad[0], fd, delta=1e-4)

        updated, scores = occupancy.refine_groups(
            d, groups, d.xyz, d.b_iso, fc, self.fmask, self.scale, ml)
        self.assertLessEqual(scores[1], scores[0])
        self.assertAlmostEqual(updated[g.first[0]] + updated[g.second[0]], 0.85)
        prior_updated, penalized = occupancy.refine_groups(
            d, groups, d.xyz, d.b_iso, fc, self.fmask, self.scale, ml,
            prior_center=x, prior_sigma=0.02)
        self.assertLessEqual(penalized[3], penalized[2])
        self.assertLessEqual(abs(prior_updated[g.first[0]] - x[0]),
                             abs(updated[g.first[0]] - x[0]) + 1e-8)
        others = np.ones(d.n_atoms, bool)
        others[np.r_[g.first, g.second]] = False
        np.testing.assert_array_equal(updated[others], d.occ[others])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "written.pdb")
            dataio.write_refined_pdb(d, d.xyz, d.b_iso, path, occ=updated)
            written = dataio.load_data(MTZ, path, all_altlocs=True)
            np.testing.assert_allclose(written.occ, updated, rtol=0, atol=5e-8)

    def test_true_angle_rmsd_is_invariant_under_uniform_scaling(self):
        restraints = geometry.Restraints(self.d)
        _, _, _, start = restraints.calc(self.d.xyz, self.d.b_iso)
        _, _, _, scaled = restraints.calc(1.1 * self.d.xyz, self.d.b_iso)
        self.assertGreater(start["rmsd_angle_deg"], 0.0)
        self.assertAlmostEqual(start["rmsd_angle_deg"], scaled["rmsd_angle_deg"])
        self.assertAlmostEqual(start["rmsd_angle_deg"], np.rad2deg(start["rmsd_angle"]))

    def test_fused_kernel_matches_numpy(self):
        G = np.ones(self.d.n_ref, dtype=np.complex128)
        fcf = np.ones(self.d.n_ref, dtype=np.float64)   # bulk-solvent factor
        fc_np = self.fc
        fc_fu, gx_fu, gb_fu = fastkernel.fused_fc_and_gradients(
            self.d, self.d.xyz, self.d.b_iso, self.d.occ, G, fcf)
        self.assertLess(_rel(fc_fu, fc_np), 1e-12)

    def test_simd_kernel_matches_numpy_when_available(self):
        ok, _ = simdkernel.probe()
        if not ok or not simdkernel.ensure_simd_lib():
            self.skipTest("AVX-512 kernel unavailable")
        G = np.ones(self.d.n_ref, dtype=np.complex128)
        fcf = np.ones(self.d.n_ref, dtype=np.float64)
        fc_np = self.fc
        fc_sd, gx_sd, gb_sd = simdkernel.simd_fc_and_gradients(
            self.d, self.d.xyz, self.d.b_iso, self.d.occ, G, fcf)
        self.assertLess(_rel(fc_sd, fc_np), 1e-9)
        fc, gx, gb = simdkernel.simd_fc_and_gradients(
            self.d, self.d.xyz, self.d.b_iso, self.d.occ, None, None, want_grad=False)
        self.assertLess(_rel(fc, fc_sd), 1e-12)
        self.assertIsNone(gx)
        self.assertIsNone(gb)
        fc, gx, gb = simdkernel.simd_fc_and_gradients(
            self.d, self.d.xyz, self.d.b_iso, self.d.occ, G, fcf, want_fc=False)
        self.assertIsNone(fc)
        self.assertLess(_rel(gx, gx_sd), 1e-12)
        self.assertLess(_rel(gb, gb_sd), 1e-12)

    def test_threaded_simd_matches_serial(self):
        if not simdkernel.ensure_simd_lib():
            self.skipTest("AVX-512 kernel unavailable")
        G = np.linspace(0.1, 1.0, self.d.n_ref) * (1 + 1j)
        args = (self.d, self.d.xyz, self.d.b_iso, self.d.occ, G,
                np.ones(self.d.n_ref, dtype=np.float64))
        serial = simdkernel.simd_fc_and_gradients(*args)
        threaded = simdkernel.simd_fc_and_gradients(*args, threads=4)
        for actual, expected in zip(threaded, serial):
            self.assertLess(_rel(actual, expected), 1e-12)

        for kwargs in (dict(want_grad=False), dict(want_fc=False)):
            expected = simdkernel.simd_fc_and_gradients(*args, **kwargs)
            actual = simdkernel.simd_fc_and_gradients(*args, threads=4, **kwargs)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)


@unittest.skipUnless(HAVE_DATA, "test_data/7enm missing")
class TestColumnDetection(unittest.TestCase):
    """Column-name autodetection is what makes 'just works' real: pin it."""

    def _rewrite(self, tmp, rename, drop=()):
        import gemmi
        m = gemmi.read_mtz_file(MTZ)
        for old, new in rename.items():
            labels = [c.label for c in m.columns]
            self.assertIn(old, labels, "test fixture lacks column %s" % old)
            m.columns[labels.index(old)].label = new
        for name in drop:
            m.remove_column([c.label for c in m.columns].index(name))
        out = os.path.join(tmp, "rewritten.mtz")
        m.write_to_file(out)
        return out

    def test_renamed_columns_are_found(self):
        tmp = tempfile.mkdtemp(prefix="mr_cols_")
        try:
            path = self._rewrite(tmp, {"F": "FP", "SIGF": "SIGFP",
                                       "FreeR_flag": "R-free-flags"})
            d = dataio.load_data(path, PDB)
            self.assertEqual(d.fp_label, "FP")
            self.assertEqual(d.free_label, "R-free-flags")
            self.assertGreater(int(d.test.sum()), 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_free_column_disables_rfree(self):
        tmp = tempfile.mkdtemp(prefix="mr_nofree_")
        try:
            path = self._rewrite(tmp, {}, drop=("FreeR_flag",))
            d = dataio.load_data(path, PDB)
            self.assertEqual(d.free_info["mode"], "absent")
            self.assertEqual(int(d.test.sum()), 0)
            self.assertEqual(int(d.work.sum()), int(d.observed.sum()))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)


def test_invalid_b_factor_optimizer_scale_is_rejected():
    from minirefmac.refine import Options, Refiner

    for scale in (0, -1, float("inf"), float("nan")):
        with unittest.TestCase().assertRaisesRegex(ValueError, "B-factor optimizer scale"):
            Refiner(None, Options(b_scale=scale))
