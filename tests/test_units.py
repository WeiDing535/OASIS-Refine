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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from minirefmac import dataio, fftcalc, likelihood, sfcalc  # noqa: E402
from minirefmac import core, fastkernel, simdkernel  # noqa: E402

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


class TestAutoSpacing(unittest.TestCase):
    def setUp(self):
        import gemmi
        self.cell = gemmi.UnitCell(60.0, 70.0, 80.0, 90.0, 90.0, 90.0)

    def test_default_is_fine_enough(self):
        sp = fftcalc.auto_spacing(self.cell)
        self.assertLessEqual(sp, 0.25 + 1e-9)

    def test_sharp_atoms_refine_the_grid(self):
        coarse = fftcalc.auto_spacing(self.cell, b_min=60.0)
        sharp = fftcalc.auto_spacing(self.cell, b_min=4.0)
        self.assertLessEqual(sharp, coarse + 1e-9)
        self.assertGreaterEqual(sharp, 0.18 - 1e-9)   # floor keeps it affordable

    def test_high_resolution_limit(self):
        sp = fftcalc.auto_spacing(self.cell, d_min=0.8)
        self.assertLessEqual(sp, 0.2 + 1e-9)

    def test_memory_cap_is_respected(self):
        sp = fftcalc.auto_spacing(self.cell, target=0.05, vmax=1e6)
        nu = [round(v / sp) for v in (60.0, 70.0, 80.0)]
        self.assertLessEqual(nu[0] * nu[1] * nu[2], 1.1e6)
        self.assertGreater(sp, 0.05)


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

    def test_perfect_fc_gives_zero_r(self):
        k = np.asarray(self.scale.k_total(self.d))
        fc = np.where(k > 0, self.d.fo / np.where(k == 0, 1.0, k), 0.0).astype(
            np.complex128)
        r_work, _, _ = likelihood.r_factors(self.d, fc, self.scale)
        self.assertLess(r_work, 1e-9)

    def test_fused_kernel_matches_numpy(self):
        G = np.ones(self.d.n_ref, dtype=np.complex128)
        fcf = np.ones(self.d.n_ref, dtype=np.float64)   # bulk-solvent factor
        fc_np = self.fc
        fc_fu, gx_fu, gb_fu = fastkernel.fused_fc_and_gradients(
            self.d, self.d.xyz, self.d.b_iso, self.d.occ, G, fcf)
        self.assertLess(_rel(fc_fu, fc_np), 1e-12)

    def test_simd_kernel_matches_numpy_when_available(self):
        ok, _ = simdkernel.probe()
        if not ok:
            self.skipTest("no AVX-512 / compiler on this machine")
        G = np.ones(self.d.n_ref, dtype=np.complex128)
        fcf = np.ones(self.d.n_ref, dtype=np.float64)
        fc_np = self.fc
        fc_sd, _, _ = simdkernel.simd_fc_and_gradients(
            self.d, self.d.xyz, self.d.b_iso, self.d.occ, G, fcf)
        self.assertLess(_rel(fc_sd, fc_np), 1e-9)

    def test_fft_kernel_accuracy(self):
        sp = fftcalc.auto_spacing(self.d.cell, self.d.d_min,
                                  b_min=float(np.min(self.d.b_iso)))
        grid, atoms, _ = fftcalc._scatter_density(
            self.d, sp, sfcalc._sym_ops(self.d), None,
            xyz=self.d.xyz, b_iso=self.d.b_iso, occ=self.d.occ)
        fc_fft = fftcalc.fft_fc(self.d, spacing=sp, radius_A=None,
                                grid=grid, atoms=atoms)
        self.assertLess(_rel(fc_fft, self.fc), 5e-3)


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
