# OASIS-Refine Linux Validation Report

Test host — Ubuntu 26.04, x86_64, 96 cores, 123 GB RAM, AVX-512F+DQ+VL
Environment — Python 3.12.15 in a uv virtual environment with gemmi 0.7.5, NumPy 2.5.5, and SciPy 1.18.1
Report date — 2026-11-21. Initial validation and retesting after revisions took place on November 20.

---

## 0. Summary

The reconstructed deployment passed all four original Linux checks. The second round fixed silent
correctness problems and replaced manual log inspection with automated pass criteria.
All seven stages and the optional comparison across three kernels passed in 126 s.

The original delivery failed because packaging flattened the directory tree (see Section 1).
The second round added `pack_release.sh` to preserve that tree.

---

## 1. First round and diagnosis

The original `/Large_pool/Workbuddy/OASIS-Refine/` directory stored every file at the top level.

```
./cli.py  ./core.py  ./dataio.py ... ./monlib_bundle.zip  ./input.mtz  ./input.pdb  ./__init__.py
```

Both entry points failed immediately.

```
$ python3 check_install.py
FAIL: cannot import minirefmac: No module named 'minirefmac'

$ ./run_linux_test.sh
FAILED: missing test data: .../OASIS-Refine/test_data/7enm/input.mtz
```

Flattening removed the directory structure for `minirefmac/`, `minirefmac/data/`, `minirefmac/tools/`,
and `test_data/7enm/`. The two files named `__init__.py` collided, leaving only the copy
from the data directory. The top-level `__init__.py` checksum `639a0aa9…` matched the manifest entry
for `minirefmac/data/__init__.py`. The original `minirefmac/__init__.py`
(checksum `480fe6a8…`) was missing.
Operations such as `zip -j` can produce this flattened layout.

The manifest identified the rebuilt file as the only checksum mismatch.

```
$ sha256sum -c MANIFEST.sha256
minirefmac/__init__.py: FAILED          # Rebuilt file
28 / 29 entries OK
```

> The rebuilt `minirefmac/__init__.py` contained a docstring and `__version__`. In the first round,
> refinement used the version hardcoded in `refine.py`. The second round centralized it in
> `minirefmac.__version__` (0.2.0), which is used in `report.json` and MTZ history.

---

## 2. Restored deployment and second-round revisions

The tree was reconstructed from the manifest under
`/Large_pool/Workbuddy/OASIS-Refine-linux/`, leaving the original directory intact.
`CHANGELOG.md` documents the revisions.

| Category | Revisions |
|---|---|
| Correctness | Automatic FreeR flag detection, shared G across four kernels, finer FFT grid (Fc error 3.5% → 0.11%), atomic `FC/PHIC` contributions |
| Interface | `--kernel {auto,numpy,fused,simd}` with legacy aliases, `--version`, `python -m minirefmac`, separate error codes |
| Tests | 17 new unit tests, self-test expanded from 8 to 10 checks, new `check_outputs.py`, Linux test expanded from 4 to 7 stages |
| Packaging | `pack_release.sh` preserves the layout, regenerates the manifest, and verifies archive extraction |

Reproduce the validation with the following commands.

```bash
cd /Large_pool/Workbuddy/OASIS-Refine-linux
./run_linux_test.sh --kernels      # All seven stages plus kernel agreement, 126 s
./run_linux_test.sh --checks-only  # Skip end-to-end refinement, 70 s
```

---

## 3. Second-round automated validation

| Stage | Criterion | Observed result | Outcome |
|---|---|---|---|
| 0/7 Environment | Python ≥ 3.9 and importable dependencies | 3.12.15, `auto → simd (prebuilt _simd.so)`, 96 cores | PASS |
| 1/7 Integrity | `RESULT : PASS` with no external dictionary paths | 35,639 residues, 47.8 MB, license present, no external paths | PASS |
| 2/7 Real-data restraints | `RESULT : PASS` with nonzero counts | 684 bonds, 915 angles, 117 planes, 242 torsions, f=7261.923272 | PASS |
| 3/7 Unit tests | `OK` (17 tests) | 7 s | PASS |
| 4/7 Numerical self-test | `SELFTEST PASS` (10 checks) | Fc vs gemmi 3.88e-08, X-ray FD 1.57e-08, geometry FD 3.29e-08, kernel agreement fused 4e-16 / SIMD 1e-14, 61 s | PASS |
| 5/7 Short refinement | Prints `RESULT` with no increase in R or Rfree | R 0.3830→0.3825, Rfree 0.4116→0.4091, rmsBOND 0.0181→0.0091, 7 s | PASS |
| 6/7 Output checks | `RESULT : PASS` | 15 MTZ columns, unchanged space group, cell, and reflection count, no NaNs, R(MTZ)=0.3862 vs report 0.3825 (difference 0.0037, see Section 5.2) | PASS |
| 7/7 Kernel agreement (`--kernels`) | R and Rfree differ by < 1e-6 | NumPy, fused, and SIMD all gave `R=0.382630 Rfree=0.409866`, maximum difference 2.5e-11 | PASS |

Total runtime was 126 s, or 70 s with `--checks-only`. Stage results and timings are in
`linux_test_out/summary.txt`. Environment details are in `env.txt`.

---

## 4. Additional validation

### 4.1 Multi-cycle refinement (5 cycles × 10 iterations)

The revised SIMD run took **39.4 s**. The earlier NumPy run took 215 s.

```
cycle 0  R=0.3830  Rfree=0.4116  FOM=0.682  rmsBOND=0.0181  w=2.59
cycle 1  R=0.3825  Rfree=0.4030  FOM=0.688  rmsBOND=0.0063  w=2.59
cycle 2  R=0.3775  Rfree=0.4017  FOM=0.701  rmsBOND=0.0056  w=3.11
cycle 3  R=0.3740  Rfree=0.4009  FOM=0.709  rmsBOND=0.0050  w=3.74
cycle 4  R=0.3710  Rfree=0.4027  FOM=0.717  rmsBOND=0.0055  w=4.48
cycle 5  R=0.3687  Rfree=0.4047  FOM=0.720  rmsBOND=0.0050  w=4.48
```

The SIMD run matched the original NumPy values cycle by cycle (final R 0.3687 and Rfree 0.4047)
and ran 5.5× as fast. All kernels used the same coefficient G.
Rfree reached its minimum in cycle 3 and rose afterward, suggesting mild overfitting by cycle 5 on both paths.

### 4.2 Larger dataset (3,056 atoms, 10,607 reflections, 3.30 Å)

`/raid0_ssd/AI-XRAY/output_hm/4L6R/4L6R_030075.{mtz,pdb}`.

```
Automatically detected columns (FOBS / SIGFOBS / R-free-flags)
free set: R-free-flags=1 -> 531 of 10607 observed (5.0 %)
1 cycle × 3 iterations in 20.6 s (SIMD), R 0.2785→0.2702, Rfree 0.2781→0.2759
```

The original run required `--free-value 1`. The old default assigned 95% of reflections
to the free set and gave an unusable Rfree of 0.3255. The revised `auto` detected the
5.0% free set and reported it in the startup log.

### 4.3 Kernel performance (7enm, 1 cycle × 1 iteration)

| Kernel | Time | Notes |
|---|---|---|
| numpy | 16.6 s | Reference implementation |
| fused | 16.8 s | Available without a compiler |
| simd | 4.8 s | 3.5× as fast as NumPy. Builds `_simd.so` on first use. |

> Version 0.3.0 removed the FFT path (`fftcalc.py`). See `CHANGELOG.md` for the earlier FFT changes.

### 4.4 Output compatibility with CCP4 tools

`mtzdump` from CCP4 9.0.017 read `refined.mtz` successfully.

```
P 21 21 21, cell 35.245 40.703 66.328 90 90 90, 11057 reflections, 34.692–1.696 Å
15 columns: H K L F SIGF FreeR_flag FWT PHWT DELFWT PHDELWT FOM FC_ALL PHIC_ALL FC PHIC
HISTORY: MINIREFMAC version=0.2.0 kernel=simd / output_scale=absolute / F_abs = F_obs/k_total
         free_column=FreeR_flag free_value=0 n_free=487 n_work=9713 / R_work R_free -LL
```

`FWT/PHWT` and `DELFWT/PHDELWT` follow CCP4 and Coot conventions for 2Fo−Fc and Fo−Fc maps.
`refined.pdb` reopened successfully with 670 atoms, 82 residues, 2 chains, B factors of 5.4–74.7 Å², and no NaNs.

### 4.5 Installation and entry points

- `pip install .` succeeded in a clean virtual environment. `minirefmac --version` and `minirefmac-selftest` worked.
- `python -m minirefmac` and `python -m minirefmac.cli` are equivalent.
- At the time of the test, `run.sh` could use the local virtual environment, `$PYTHON`, or `python3`.
- The launcher worked from other directories and with `PYTHONPATH=/tmp`.
- `pack_release.sh` produced `oasis-refine-0.2.0.tar.gz` (46 MB, 36 files).
  After extraction, `sha256sum -c MANIFEST.sha256` and `import minirefmac` succeeded.
- A clean extraction with a fresh uv environment passed all five checks in
  `./run_linux_test.sh --checks-only` in 72 s.
  The SIMD kernel was rebuilt automatically in the extracted directory.

---

## 5. Correctness issues identified in validation

### 5.1 FreeR flag selection

The original default `--free-value 0` followed the CCP4 convention. The AI-XRAY MTZ used
`0=work, 1=free` for 10,076 and 531 reflections respectively. The run continued with
`work 531 / free 10076`, giving an unusable Rfree of 0.3255 on 95% of the reflections.

The revised default `--free-value auto` examines the flag histogram. For 4L6R it reports
`R-free-flags=1 -> 531 / 10607 (5.0 %)` and warns when selection falls outside the expected range.
It reports `Rfree unavailable` when the data lack a FreeR column.

### 5.2 MTZ and log R factors

The log gives R/Rfree of 0.3825/0.4091. Direct calculation from the output `F` and `FC_ALL` gives 0.3862/0.4106.
The log uses the observed scale, `Σ|Fo − k·|Fc||/ΣFo`, with a scale factor k for each reflection.
The MTZ writes absolute-scale values, including `F_abs = F/k` and `FC_ALL`. Calculating on the observed scale
reproduces 0.3825/0.4091. Numerical self-test 8 checks the algebra.
`check_outputs.py` allows a difference of 0.01 when comparing output files and the report.

---

## 6. Repackaging the deployment

1. Run `./pack_release.sh` to preserve paths, regenerate the checksum manifest, and verify archive extraction.
   Preserve the full directory layout when transferring the delivery.
2. The archive excludes `.venv/`, `__pycache__/`, and `*.so` (including `_simd.so`, rebuilt on the target),
   `linux_test_out/`, `dist/`, `.git/`.
3. Verify the manifest with `shasum -a 256 -c MANIFEST.sha256` on macOS before transfer.
4. On the target host, extract the tarball and enter it with `tar xzf oasis-refine-0.2.0.tar.gz && cd oasis-refine-0.2.0`, then run
   `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` →
   `./run_linux_test.sh`.

---

## 7. Validation checklist

- [x] Deployment integrity with bundled dictionary and on-demand SIMD build
- [x] Monomer restraint construction (684/915/117/242)
- [x] Ten numerical checks (Fc, −LL, X-ray and geometry finite differences, ML parameters, short refinement,
      MTZ columns, absolute scaling, kernel agreement, and FreeR conventions)
- [x] Seventeen unit tests (FreeR flags, FFT grids, kernels, R algebra, MTZ column detection, and absent FreeR column)
- [x] End-to-end short refinement from MTZ and PDB to PDB, MTZ, and JSON
- [x] Five-cycle run (39.4 s, matching NumPy cycle by cycle)
- [x] NumPy, fused, and SIMD agreement within 2.5e-11 for R (FFT removed in 0.3.0)
- [x] `pip install .` and the `minirefmac` and `minirefmac-selftest` commands
- [x] `check_outputs.py` verifies 11 required MTZ columns, finite values, and agreement when recalculating R
- [x] Output MTZ readable by CCP4 mtzdump (15 columns, P 21 21 21, 1.70–34.69 Å)
- [x] Output PDB readable with finite values and plausible B factors
- [x] Larger 3,056-atom dataset (20.6 s per cycle, correct automatic FreeR selection)
- [x] Launch from other working directories and with a custom PYTHONPATH
- [x] Archive creation, extraction, manifest verification, and package import
- [x] Fresh archive extraction and virtual environment with an on-demand SIMD build (five stages passed in 72 s)
