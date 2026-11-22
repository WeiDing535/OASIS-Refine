# OASIS-Refine Linux Guide

### Desktop interface

Run `./run.sh --gui` from a graphical desktop, or launch `minirefmac-gui` after installing with pip. Your Python installation must provide Tkinter.

Select an MTZ file, a PDB or mmCIF model, an output folder, and an optional job name. The **Weights and reflections** tab lists compatible MTZ labels for F, SIGF, and FreeR. Leave the selections at **(Auto)** to use automatic detection.

The **Progress** tab plots Rwork, Rfree, bond-angle RMSD, and bond-length RMSD from cycle 0 as refinement runs. The **Run log** tab shows engine messages. The buttons below the tabs open the output folder, PDB, MTZ, and JSON report.

Each run creates a separate folder with its parameters and log. Open **Job history** to review a previous run or fill the form with its saved parameters. The history lists runs in the selected output folder. Use **Browse folder...** to find jobs saved elsewhere.

By default, refinement fits the work set and saves the final cycle. To refine occupancies only in a final stage, run **All conformers · Fixed occupancy** first, then load the resulting PDB with the same original MTZ and select **All conformers · Refine occupancy**.

This self-contained deployment uses the monomer restraints in `minirefmac/data/monlib_bundle.zip` (47.8 MB, 35,639 entries). The only external runtime dependencies are the PyPI packages `gemmi`, `numpy`, and `scipy`.

---

## 1. Project layout

| Path | Purpose | Notes |
|---|---|---|
| `run.sh` | Refinement launcher, `./run.sh --mtz X --pdb Y --out-dir Z` | POSIX sh with `python3` fallback |
| `check_install.py` | Checks dependencies, package paths, dictionary, and license | Add `--mtz --pdb` for checks with real data |
| `check_outputs.py` | Checks MTZ columns, finite PDB values, recalculated R, and report consistency | Works independently of the log |
| `run_linux_test.sh` | Seven-stage Linux validation (Section 4) | `--checks-only` / `--kernels` |
| `pack_release.sh` | Creates a tar.gz with the correct layout and verifies the regenerated manifest | See Section 8 |
| `MANIFEST.sha256` | Checksums for verifying a transferred copy | |
| `requirements.txt` | `gemmi>=0.6.5 / numpy>=1.20 / scipy>=1.7` | |
| `pyproject.toml` | `pip install .` provides `minirefmac`, `minirefmac-selftest`, and `minirefmac-gui` | Optional for direct invocation |
| `tests/test_units.py` | Standard-library unit tests (17 tests in the original validation, about 7 s) | Free flags, FFT grids, kernel agreement, R algebra, MTZ column detection |
| `minirefmac/*.py` | Engine source (dataio, sfcalc, likelihood, gradient, geometry, refine, maps, etc.) | Python |
| `minirefmac/data/monlib_bundle.zip` | Bundled monomer dictionary used at runtime | Required |
| `minirefmac/data/LICENSE-CCP4-MONOMERS.txt` | Full LGPL-3.0 license for the dictionary data | Required |
| `minirefmac/_simd.c` | Optional AVX-512 C kernel source | Compiled on demand |
| `minirefmac/tools/build_monlib_bundle.py` | Rebuilds the dictionary at build time | Requires `--src` |
| `test_data/7enm/input.mtz` and `input.pdb` | Bundled test inputs (315 KB) | |
| `linux_test_out/` | Validation logs and outputs | Generated at runtime |

### Bundled test data

`test_data/7enm/` comes from the OASIS case `7_7enm_3se_P212121`.
Space group P 21 21 21, resolution 1.70–34.69 Å, 11,057 reflections (9,713 work and 487 free), 670 atoms.
This 315 KB dataset exercises symmetry expansion, the bulk-solvent mask, ML parameters,
geometry restraints, MTZ and PDB I/O, and absolute-scale output.

---

## 2. Linux requirements

| Item | Requirement |
|---|---|
| OS | Modern Linux with glibc ≥ 2.17, on x86_64 or aarch64 |
| Python | ≥ 3.9 with `venv` and `pip` (tested on 3.12.15) |
| Python packages | `gemmi>=0.6.5`, `numpy>=1.20`, `scipy>=1.7` (tested with 0.7.5, 2.5.5, and 1.18.1 respectively) |
| External crystallography suite and dictionary | None required |
| Compiler | Clang or GCC to build the optional SIMD kernel on first use |
| Memory | Under 1 GB for 7enm. At least 4 GB recommended for about 3,000 atoms at 3 Å. |
| Disk space | About 100 MB, including the 48 MB package and outputs |

---

## 3. Installation and validation

Copy the whole directory to Linux, preserving relative paths and filename case. ASCII paths are recommended.
For the distributed archive, extract it and enter the resulting directory with
`tar -xzf oasis-refine-0.2.0.tar.gz && cd oasis-refine-0.2.0`. Then follow the steps below.

```bash
cd oasis-refine-0.2.0

# 1) Verify file integrity (optional)
sha256sum -c MANIFEST.sha256     # On macOS use shasum -a 256 -c MANIFEST.sha256

# 2) Install dependencies (use a local wheelhouse or Conda if offline)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3) Run validation
chmod +x run.sh run_linux_test.sh pack_release.sh
./run_linux_test.sh
```

The script can use `.venv/bin/python`. To select a specific interpreter, such as a Conda environment, use
`PYTHON=/path/to/python ./run_linux_test.sh` or `--python PATH`.
Set `MINIREFMAC_TEST_OUT=/absolute/path` to direct logs and outputs to a separate directory.
`--kernels` reports unavailable SIMD and verifies which kernel each run actually used.

Other validation modes

```bash
./run_linux_test.sh --checks-only    # Skip end-to-end refinement, about 75 s
./run_linux_test.sh --kernels        # Compare NumPy, fused, and supported SIMD kernels
```

> When a dependency is missing, the script prints the virtual environment setup commands and exits.

---

## 4. Expected outputs and pass criteria

A successful seven-stage run prints `ALL LINUX CHECKS PASSED` and records
per-stage results and timings in `linux_test_out/summary.txt`.
The measurements below used Ubuntu 26.04, x86_64, 96 cores, Python 3.12.15, gemmi 0.7.5, NumPy 2.5.5, and SciPy 1.18.1.

| Stage | Pass criterion | Observed result |
|---|---|---|
| 0/7 Environment | Python ≥ 3.9 and successful imports of gemmi, NumPy, and SciPy | 3.12.15, `auto → simd (prebuilt _simd.so)` |
| 1/7 Integrity | `external dictionary paths in sys.path : NONE` and `RESULT : PASS` | 35,639 residues, 47.8 MB dictionary, license present |
| 2/7 Restraints | `RESULT : PASS` and nonzero restraint counts | `684 bonds / 915 angles / 117 planes / 242 torsions`, `f=7261.923272` |
| 3/7 Unit tests | `OK` (17 tests) | 7 s |
| 4/7 Numerical self-test | `SELFTEST PASS` for all 10 checks | Fc vs gemmi 3.88e-08, X-ray gradient FD 1.57e-08, geometry gradient FD 3.29e-08, kernel agreement fused 4e-16 / SIMD 1e-14, 61 s |
| 5/7 Short refinement | Prints `RESULT` with R and Rfree no higher than their initial values | `R 0.3830→0.3825`, `Rfree 0.4116→0.4091`, `rmsBOND 0.0181→0.0091`, 7 s |
| 6/7 Output checks | `RESULT : PASS` with 11 required columns, unchanged space group and cell, no NaNs, and consistent R | 15 MTZ columns, `R(MTZ)=0.3862` and `report=0.3825` (difference 0.0037, see Section 6.2) |
| 7/7 Kernel agreement (optional) | R and Rfree differ by < 1e-6 | `R=0.382630 Rfree=0.409866` for all three, a difference of 2.5e-11, timings 18.6 / 18.4 / 5.0 s |

Each run writes to `linux_test_out/` and recreates the `selftest/`, `refine/`, and `kernels/` subdirectories.

```
linux_test_out/env.txt             Interpreter, dependencies, host, and kernel detection
linux_test_out/summary.txt         Stage results and timings
linux_test_out/check_install.txt   Integrity check
linux_test_out/restraints.txt      Real-data restraint check
linux_test_out/unit_tests.log      Unit test log
linux_test_out/selftest.log       Ten numerical checks
linux_test_out/selftest.json      Machine-readable self-test results
linux_test_out/selftest/*          Self-test MTZ and PDB outputs
linux_test_out/refine.log          Short refinement log
linux_test_out/refine/refined.pdb / refined.mtz / report.json
linux_test_out/check_outputs.txt   Output validation
linux_test_out/kernels/{numpy,fused,simd}/   Kernel comparison runs (--kernels)
```

The last digits can vary with CPU, BLAS, and NumPy. Validation requires passing each stage
and finishing with R and Rfree at or below their initial values.

### 4.1 Numerical self-test

| # | Check | Observed result |
|---|---|---|
| 1 | Maximum relative Fc error against gemmi's independent implementation | 3.88e-08 |
| 2 | Agreement between `negll_only` and `negll_and_G` | 0 |
| 3 | X-ray gradient finite differences | 1.57e-08 |
| 4 | Geometry gradient finite differences | 3.29e-08 |
| 5 | ML parameters in their physical ranges | D 0.754–0.999, S 740–32187 |
| 6 | Decrease in negative log-likelihood after short refinement | 92070.9 → 91476.6 |
| 7 | Readable MTZ with required columns | 15 columns, `missing=[]` |
| 8 | REFMAC-style absolute scaling | `rel\|F_out·k − F_in\| = 3.33e-08`, `mean\|FC−FC_ALL\| = 11.9` |
| 9 | Fc and gradient agreement across kernels with the same G | fused 4e-16 / SIMD 1e-14 |
| 10 | FreeR flag detection (0=free, 0=work, and single-value fallback) | All three passed |

---

## 5. Manual validation for troubleshooting

```bash
cd oasis-refine-0.2.0
PY=.venv/bin/python              # Or python3

$PY check_install.py
$PY check_install.py --mtz test_data/7enm/input.mtz --pdb test_data/7enm/input.pdb
$PY -m unittest discover -s tests -v
$PY -m minirefmac.selftest --mtz test_data/7enm/input.mtz \
    --pdb test_data/7enm/input.pdb --out-dir /tmp/mr_selftest \
    --json /tmp/mr_selftest.json
$PY -m minirefmac --mtz test_data/7enm/input.mtz \
    --pdb test_data/7enm/input.pdb --out-dir /tmp/mr_refine \
    --ncyc 1 --maxiter 3
$PY check_outputs.py --mtz test_data/7enm/input.mtz \
    --pdb test_data/7enm/input.pdb --out-dir /tmp/mr_refine --ncyc 1
$PY -m minirefmac --help
```

---

## 6. Linux runtime notes

### 6.1 Compute kernels and refinement options

The archive ships with SIMD source and builds `_simd.so` on the target machine to avoid
platform-specific loader errors such as `invalid ELF header`.

The default `--kernel auto` selects SIMD when the CPU supports AVX-512F/DQ/VL and FMA and the kernel can be loaded or compiled.

```bash
./run.sh --mtz X.mtz --pdb X.pdb --out-dir out            # Automatic kernel selection
./run.sh ... --kernel numpy                               # NumPy reference implementation
./run.sh ... --kernel simd                                # Build _simd.so on first use
./run.sh ... --kernel fused                               # Fused NumPy, no compiler required
```

- `--simd-kernel` and `--fused-kernel` are aliases for their respective `--kernel` values.
- Version 0.3.0 removed the FFT path because of grid error and memory cost.
- CPU incompatibility, compilation failure, or load failure is logged before falling back to NumPy. Set `MINIREFMAC_STRICT_KERNEL=1` to make an explicit SIMD request fail when SIMD is unavailable.
- If a previously built `_simd.so` cannot be loaded, the program attempts to rebuild it from the bundled source.

`rmsANGL` in the log and `rmsd_angle_deg` in `report.json` give the RMSD from dictionary three-atom bond angles in degrees. The per-cycle `rmsd_angle` reports the same quantity in radians. Earlier versions used that field for 1–3 distance deviations. The log lists the detected disulfide bonds. Explicit disulfides within the asymmetric unit (PDB `SSBOND` or mmCIF `_struct_conn`) take precedence at any starting distance. Other CYS SG contacts are restrained only if the initial distance is at most 2.5 Å and the pairing is unique.

On AVX-512 Linux hosts, `--kernel simd --simd-threads 4` parallelizes atom blocks. The default is one thread. Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` to avoid oversubscribing cores with other thread pools. `--simd-threads` applies only to SIMD.

Use `--restraint-cif ligand.cif` to supply component dictionaries for unknown ligands. Repeat the option for multiple files. Component IDs match residue names and override matching bundled components. The option supports intracomponent geometry. General ligand–protein LINK restraints are currently unavailable.

`--all-altlocs` includes alternate conformers in Fcalc at their input occupancies and refines individual coordinates and isotropic B factors. Shared atoms take part in each conformer's geometry restraints, with shared terms counted once. Mutually exclusive conformers of the same residue are exempt from intramolecular ASU nonbonded repulsion. This mode fixes occupancies and expects one input model with the same chemical identity at a given residue position.

`--refine-occupancies` extends this mode to grouped occupancy refinement. A supported residue has two conformers with matching atom-name sets. Within each conformer, the occupancy must be uniform and representable to two decimal places in PDB. Refinement preserves the pair's original total occupancy and keeps other occupancies fixed. The occupancy update follows the coordinate and B-factor update in each cycle and optimizes likelihood on work reflections only. The selected cycle's occupancies are used in both PDB and MTZ outputs. Use `--no-auto-weight --no-keep-best --early-stop off` to reserve FreeR for independent validation.

With occupancy refinement enabled, JSON reports PHE and TYR RMSDs for direct atom-name matching and for connectivity-validated equivalent aromatic ring swaps. The group is flagged as nearly overlapping when the smaller RMSD is at most 0.25 Å. `--freeze-overlapping-occupancies` fixes the occupancies of groups flagged at the start of the stage while still refining coordinates and B factors. `--occupancy-prior-sigma 0.25` adds a Gaussian prior centered on stage input occupancies. The log reports the work-set negative log-likelihood and the objective including the prior separately. Both options require `--refine-occupancies`.

### 6.2 MTZ and log R factors

The log computes R on the observed scale with a per-reflection scale factor k.
`R = Σ|Fo − k·|Fc|| / ΣFo`. The written MTZ follows the REFMAC absolute-scale convention
with `F_out = Fo / k_total` and an absolute-scale `FC_ALL`.
Computing R directly from these two columns typically gives a value about 0.002–0.004 higher.
Reapplying the per-reflection scale reproduces the log value, as checked by self-test 8.

### 6.3 FreeR flag conventions

The default `--free-value auto` examines the flag histogram.

1. If flag `0` accounts for 1–25% of observed reflections, select it as free under the CCP4 convention.
2. Otherwise select the flag within 1–25% closest to 5%. This supports files with `0=work, 1=free`.
3. If neither rule applies, use `0` and issue a warning. A free set above 30% triggers an additional warning.

Use `--free-value 1` to override the auto selection. Check the startup log and `report.json` for the
selected flag and proportion, shown in the log as `free set: <column>=<value> -> N of M (x.x %)`.

### 6.4 Preserving the package layout

Flattening the original delivery lost four directory levels and caused two `__init__.py` files to collide.
The resulting Linux import failed with `No module named 'minirefmac'`. Use `pack_release.sh`
to preserve the directory layout, regenerate the manifest, and verify extraction.

### 6.5 Additional notes

- `tools/build_monlib_bundle.py` requires `--src` when rebuilding the bundled dictionary.
  The engine reads the bundled dictionary by default and can load explicit ligand restraints as described in Section 6.1.
- `--monlib` has been removed. `--no-monlib` disables the bundled dictionary
  and switches to jelly-body restraints.
- Scripts and Python source use LF line endings and UTF-8 or ASCII. To restore executable permissions after extraction, run
  `chmod +x run.sh run_linux_test.sh pack_release.sh`.
- `monlib_bundle.zip` derives from the source chemical component dictionary and is redistributed under LGPL-3.0.
  Keep the license in `minirefmac/data/LICENSE-CCP4-MONOMERS.txt` with the distribution.

---

## 7. Running refinement

```bash
./run.sh --mtz /path/to/input.mtz --pdb /path/to/input.pdb \
         --out-dir out --ncyc 5 --maxiter 10 --solvent mask --map-scale absolute

# Set column names, resolution limits, and the free flag
./run.sh --mtz in.mtz --pdb in.pdb --out-dir out \
         --fp F --sigfp SIGF --free FreeR_flag --free-value 1 \
         --dmin 2.5 --dmax 50

# Include experimental phases (MLHL)
./run.sh ... --phib PHIB --fom FOM --phase-prior
```

Supply observed amplitudes, sigmas, and FreeR flags in the MTZ. Column names can be detected automatically or set with `--fp`, `--sigfp`, and `--free`.
The engine writes `refined.pdb`, `refined.mtz` (absolute scale with FC/PHIC/FC_ALL/FWT/DELFWT/FOM and related columns),
and `report.json`. Run `python3 -m minirefmac --help` for all options.

The 3.30 Å dataset with 3,056 atoms and 10,607 reflections took about 100 s per cycle in NumPy.
On 7enm (1.70 Å, 670 atoms), a cycle took about 3–19 s depending on the kernel.

---

## 8. Packaging and distribution

```bash
./pack_release.sh                 # Create dist/oasis-refine-0.2.0.tar.gz and verify it
./pack_release.sh --zip           # Also create a ZIP file
```

The script stages the files with their original paths and excludes `.venv/`, `__pycache__/`, `*.so`,
`linux_test_out/`, `dist/`, and `.git/`. It checks required files, regenerates `MANIFEST.sha256`,
creates and re-extracts the archive, verifies checksums, and checks `import minirefmac`.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Missing gemmi / numpy / scipy` | Missing dependencies | Create a virtual environment and install `requirements.txt` as shown in Section 3 |
| `ModuleNotFoundError: gemmi` | System Python used with packages installed in a virtual environment | Use `.venv/bin/python` or set `PYTHON=...` |
| `No module named 'minirefmac'` | Flattened layout or wrong working directory | Repackage with `pack_release.sh` and enter the directory containing `minirefmac/` |
| `FAIL: bundled dictionary missing/cannot be opened` | Incomplete copy or corrupt ZIP | Run `sha256sum -c MANIFEST.sha256` and copy the affected files again |
| `FAIL: minirefmac was imported from outside this deployment` | Another copy appears in `PYTHONPATH` | Clear `PYTHONPATH` and run from the deployment directory |
| `SELFTEST FAIL` | A numerical check failed | Inspect `checks` in `linux_test_out/selftest.json` |
| `simdkernel: no C compiler found` or `build failed` | The requested SIMD kernel cannot be built or run | Use `--kernel auto` to allow NumPy fallback |
| R or Rfree worsens significantly | Input or column mismatch, or wrong free flag | Check the startup free-set line and `free` and `data_stats` in `report.json` |
| Rfree nearly equals R | Nearly all reflections were assigned to the free set | Set `--free-value` explicitly |
| `Permission denied` | Extraction lost executable permissions | Run `chmod +x run.sh run_linux_test.sh pack_release.sh` |

---

## 10. Validation summary

On Linux, install the dependencies with `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
and run `./run_linux_test.sh`. `ALL LINUX CHECKS PASSED` confirms the bundled dictionary, geometry, structure factors,
gradients, ML objective, refinement, and outputs work in the self-contained deployment.
Use `--kernels` to verify refinement agreement across the available kernels.
