# Changelog

## 0.2.0 — Linux validation and deployment improvements

The Linux validation exposed several silent correctness issues and gaps in deployment tooling.
This release fixes them and adds an automated acceptance test.
The changes were tested on Ubuntu 26.04, x86_64, 96 cores, and Python 3.12.15.

### Correctness

- **FreeR flags.** `--free-value` now defaults to `auto`. The old default of
  `0` treated 95% of the reflections as free for files with `0=work, 1=free`, such as 4L6R.
  It silently reported a meaningless Rfree of 0.3255. The new rule uses flag 0 if it represents
  1–25% of reflections. Otherwise, it picks the flag in that range closest to 5%, or warns and falls back to 0.
  A free set above 30% triggers another warning. The startup log, the `free` field in `report.json`,
  and MTZ history record the selected value and proportion.
- **MTZ without a FreeR column.** The reader sets `mode=absent` and issues a warning via
  `warnings.warn`. Rfree is unavailable.
- **Shared ML coefficient G.** The fused and SIMD paths formerly reused G from a previous warm start.
  The resulting inaccurate gradients diverged from NumPy by about 4e-4 in R and Rfree.
  NumPy, fused, SIMD, and FFT now use the same G, with a measured R and Rfree difference of 2.5e-11.
- **FFT resolution.** The original fixed 0.5 Å `auto_spacing` yielded 3.5% relative Fc error
  versus the documented 0.1–0.2%, raising R by around 0.004. Adaptive spacing yields about 0.234 Å for
  typical protein B factors. The calculation floors B at 3 Å² to avoid a 1000³ grid for one sharp atom.
  Fc error dropped to 1.1e-3 and R agrees with direct summation to four decimal places.
  The `vmax` memory cap remains in place and the log prints grid dimensions and estimated memory use.
- **MTZ column semantics.** `FC/PHIC` contain atomic contributions and `FC_ALL` includes bulk solvent.
  A column-by-column comparison against a local REFMAC5 run verified this convention.
- **Input diagnostics.** Invalid sigma values are replaced by the median. `data_stats` counts these values,
  missing or nonpositive F, duplicate HKLs, alternate conformers, and hydrogens. The startup log reports them as applicable.
- The kernels use the real part of `fc_atom_factor` explicitly, avoiding a `ComplexWarning` from implicit truncation.

### Interface

- Added `--kernel {auto,numpy,fused,simd,fft}`. `auto` uses SIMD when AVX-512F/DQ/VL and a C compiler
  are available and otherwise selects NumPy. `--simd-kernel`, `--fused-kernel`, and `--fft-kernel` are
  deprecated aliases. Invalid values raise an error and unsuccessful detection falls back to NumPy.
- Added `--version` and `python -m minirefmac` (`minirefmac/__main__.py`).
- Grouped CLI options and added separate exit codes for load failure (2) and refinement failure (3).
  These failures now produce concise error messages.
- `pyproject.toml` declares Python ≥ 3.9 and includes README, keywords, classifiers, and license metadata.
  Added the `minirefmac-selftest` console entry point.
- `minirefmac.__version__` supplies the version to `report.json` and MTZ history.

### Tests and tools

- `tests/test_units.py` adds 17 standard-library tests (7 s) covering three FreeR conventions,
  FFT grid rules, kernel consistency, R algebra (perfect Fc yields R=0), automatic MTZ column selection,
  and MTZ files with no FreeR column.
- `minirefmac/selftest.py` expands from eight to ten checks, adding kernel agreement with shared G
  and FreeR convention detection.
- `check_outputs.py` validates outputs directly. It checks 11 required MTZ columns,
  unchanged space group, cell, and reflection count, no NaNs, nonzero `|FWT|`, phases in [0, 360],
  and R recalculated from files within 0.01 of `report.json`. It also checks finite PDB coordinates, B, and occupancy,
  B within `[bmin,bmax]`, and report cycle count equal to `ncyc`.
- `run_linux_test.sh` expands from four to seven stages covering environment, integrity, restraints, unit tests,
  numerical checks, short refinement, and output checks, with an optional eighth kernel check.
  Added `--checks-only`, `--kernels`, and `--python PATH`. `summary.txt` records results and timings, and `env.txt`
  records the environment. The script stops on failure and prints the relevant log.
- `pack_release.sh` preserves the package layout and excludes `.venv/`, `__pycache__/`, `*.so`,
  `linux_test_out/`, `dist/`, and `.git/`. It verifies required files, regenerates `MANIFEST.sha256`,
  archives and re-extracts the files, verifies checksums, and tests `import minirefmac`.
  This prevents the package layout from being flattened by operations such as `zip -j` or Finder compression (which previously caused
  `No module named 'minirefmac'`).

### Performance (7enm, 1 cycle × 1 iteration)

| Kernel | Time | Notes |
|---|---|---|
| numpy | 18.6 s | Reference implementation |
| fused | 18.4 s | Available without a compiler |
| simd | 5.0 s | 3.7× as fast as NumPy. Builds `_simd.so` on first use. |
| fft | 44 s | At 0.234 Å grid spacing, slower on the test host. Useful only for very large reflection sets. |

### Documentation

- `README.md` covers the engine, kernels, layout, and known limitations.
- `README_LINUX.md` covers the seven validation stages, ten numerical checks,
  MTZ scaling and R factors, FreeR conventions, packaging, and troubleshooting.
- Added `CHANGELOG.md`.

---

## 0.1.0 — Initial release

- Self-contained ML refinement engine with direct-sum structure factors, resolution-binned D/S, mask and Babinet bulk solvent,
  monomer restraints, L-BFGS, automatic weighting, and REFMAC-style MTZ output.
- Optional AVX-512 C, fused NumPy, and FFT compute paths.
- `check_install.py`, four-stage `run_linux_test.sh`, and eight-check `minirefmac.selftest`.
