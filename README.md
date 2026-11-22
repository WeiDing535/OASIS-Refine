# minirefmac · oasis-refine

### Desktop interface

Run `./run.sh --gui` from a graphical desktop, or launch `minirefmac-gui` after installing with pip. Your Python installation must provide Tkinter.

Select an MTZ file, a PDB or mmCIF model, an output folder, and an optional job name. The **Weights and reflections** tab lists compatible MTZ labels for F, SIGF, and FreeR. Leave the selections at **(Auto)** to use automatic detection.

The **Progress** tab plots Rwork, Rfree, bond-angle RMSD, and bond-length RMSD from cycle 0 as refinement runs. The **Run log** tab shows engine messages. The buttons below the tabs open the output folder, PDB, MTZ, and JSON report.

Each run creates a separate folder with its parameters and log. Open **Job history** to review a previous run or fill the form with its saved parameters. The history lists runs in the selected output folder. Use **Browse folder...** to find jobs saved elsewhere.

By default, refinement fits the work set and saves the final cycle. To refine occupancies only in a final stage, run **All conformers · Fixed occupancy** first, then load the resulting PDB with the same original MTZ and select **All conformers · Refine occupancy**.

OASIS-Refine is a self-contained maximum-likelihood (ML) macromolecular crystallographic refinement engine with a CCP4 REFMAC-style interface and outputs. Its runtime dependencies are `gemmi`, `numpy`, and `scipy`. The distribution includes the monomer dictionary and compiles the optional SIMD kernel on demand.

- Version 0.2.0 requires Python ≥ 3.9 and was validated on Linux with Python 3.12.
- `minirefmac/data/monlib_bundle.zip` contains 35,639 monomer definitions (47.8 MB, LGPL-3.0).
- `test_data/7enm/` contains a P 21 21 21 dataset spanning 1.70–34.69 Å, with 11,057 reflections and a 670-atom model.
- See the [Linux installation and validation guide](README_LINUX.md).

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Validate the environment, installation, restraints, numerical checks and refinement outputs
./run_linux_test.sh                 # About 2 minutes
./run_linux_test.sh --kernels       # Also compare all available compute kernels

# Run refinement
./run.sh --mtz input.mtz --pdb input.pdb --out-dir out --ncyc 5 --maxiter 10
```

The run writes `out/refined.pdb`, `out/refined.mtz` on the absolute scale
(F/SIGF/FC/PHIC/FC_ALL/FWT/PHWT/DELFWT/PHDELWT/FOM), `out/report.json`, and `out/refine.log`.

---

## Refinement engine

| Component | Implementation |
|---|---|
| Objective | Maximum likelihood (Pannu & Read 1996). `-LL = Σ [ \|Fo - k·D·\|Fc\|\|² / (2σ²) + ln σ ]`. Estimates σ_A-type D and S by resolution bin. |
| Phase information | `--phase-prior` enables MLHL with `PHIB/FOM` or Hendrickson–Lattman `HLA..HLD` coefficients. |
| Scaling | Overall k, overall B, a traceless anisotropic B tensor, and bulk solvent k_sol/B_sol (mask or Babinet). |
| Bulk solvent | Molecular surface mask and solvent contribution `FC_ALL = FC·(1 - k_sol·exp(-B_sol s²))`. |
| Geometry restraints | Bundled monomer restraints for bonds, angles, planes, chirality, and torsions, plus symmetry-expanded van der Waals repulsion. |
| Optimizer | SciPy L-BFGS jointly refines coordinates and isotropic B factors with a maximum displacement limit. |
| Weighting | Balances X-ray and geometry gradient norms each cycle. The default `--weight-scale` is 0.05. |
| Structure factors | Direct summation via NumPy, fused NumPy, or AVX-512 C using shared ML coefficients. |
| Outputs | REFMAC-style MTZ columns and per-cycle R, Rfree, −LL, FOM, RMSD, and weights in `report.json`. |

### Compute kernels

The default `--kernel auto` selects `simd` when AVX-512F/DQ/VL and a C compiler are available. It falls back to `numpy`.
All kernels use the same ML coefficient G to calculate Fc and its gradient.
The `--kernels` test checks that R and Rfree differ by less than 1e-6. The measured difference was 2.5e-11.

| Kernel | 7enm (1 cycle × 1 iteration) | Notes |
|---|---|---|
| `numpy` | 16.6 s | Reference implementation, always available. |
| `fused` | 16.8 s | Fused phase calculations in NumPy, available without a compiler. |
| `simd` | 4.8 s | AVX-512 C kernel. The first run builds `_simd.so` with Clang or GCC. |

> Version 0.3.0 removed `fftcalc.py` because the FFT path traded accuracy and memory for little speed benefit.
> The implementation is available in Git history at commit `e8d0ab9`.

---

## Project layout

```
run.sh                  POSIX shell launcher
check_install.py        Checks dependencies, package paths, dictionary and license
check_outputs.py        Checks MTZ columns, finite PDB values, R factors and report consistency
run_linux_test.sh       Seven-stage Linux validation
pack_release.sh         Preserves layout, regenerates the manifest and verifies extraction
MANIFEST.sha256         File checksums
requirements.txt        gemmi / numpy / scipy
pyproject.toml          Package metadata and CLI, self-test and GUI entry points
tests/                  Standard-library unit tests
test_data/7enm/         Bundled test inputs
minirefmac/             Engine source code
  data/monlib_bundle.zip  Bundled monomer dictionary
  _simd.c                 AVX-512 source compiled on demand
  tools/build_monlib_bundle.py  Rebuilds the dictionary
README_LINUX.md         Linux setup, validation and troubleshooting
CHANGELOG.md            Release history
LINUX_TEST_REPORT.md    macOS-to-Linux validation report
```

---

## Known limitations

1. Refinement uses the waters in the input model. Automatic water placement and removal and TLS refinement are unavailable.
2. Per-atom B-factor refinement is isotropic. Overall anisotropic scaling is supported.
3. `best_rfree` records the lowest Rfree observed. The engine does not plot a free-R cross-validation curve.
4. The bundled dictionary covers standard amino acids, nucleic acids, and common ligands. Unknown residues use jelly-body restraints. Use `--no-monlib` to apply that fallback throughout.

---

## License

The source code belongs to its author. The bundled `minirefmac/data/monlib_bundle.zip` derives from the CCP4 monomer dictionary and is redistributed under LGPL-3.0.
See `minirefmac/data/LICENSE-CCP4-MONOMERS.txt` for the full license.
