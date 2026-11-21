# oasis-refine · Linux 测试说明书

本目录是一份**自包含**的 minirefmac 精修引擎部署包，用于在 Linux 机器上验证与运行。
它完全自包含：不需要任何外部晶体学套件、不需要 `refmac5`、不需要外部
`mon_lib_list.cif` 或任何单体 CIF 文件。所需的单体库约束已经打包进
`minirefmac/data/monlib_bundle.zip`（47.8 MB / 35639 个残基），运行时固定从这里读取，
没有任何覆盖开关。

运行时唯一的外部依赖是三个 PyPI 包：`gemmi`、`numpy`、`scipy`。

---

## 1. 目录内容

| 路径 | 作用 | 备注 |
|---|---|---|
| `run.sh` | 精修主入口，`./run.sh --mtz X --pdb Y --out-dir Z` | POSIX sh，默认 `python3` |
| `check_install.py` | 部署完整性自检（依赖 / 包路径 / 字典 / 许可证） | 可带 `--mtz --pdb` 追加真实数据检查 |
| `run_linux_test.sh` | **一键 Linux 测试**（完整性 → 约束 → 8 项数值自检 → 短精修） | 本次新增 |
| `README_LINUX.md` | 本说明书 | 本次新增 |
| `MANIFEST.sha256` | 全目录校验和，用于确认传输无损 | 本次新增 |
| `requirements.txt` | `gemmi>=0.6.5 / numpy>=1.20 / scipy>=1.7` | |
| `pyproject.toml` | 可选：`pip install .` 时用；**直接运行不需要** | |
| `minirefmac/*.py` | 引擎源码（dataio / sfcalc / likelihood / gradient / geometry / refine / maps / …） | 纯 Python |
| `minirefmac/data/monlib_bundle.zip` | **随包单体字典**，运行时默认从这里读 | 不可缺失 |
| `minirefmac/data/LICENSE-CCP4-MONOMERS.txt` | 字典数据的 LGPL-3.0 许可全文 | 不可缺失 |
| `minirefmac/_simd.c` | 可选 AVX-512 C 内核源码 | 仅 `--simd-kernel` 用，按需编译 |
| `minirefmac/tools/build_monlib_bundle.py` | **构建期**脚本，仅重建字典时才用 | 运行期不执行，`--src` 必填 |
| `test_data/7enm/input.mtz` + `input.pdb` | 随包迷你测试输入（315 KB） | 本次新增 |
| `linux_test_out/` | 测试脚本产生的日志与输出 | 运行时生成，非交付文件 |

> `tests/`、`benchmarks/`、旧的虚拟环境、构建产物都不在包内——本目录只放运行与验证所需文件。

### 随包测试数据

`test_data/7enm/` 来自 OASIS 案例 `7_7enm_3se_P212121`：
P 21 21 21，1.70–34.69 Å，11057 个反射（work 9713 / free 487），670 个原子。
它足够小（315 KB），但覆盖全部代码路径：对称展开、体溶剂 mask、ML 参数、
几何约束、MTZ/PDB 读写、绝对刻度输出。

---

## 2. Linux 环境要求

| 项 | 要求 |
|---|---|
| 系统 | 任意现代 Linux（glibc ≥ 2.17），x86_64 或 aarch64 |
| Python | 3.9 – 3.12（已验证 3.12.11）；需 `python3 -m venv`、`pip` |
| Python 包 | `gemmi>=0.6.5`、`numpy>=1.20`、`scipy>=1.7`（已验证 gemmi 0.7.5 / numpy 2.5.5 / scipy 1.18.1） |
| 外部晶体学套件 / 外部字典 | **完全不需要** |
| 编译器 | 不需要（除非要启用可选 SIMD 内核，届时从 `_simd.c` 按需编译） |
| 内存 | 7enm 测试 < 1 GB；跑 5hhk 级大数据建议 ≥ 8 GB |
| 磁盘 | 约 100 MB（包 48 MB + 输出） |

---

## 3. 三步安装并测试

把整个 `oasis-refine/` 目录原样拷到 Linux（保持相对路径与大小写，建议纯 ASCII 路径）。
若分发的是压缩包 `oasis-refine-linux-test.tar.gz`，先
`tar -xzf oasis-refine-linux-test.tar.gz && cd oasis-refine`。然后：

```bash
cd oasis-refine

# 1) 校验文件完整性（可选但推荐）
sha256sum -c MANIFEST.sha256          # 在 macOS 上校验时改用: shasum -a 256 -c MANIFEST.sha256

# 2) 装依赖（本机离线时可换成本地 wheelhouse / conda）
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3) 一键测试
chmod +x run.sh run_linux_test.sh
./run_linux_test.sh
```

脚本会自动优先使用 `.venv/bin/python`；也可用
`PYTHON=/path/to/python ./run_linux_test.sh` 指定解释器（例如 conda 环境）。

只想快速验证、跳过短精修：

```bash
./run_linux_test.sh --checks-only
```

> 脚本**不会**自动联网安装依赖：检测到缺包时会打印上面的 venv 命令并退出。

---

## 4. 预期输出与通过判据

一键脚本共 4 个阶段，全部通过时最后打印 `ALL LINUX CHECKS PASSED`。
以下为实测（Python 3.12.11 / gemmi 0.7.5 / numpy 2.5.5 / scipy 1.18.1）：

| 阶段 | 通过判据 | 实测 |
|---|---|---|
| 1/4 完整性 | `external dictionary paths in sys.path : NONE`；`RESULT : PASS` | 35639 residues，字典 47.8 MB，licence present |
| 2/4 真实数据约束 | `RESULT : PASS`，约束计数非零 | `684 bonds / 915 angles / 117 planes / 242 torsions`，`f=7261.923272`，`rmsd_bond=0.018109` |
| 3/4 数值自检 | `SELFTEST PASS`，8 项全过 | `Fc vs gemmi 3.59e-08`；X-ray 梯度 FD `1.48e-08`；几何梯度 FD `3.29e-08`；约 1–2 min |
| 4/4 短精修 | R 与 Rfree 不升高，输出 3 个文件 | `R 0.3830→0.3825`，`Rfree 0.4116→0.4091`，`rmsBOND 0.0181→0.0091`，约 25 s |

日志与产物写在 `linux_test_out/`（**每次运行会清空重建该目录**，重要结果请先另存）：

```
linux_test_out/check_install.txt   完整性自检输出
linux_test_out/restraints.txt      真实数据约束输出
linux_test_out/selftest.log        8 项自检日志
linux_test_out/selftest.json       自检结果（机器可读）
linux_test_out/selftest/*          自检写出的 MTZ/PDB
linux_test_out/refine.log          短精修日志
linux_test_out/refine/refined.pdb  / refined.mtz / report.json
```

不同 CPU / BLAS / numpy 版本下末位数值可能有微小差异；**判据是 8 项 PASS 且 R/Rfree 不升高**，
不要求与上表逐位一致。

---

## 5. 手动分步复现（定位问题时用）

```bash
cd oasis-refine
PY=python3                      # 或 .venv/bin/python

$PY check_install.py
$PY check_install.py --mtz test_data/7enm/input.mtz --pdb test_data/7enm/input.pdb
$PY -m minirefmac.selftest --mtz test_data/7enm/input.mtz \
    --pdb test_data/7enm/input.pdb --out-dir /tmp/mr_selftest \
    --json /tmp/mr_selftest.json
PYTHON=$PY ./run.sh --mtz test_data/7enm/input.mtz \
    --pdb test_data/7enm/input.pdb --out-dir /tmp/mr_refine \
    --ncyc 1 --maxiter 3
$PY -m minirefmac.cli --help
```

---

## 6. Linux 专属注意事项（重要）

1. **包内不再附带任何预编译二进制。** 原来的 macOS Mach-O `_simd.so` 已删除，
   Linux 上不会出现「invalid ELF header」一类问题。
   `--simd-kernel` 仍可用，但会由 `simdkernel.py` 从 `_simd.c` **按需自动编译**
   （优先 `clang`，回退 `gcc`；需要 CPU 支持 AVX-512F/DQ/VL 且有 C 编译器）。
   没有编译器或 CPU 不支持时会打印提示并禁用该内核，不会崩溃。
   **推荐做法：Linux 上直接不用这个开关**，纯 Python 路径已足够快。

2. **平台无关的加速开关可以正常使用**：`--fused-kernel`（numpy 融合内核）、
   `--fft-kernel`（gemmi FFT 结构因子，可配 `--fft-spacing` / `--fft-radius`）。
   它们不依赖任何本地二进制。

3. **代码里没有任何本机绝对路径，也没有外部字典开关。**
   `tools/build_monlib_bundle.py` 的 `--src` 现在是必填参数（构建期才用），
   示例用 `/path/to/monomers` 占位；运行期顶层模块不会 `gemmi.cif.read` 外部 CIF，
   也不接受任何外部字典路径参数。

4. **`--monlib` 已删除**：字典固定从包内 `monlib_bundle.zip` 读取，没有覆盖入口。
   需要更新字典时用 `tools/build_monlib_bundle.py --src <源目录>` 重建 zip 再替换。
   `--no-monlib` 仍保留，那是禁用字典、改用 jelly-body fallback 的内部功能开关。

5. **文件传输细节**：脚本与 Python 源文件均为 LF 行尾、UTF-8/ASCII；解压后若丢失执行位，
   执行 `chmod +x run.sh run_linux_test.sh`。Linux 文件系统大小写敏感，
   请保持 `test_data/7enm/input.mtz` 这类路径原样。

6. **字典许可**：`monlib_bundle.zip` 是源化学组分字典的派生数据，按 LGPL-3.0 随包再分发，
   许可全文见 `minirefmac/data/LICENSE-CCP4-MONOMERS.txt`。请勿删除该文件
   （文件名含来源缩写属许可合规要求；除此之外包内代码已无相关字样）。

---

## 7. 正式精修用法

```bash
./run.sh --mtz /path/to/input.mtz --pdb /path/to/input.pdb \
         --out-dir out --ncyc 5 --maxiter 10 --solvent mask \
         --map-scale absolute

# 大数据可开 FFT 结构因子内核（纯 gemmi，Linux 安全）
./run.sh --mtz big.mtz --pdb big.pdb --out-dir out --fft-kernel --ncyc 5
```

输入 MTZ 需含观测振幅、sigma、FreeR 标志列（列名自动识别，也可用 `--fp/--sigfp/--free` 指定）；
输出为 `refined.pdb`、`refined.mtz`（含绝对刻度与 FC/PHIC/FOM 等列）、`report.json`。
完整参数见 `python3 -m minirefmac.cli --help`。

---

## 8. 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `Missing gemmi / numpy / scipy` | 未装依赖 | 按第 3 节建 venv 并 `pip install -r requirements.txt` |
| `ModuleNotFoundError: gemmi` | 用了系统 `python3` 但包装在 venv 里 | 用 `.venv/bin/python`，或设 `PYTHON=...` |
| `FAIL: bundled dictionary missing/cannot be opened` | 拷贝不完整或 zip 损坏 | `sha256sum -c MANIFEST.sha256` 定位，重新拷贝 |
| `FAIL: minirefmac was imported from outside this deployment` | `PYTHONPATH` 里有另一份 minirefmac | 清空 `PYTHONPATH`，在部署目录内执行 |
| `SELFTEST FAIL` | 数值检查未过 | 看 `linux_test_out/selftest.json` 的 `checks`，定位是哪一项 |
| `simdkernel: no C compiler found` / `build failed` | 加了 `--simd-kernel` 但缺编译器或 CPU 不支持 AVX-512 | 去掉该开关（默认路径不受影响），或装好 clang/gcc 后重试 |
| `OptimizeWarning: Unknown solver options: disp` | 旧版 scipy 兼容问题 | 本包已移除该选项；若仍出现说明用了旧代码，重新拷贝 |
| `Permission denied` | 解压丢了执行位 | `chmod +x run.sh run_linux_test.sh` |
| `R/Rfree 明显变差` | 输入/列名不匹配或 `--free-value` 设错 | 检查 MTZ 列标签与 free 标志取值（默认 0 为 free） |

---

## 9. 一句话结论

在 Linux 上，只要 `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
再执行 `./run_linux_test.sh` 得到 `ALL LINUX CHECKS PASSED`，就说明
**字典、几何约束、结构因子、梯度、ML 目标、精修与输出全部在完全自包含的环境下工作正常**。
