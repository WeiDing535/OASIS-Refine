# oasis-refine · Linux 验收与使用说明书

本目录是一份**自包含**的 minirefmac 精修引擎部署包，用于在 Linux 上验证与运行。
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
| `check_outputs.py` | 精修产物自检（MTZ 列 / PDB 有限性 / R 复算 / report 一致性） | 不依赖程序自身日志 |
| `run_linux_test.sh` | **一键 Linux 验收**（7 阶段，见第 4 节） | `--checks-only` / `--kernels` |
| `pack_release.sh` | 打包含**正确目录布局**的 tar.gz + 重生成 MANIFEST + 解包复验 | 见第 8 节 |
| `MANIFEST.sha256` | 全目录校验和，用于确认传输无损 | |
| `requirements.txt` | `gemmi>=0.6.5 / numpy>=1.20 / scipy>=1.7` | |
| `pyproject.toml` | 可选：`pip install .` 后得到 `minirefmac`、`minirefmac-selftest` 命令 | 直接运行不需要 |
| `tests/test_units.py` | 单元测试（stdlib unittest，17 项，约 7 s） | 覆盖 free 标志、FFT 网格、内核一致性、R 代数、列名识别 |
| `minirefmac/*.py` | 引擎源码（dataio / sfcalc / likelihood / gradient / geometry / refine / maps / …） | 纯 Python |
| `minirefmac/data/monlib_bundle.zip` | **随包单体字典**，运行时默认从这里读 | 不可缺失 |
| `minirefmac/data/LICENSE-CCP4-MONOMERS.txt` | 字典数据的 LGPL-3.0 许可全文 | 不可缺失 |
| `minirefmac/_simd.c` | 可选 AVX-512 C 内核源码 | 首次用 `--kernel simd` 时按需编译 |
| `minirefmac/tools/build_monlib_bundle.py` | **构建期**脚本，仅重建字典时才用 | 运行期不执行，`--src` 必填 |
| `test_data/7enm/input.mtz` + `input.pdb` | 随包迷你测试输入（315 KB） | |
| `linux_test_out/` | 测试脚本产生的日志与输出 | 运行时生成，非交付文件 |

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
| Python | **≥ 3.9**（已在 3.12.15 验证）；需 `python3 -m venv`、`pip` |
| Python 包 | `gemmi>=0.6.5`、`numpy>=1.20`、`scipy>=1.7`（已验证 gemmi 0.7.5 / numpy 2.5.5 / scipy 1.18.1） |
| 外部晶体学套件 / 外部字典 | **完全不需要** |
| 编译器 | 不需要；只有想用 `--kernel simd` 时才需要 clang/gcc（首次运行自动编译） |
| 内存 | 7enm 测试 < 1 GB；真实规模（3 Å、3000 原子级）建议 ≥ 4 GB |
| 磁盘 | 约 100 MB（包 48 MB + 输出） |

---

## 3. 三步安装并测试

把整个目录原样拷到 Linux（保持相对路径与大小写，建议纯 ASCII 路径）。
若分发的是压缩包 `oasis-refine-0.2.0.tar.gz`，先
`tar -xzf oasis-refine-0.2.0.tar.gz && cd oasis-refine-0.2.0`。然后：

```bash
cd oasis-refine-0.2.0

# 1) 校验文件完整性（可选但推荐）
sha256sum -c MANIFEST.sha256     # macOS 上改用: shasum -a 256 -c MANIFEST.sha256

# 2) 装依赖（离线时可换成本地 wheelhouse / conda）
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3) 一键验收
chmod +x run.sh run_linux_test.sh pack_release.sh
./run_linux_test.sh
```

脚本会自动优先使用 `.venv/bin/python`；也可用
`PYTHON=/path/to/python ./run_linux_test.sh` 或 `--python PATH` 指定解释器（例如 conda 环境）。

常用组合：

```bash
./run_linux_test.sh --checks-only    # 跳过端到端精修，约 75 s
./run_linux_test.sh --kernels        # 追加三种内核一致性验证，约 126 s
```

> 脚本**不会**自动联网安装依赖：检测到缺包时会打印上面的 venv 命令并退出。

---

## 4. 预期输出与通过判据

一键脚本共 7 个阶段，全部通过时最后打印 `ALL LINUX CHECKS PASSED`，
并把逐阶段 PASS/耗时写入 `linux_test_out/summary.txt`。
以下为实测（Ubuntu 26.04 / x86_64 / 96 核 / Python 3.12.15 / gemmi 0.7.5 / numpy 2.5.5 / scipy 1.18.1）：

| 阶段 | 通过判据 | 实测 |
|---|---|---|
| 0/7 环境 | Python ≥ 3.9，`import gemmi, numpy, scipy` 成功 | 3.12.15；`auto → simd (prebuilt _simd.so)` |
| 1/7 完整性 | `external dictionary paths in sys.path : NONE`；`RESULT : PASS` | 35639 residues，字典 47.8 MB，licence present |
| 2/7 真实数据约束 | `RESULT : PASS`，约束计数非零 | `684 bonds / 915 angles / 117 planes / 242 torsions`，`f=7261.923272` |
| 3/7 单元测试 | `OK`（17 项） | 7 s |
| 4/7 数值自检 | `SELFTEST PASS`，10 项全过 | `Fc vs gemmi 3.88e-08`；X-ray 梯度 FD `1.57e-08`；几何梯度 FD `3.29e-08`；内核一致 `fused 4e-16 / simd 1e-14 / fft 1.1e-03`；64 s |
| 5/7 短精修 | 打印 `RESULT`，R/Rfree 不升高 | `R 0.3830→0.3825`，`Rfree 0.4116→0.4091`，`rmsBOND 0.0181→0.0091`，7 s |
| 6/7 产物校验 | `RESULT : PASS`（11 列齐全、空间群/晶胞不变、无 NaN、R 复算吻合） | 15 列 MTZ；`R(MTZ)=0.3862` vs `report=0.3825`（差 0.0037，见 6.2 节） |
| 7/7 内核一致（可选） | 三个内核 R/Rfree 差 < 1e-6 | `numpy/fused/simd` 全部 `R=0.382630 Rfree=0.409866`，差 2.5e-11；18.6 / 18.4 / 5.0 s |

日志与产物写在 `linux_test_out/`（**每次运行会清空重建 `selftest/`、`refine/`、`kernels/`**）：

```
linux_test_out/env.txt             解释器 / 依赖 / 机器 / 内核探测
linux_test_out/summary.txt         逐阶段 PASS + 耗时
linux_test_out/check_install.txt   完整性自检输出
linux_test_out/restraints.txt      真实数据约束输出
linux_test_out/unit_tests.log      单元测试日志
linux_test_out/selftest.log       10 项自检日志
linux_test_out/selftest.json      自检结果（机器可读）
linux_test_out/selftest/*          自检写出的 MTZ/PDB
linux_test_out/refine.log          短精修日志
linux_test_out/refine/refined.pdb / refined.mtz / report.json
linux_test_out/check_outputs.txt   产物校验输出
linux_test_out/kernels/{numpy,fused,simd}/   内核一致性运行（--kernels 时）
```

不同 CPU / BLAS / numpy 版本下末位数值可能有微小差异；**判据是各阶段 PASS 且 R/Rfree 不升高**，
不要求与上表逐位一致。

### 4.1 数值自检的 10 项是什么

| # | 检查 | 实测 |
|---|---|---|
| 1 | Fc 与 gemmi 独立实现比对（最大相对误差） | 3.88e-08 |
| 2 | `-LL` 与梯度路径自洽（`negll_only` vs `negll_and_G`） | 0 |
| 3 | X-ray 梯度有限差分校验 | 1.57e-08 |
| 4 | 几何梯度有限差分校验 | 3.29e-08 |
| 5 | ML 参数 D/S 是否落在物理区间 | D 0.754–0.999，S 740–32187 |
| 6 | 短精修是否真的降低 -LL | -LL 92070.9 → 91476.6 |
| 7 | 输出 MTZ 列是否齐全、可读 | 15 列，`missing=[]` |
| 8 | 输出是否在 REFMAC 绝对刻度上 | `rel\|F_out·k − F_in\| = 3.33e-08`，`mean\|FC−FC_ALL\| = 11.9` |
| 9 | **四个内核在同一 G 下 Fc/梯度是否一致** | fused 4e-16 / simd 1e-14 / fft 1.1e-03 |
| 10 | **FreeR 标志约定是否判对**（0=free / 0=work / 单值回退） | 三种情形全部判对 |

---

## 5. 手动分步复现（定位问题时用）

```bash
cd oasis-refine-0.2.0
PY=.venv/bin/python              # 或 python3

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

## 6. Linux 专属注意事项（重要）

### 6.1 加速内核与预编译二进制

包内**不含任何预编译二进制**（原来的 macOS Mach-O `_simd.so` 已删除），
Linux 上不会出现「invalid ELF header」一类问题。

`--kernel auto`（默认）按「CPU 是否支持 AVX-512F/DQ/VL + 是否有 C 编译器」自动选择：

```bash
./run.sh --mtz X.mtz --pdb X.pdb --out-dir out            # auto：有 AVX-512 就用 simd
./run.sh ... --kernel numpy                               # 纯 numpy 参考实现，永远可用
./run.sh ... --kernel simd                                # 强制 SIMD（首次运行自动编译 _simd.so）
./run.sh ... --kernel fft --fft-spacing 0.2               # FFT 结构因子（近似）
```

- 旧的 `--simd-kernel` / `--fused-kernel` / `--fft-kernel` 仍可用，等价于对应的 `--kernel X`。
- 探测失败（无编译器 / CPU 不支持）时**不会崩溃**：打印原因并回落到 numpy。
- `_simd.so` 是机器相关的，**不要跨机器拷贝**；删掉它下次会自动重编。

### 6.2 为什么从输出 MTZ 复算的 R 比日志高约 0.004

日志里的 R 在**观测刻度**上按逐反射 k 加权计算：
`R = Σ|Fo − k·|Fc|| / ΣFo`。而输出 MTZ 按 REFMAC 约定写在**绝对刻度**上
（`F_out = Fo / k_total`、`FC_ALL` 也是绝对刻度），两者相差一个逐反射的 k 因子，
所以直接拿输出文件的 `F` 与 `FC_ALL` 算 R 会高约 0.002–0.004。
把 `F_out × k_total` 代回去就能精确复现日志数值（自检第 8 项即验证这一点）。

### 6.3 FreeR 标志约定（最容易静默出错的地方）

`--free-value auto`（默认）按标志直方图判定，规则保守：

1. 标志 `0` 占观测反射的 1–25% → 按 CCP4 惯例，`0` 为 free；
2. 否则取占比 1–25%、最接近 5% 的少数标志（覆盖 `0=work, 1=free` 这类自建流程）；
3. 都不满足 → 回退 `0` 并告警；若 free 集超过 30%，额外警告「Rfree 无意义」。

需要强制时：`--free-value 1`。启动日志与 `report.json` 都会写明
`free set: <列名>=<值> -> N of M (x.x %)`，**先看这一行再相信 Rfree**。

### 6.4 打包：不要用 `zip -j` 或 Finder 的「压缩」

最初的交付目录曾因为被拍平（四层目录结构丢失、两个同名 `__init__.py` 相互覆盖）
而在 Linux 上直接报 `No module named 'minirefmac'`。请用 `pack_release.sh`
（内部 `tar` + 重生成 MANIFEST + 解包复验）或 `tar czf`，**绝不要 `zip -j`**。

### 6.5 其他

- 代码里没有任何本机绝对路径，也没有外部字典开关。`tools/build_monlib_bundle.py`
  的 `--src` 是必填参数（构建期才用），运行期顶层模块不会 `gemmi.cif.read` 外部 CIF。
- `--monlib` 已删除：字典固定从包内 `monlib_bundle.zip` 读取。
  `--no-monlib` 保留，用于禁用字典、改用 jelly-body fallback。
- 脚本与 Python 源文件均为 LF 行尾、UTF-8/ASCII；解压后若丢失执行位，
  执行 `chmod +x run.sh run_linux_test.sh pack_release.sh`。
- 字典许可：`monlib_bundle.zip` 是源化学组分字典的派生数据，按 LGPL-3.0 随包再分发，
  全文见 `minirefmac/data/LICENSE-CCP4-MONOMERS.txt`，请勿删除。

---

## 7. 正式精修用法

```bash
./run.sh --mtz /path/to/input.mtz --pdb /path/to/input.pdb \
         --out-dir out --ncyc 5 --maxiter 10 --solvent mask --map-scale absolute

# 指定列名 / 限制分辨率 / 强制 free 标志
./run.sh --mtz in.mtz --pdb in.pdb --out-dir out \
         --fp F --sigfp SIGF --free FreeR_flag --free-value 1 \
         --dmin 2.5 --dmax 50

# 有实验相位（MLHL）
./run.sh ... --phib PHIB --fom FOM --phase-prior
```

输入 MTZ 需含观测振幅、sigma、FreeR 标志列（列名自动识别，也可用 `--fp/--sigfp/--free` 指定）；
输出为 `refined.pdb`、`refined.mtz`（含绝对刻度与 FC/PHIC/FC_ALL/FWT/DELFWT/FOM 等列）、
`report.json`。完整参数见 `python3 -m minirefmac --help`。

典型规模参考：3.30 Å / 3056 原子 / 10607 反射，单周期约 100 s（numpy）；
7enm（1.70 Å / 670 原子）单周期约 3–19 s（视内核）。

---

## 8. 打包与分发

```bash
./pack_release.sh                 # dist/oasis-refine-0.2.0.tar.gz + 重生成 MANIFEST + 解包复验
./pack_release.sh --zip           # 同时出 zip
```

脚本会：按正确布局收集文件（排除 `.venv/`、`__pycache__/`、`*.so`、`linux_test_out/`、
`dist/`、`.git/`）→ 检查关键文件齐全 → 重生成 `MANIFEST.sha256` → 打包 →
**重新解包并 `sha256sum -c` 校验** → 用本机解释器试 `import minirefmac`。

---

## 9. 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `Missing gemmi / numpy / scipy` | 未装依赖 | 按第 3 节建 venv 并 `pip install -r requirements.txt` |
| `ModuleNotFoundError: gemmi` | 用了系统 `python3` 但包装在 venv 里 | 用 `.venv/bin/python`，或设 `PYTHON=...` |
| `No module named 'minirefmac'` | 目录被拍平 / 不在包根目录运行 | 用 `pack_release.sh` 重新打包；`cd` 到含 `minirefmac/` 的目录 |
| `FAIL: bundled dictionary missing/cannot be opened` | 拷贝不完整或 zip 损坏 | `sha256sum -c MANIFEST.sha256` 定位，重新拷贝 |
| `FAIL: minirefmac was imported from outside this deployment` | `PYTHONPATH` 里有另一份 minirefmac | 清空 `PYTHONPATH`，在部署目录内执行 |
| `SELFTEST FAIL` | 数值检查未过 | 看 `linux_test_out/selftest.json` 的 `checks`，定位是哪一项 |
| `simdkernel: no C compiler found` / `build failed` | 强制 `--kernel simd` 但缺编译器或 CPU 不支持 AVX-512 | 去掉该开关（默认 auto 会回落 numpy） |
| `R/Rfree 明显变差` | 输入/列名不匹配或 free 标志判错 | 看启动日志的 `free set:` 行与 `report.json` 的 `free`/`data_stats` |
| `Rfree` 与 `R` 几乎相同 | free 集判成了几乎全部反射 | 显式 `--free-value <正确的标志值>` |
| `Permission denied` | 解压丢了执行位 | `chmod +x run.sh run_linux_test.sh pack_release.sh` |

---

## 10. 一句话结论

在 Linux 上，只要 `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
再执行 `./run_linux_test.sh` 得到 `ALL LINUX CHECKS PASSED`，就说明
**字典、几何约束、结构因子、梯度、ML 目标、精修与输出全部在完全自包含的环境下工作正常**，
并且四个加速内核给出的是同一条精修轨迹。
