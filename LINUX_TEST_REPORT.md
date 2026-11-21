# oasis-refine · Linux 实测报告

测试机：Ubuntu 26.04 / x86_64 / 96 核 / 123 GB RAM
测试日期：2026-11-20
解释器：Python 3.12.15（uv 建 venv）+ gemmi 0.7.5 / numpy 2.5.5 / scipy 1.18.1
（与 README_LINUX.md 第 4 节记录的实测版本完全一致）

---

## 0. 一句话结论

**代码本身在 Linux 上完全跑通**：`./run_linux_test.sh` 打印 `ALL LINUX CHECKS PASSED`，
4 个阶段全部通过，数值与 README 的实测表逐项吻合。

**但交付目录原样跑不通**——不是代码问题，是打包时目录结构被拍平了。见第 1 节。

---

## 1. 原样运行失败的原因（先修这个）

原目录 `/Large_pool/Workbuddy/OASIS-Refine/` 里所有文件都在顶层：

```
./cli.py  ./core.py  ./dataio.py ... ./monlib_bundle.zip  ./input.mtz  ./input.pdb  ./__init__.py
```

实测两条入口都直接失败：

```
$ python3 check_install.py
FAIL: cannot import minirefmac: No module named 'minirefmac'

$ ./run_linux_test.sh
FAILED: missing test data: .../OASIS-Refine/test_data/7enm/input.mtz
```

原因：`minirefmac/`、`minirefmac/data/`、`minirefmac/tools/`、`test_data/7enm/`
四层目录结构丢失；其中两个同名文件 `minirefmac/__init__.py` 与
`minirefmac/data/__init__.py` 冲突，只剩下了 data 的那个
（顶层 `__init__.py` 的 sha256 = `639a0aa9…`，对应 MANIFEST 里的
`minirefmac/data/__init__.py`），**`minirefmac/__init__.py`（`480fe6a8…`）整个丢失**。
这是 `zip -j` / Finder「压缩」一类操作的典型结果。

用 MANIFEST 校验可精确定位：

```
$ sha256sum -c MANIFEST.sha256
minirefmac/__init__.py: FAILED          ← 唯一一项，就是丢失后重建的那个
28 / 29 项 OK
```

> 重建的 `minirefmac/__init__.py` 只有 docstring + `__version__ = "0.1.0"`。
> 代码里没有任何地方读 `minirefmac.__version__`（版本串在 `refine.py` 里是硬编码的），
> 所以不影响功能，但**请从 Mac 上把原件补回来**再重新打包。

---

## 2. 修复后的可运行部署

已按 MANIFEST 的路径还原成正确的树，放在：

```
/Large_pool/Workbuddy/OASIS-Refine-linux/       ← 可运行版本（原目录未改动）
├── README.md  README_LINUX.md  MANIFEST.sha256
├── check_install.py  pyproject.toml  requirements.txt  run.sh  run_linux_test.sh
├── minirefmac/            *.py + _simd.c
│   ├── data/              __init__.py, monlib_bundle.zip, LICENSE-CCP4-MONOMERS.txt
│   └── tools/             build_monlib_bundle.py
├── test_data/7enm/        input.mtz, input.pdb
├── .venv/                 Python 3.12.15 + gemmi/numpy/scipy
└── linux_test_out/        本次测试的日志与产物
```

复现命令：

```bash
cd /Large_pool/Workbuddy/OASIS-Refine-linux
./run_linux_test.sh                # 全量，1m42s
./run_linux_test.sh --checks-only  # 跳过短精修
```

---

## 3. 一键测试结果（与 README 实测表对照）

| 阶段 | 判据 | 本机实测 | README 记录 | 结论 |
|---|---|---|---|---|
| 1/4 完整性 | `RESULT : PASS`，无外部字典路径 | 35639 residues / 47.8 MB / licence present / external paths NONE | 同 | 一致 |
| 2/4 真实数据约束 | `RESULT : PASS`，计数非零 | 684 bonds / 915 angles / 117 planes / 242 torsions；f=7261.923272；rmsd_bond=0.018109 | 同 | 一致 |
| 3/4 数值自检 | 8 项全 PASS（`selftest.json` 另有 1 条 `geometry_stats` 信息项） | 74.1 s；Fc vs gemmi 3.88e-08；X-ray 梯度 FD 1.57e-08；几何梯度 FD 3.29e-08；-LL 一致性 0 | 3.59e-08 / 1.48e-08 / 3.29e-08 | 一致（末位差异属正常） |
| 4/4 短精修 | R/Rfree 不升高，输出 3 文件 | R 0.3830→0.3825，Rfree 0.4116→0.4091，rmsBOND 0.0181→0.0091，23.8 s | 同 | 一致 |

总计 1m42s（96 核，user 12m54s，BLAS 多线程生效）。

---

## 4. 脚本之外的额外验证

### 4.1 多周期精修（5 cycle × 10 iter，215 s）

```
cycle 0  R=0.3830  Rfree=0.4116  FOM=0.682  rmsBOND=0.0181
cycle 1  R=0.3825  Rfree=0.4030  FOM=0.688  rmsBOND=0.0063
cycle 2  R=0.3775  Rfree=0.4017  FOM=0.701  rmsBOND=0.0056
cycle 3  R=0.3740  Rfree=0.4009  FOM=0.709  rmsBOND=0.0050
cycle 4  R=0.3710  Rfree=0.4027  FOM=0.717  rmsBOND=0.0055
cycle 5  R=0.3687  Rfree=0.4047  FOM=0.720  rmsBOND=0.0050
```

Rfree 在第 3 周期见底（0.4009）后回升，R 继续下降——5 周期已经轻微过拟合，
自动权重从第 2 周期起逐步把 w_xray 从 2.59 提到 4.48。这是方法学行为，与 Linux 无关。

### 4.2 三个加速内核

| 开关 | 结果 |
|---|---|
| `--simd-kernel` | **首次运行自动用 gcc 编译出 `minirefmac/_simd.so`**（本机有 AVX-512F/DQ/VL），成功；单周期 18.1 s → 3.4 s（约 5×）；结果 Rfree 0.4095（与纯 Python 的 0.4091 属内核数值差异） |
| `--fused-kernel` | 正常，单周期 10.7 s |
| `--fft-kernel` | 正常，单周期 8.8 s |

即 README 第 6 节说的「Linux 上删掉 Mach-O、按需从 `_simd.c` 编译」确实成立。

### 4.3 安装 / 入口 / 环境

- `pip install .`（Python 3.12 干净 venv）成功，控制台命令 `minirefmac --help` 可用
- 从任意工作目录调用 `run.sh`（绝对路径）正常
- 故意设 `PYTHONPATH=/tmp` 也不影响（`run.sh` 先 cd 到包根目录）

### 4.4 输出文件与 CCP4 生态的互操作

`mtzdump`（CCP4 9.0.017）读取 `refined.mtz` 正常：

```
P 21 21 21, cell 35.245 40.703 66.328 90 90 90, 11057 reflections, 34.692–1.696 Å
15 columns: H K L F SIGF FreeR_flag FWT PHWT DELFWT PHDELWT FOM FC_ALL PHIC_ALL FC PHIC
HISTORY: MINIREFMAC output_scale=absolute / k_overall / k_total / F_abs = F_obs / k_total / FC/PHIC atom-only
```

`FWT/PHWT`、`DELFWT/PHDELWT` 命名符合 CCP4/COOT 习惯，可直接进 COOT 看 2Fo-Fc / Fo-Fc 图。
`refined.pdb` 用 gemmi 回读正常：670 原子 / 82 残基 / 2 链，B 3.88–62.75 Å²，无 NaN。

### 4.5 真实规模数据（3056 原子 / 10607 反射 / 3.30 Å）

用 `/raid0_ssd/AI-XRAY/output_hm/4L6R/4L6R_030075.{mtz,pdb}`：

```
列名自动识别（FOBS / SIGFOBS / R-free-flags）
约束 3117 bonds / 4217 angles / 532 planes / 1052 torsions / 26231 VDW 对
1 cycle × 3 iter：97 s（首次），R 0.2785→0.2702，Rfree 0.2781→0.2759
```

3000 原子量级内存/耗时都正常（单进程，BLAS 多线程）。

---

## 5. 两个会误导使用者的坑（与 Linux 无关，建议改）

### 5.1 free 标志约定：默认值会把 95% 的数据当 free

程序默认 `--free-value 0`（CCP4 惯例：0 = 排除/free）。但 AI-XRAY 生成的 MTZ 里
`R-free-flags` 是 **0 = work、1 = free**（10076 / 531）。默认跑不报错，只在日志里写成：

```
reflections 10607 (work 531 / free 10076), atoms 3056      ← 明显反了
```

于是 Rfree 是在 95% 数据上算的，完全失去意义（0.3255 这个数不能用）。
加 `--free-value 1` 后恢复正常：`work 10076 / free 531`，R 0.2785→0.2702，Rfree 0.2781→0.2759。

**建议**：在 `refine.py` 的启动日志里加一句 free 比例，比例 > 50% 或 < 1% 时打印警告
（例如 `WARNING: free set is 95.0% of reflections -- check --free-value`）。
这类错误不会崩，只会静默给出错误的 Rfree，是最难发现的一类。

### 5.2 从输出 MTZ 直接复算 R，会比日志略高约 0.002

日志 R = 0.3825 / Rfree 0.4091；直接拿输出 MTZ 的 `F` 和 `FC_ALL` 算得 0.3849 / 0.4106。
不是错，是刻度约定：日志的 R 在观测刻度上算（`Σ|Fo − k·|Fc||/ΣFo`，逐反射 k 加权），
输出文件是绝对刻度（`F_abs = F/k`、`FC_ALL = |Fc|`）。用 `k·FC_ALL` 复算即得 0.3825 / 0.4091，代数上完全一致。

**建议**：README 里加一句「输出为绝对刻度，直接复算 R 会略有出入」，省得别人以为引擎和输出不一致。

---

## 6. 重新打包的建议

1. 用 `tar czf` 或 `rsync`（保目录），**不要**用 Finder 压缩 / `zip -j`——本次就是被拍平的。
2. 先补回 `minirefmac/__init__.py` 原件。
3. 打包前排除 `.venv/`、`minirefmac/_simd.so`（Linux 会自动重编）、`linux_test_out/`。
4. 打完包在 Mac 上跑一遍 `shasum -a 256 -c MANIFEST.sha256`，29 项全 OK 再传。

---

## 7. 本机已验证清单

- [x] 部署完整性（自包含字典，无外部 CIF / 无预编译二进制）
- [x] 单体字典约束构建（684/915/117/242）
- [x] 数值自检 8 项（Fc、-LL、X-ray 梯度 FD、几何梯度 FD、ML 参数、短精修、输出 MTZ、绝对刻度）
- [x] 短精修端到端（读 MTZ+PDB → 精修 → 写 PDB/MTZ/JSON）
- [x] 5 周期精修
- [x] 三种加速内核（含 AVX-512 按需编译）
- [x] `pip install .` + 控制台入口
- [x] 输出 MTZ 被 CCP4 mtzdump 正常读取（15 列，P212121，1.70–34.69 Å）
- [x] 输出 PDB 可回读（无 NaN，B 合理）
- [x] 3056 原子真实数据规模
- [x] 任意工作目录 / PYTHONPATH 干扰下运行
