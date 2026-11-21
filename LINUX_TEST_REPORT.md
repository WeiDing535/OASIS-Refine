# oasis-refine · Linux 实测与修订报告

测试机：Ubuntu 26.04 / x86_64 / 96 核 / 123 GB RAM / AVX-512F+DQ+VL
解释器：Python 3.12.15（uv 建 venv）+ gemmi 0.7.5 / numpy 2.5.5 / scipy 1.18.1
报告日期：2026-11-21（第一轮 11-20 验收，第二轮同日修订后复测）

---

## 0. 一句话结论

**代码在 Linux 上完全跑通**，第一轮 4 阶段验收全绿；第二轮把验收中暴露的
**会静默给错结果的缺陷**修掉、把测试从「看日志」变成「一键判据」，
复测 7 阶段全绿（含三内核一致性），耗时 126 s。

**原交付目录原样跑不通**——不是代码问题，是打包时目录结构被拍平了（第 1 节）。
第二轮新增的 `pack_release.sh` 专门防止这件事再发生。

---

## 1. 第一轮：原样运行失败的原因

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

> 重建的 `minirefmac/__init__.py` 只有 docstring + `__version__`。第一轮时版本串在
> `refine.py` 里硬编码，所以不影响功能；第二轮已把版本集中到
> `minirefmac.__version__`（现为 0.2.0），由 `report.json` 与 MTZ history 引用。

---

## 2. 可运行部署与第二轮修订

按 MANIFEST 的路径还原成正确的树，放在
`/Large_pool/Workbuddy/OASIS-Refine-linux/`（原目录未改动）。第二轮在此基础上做了
**功能修订 + 工程化**，全部记录在 `CHANGELOG.md`，要点：

| 类别 | 修订 |
|---|---|
| 会静默出错 | FreeR 标志判定（`--free-value auto`）、四内核共用同一 G、FFT 网格过粗（Fc 误差 3.5% → 0.11%）、`FC/PHIC` 改为原子贡献 |
| 接口 | `--kernel {auto,numpy,fused,simd,fft}`（旧布尔开关保留为别名）、`--version`、`python -m minirefmac`、错误码分级 |
| 测试 | 新增 17 项单元测试、自检 8 → 10 项、新增 `check_outputs.py` 产物校验、`run_linux_test.sh` 4 → 7 阶段 |
| 打包 | 新增 `pack_release.sh`：正确布局 + 重生成 MANIFEST + 解包复验 |

复现命令：

```bash
cd /Large_pool/Workbuddy/OASIS-Refine-linux
./run_linux_test.sh --kernels      # 全量 7 阶段 + 内核一致性，126 s
./run_linux_test.sh --checks-only  # 跳过端到端精修，70 s
```

---

## 3. 第二轮一键验收结果（7 阶段）

| 阶段 | 判据 | 本机实测 | 结论 |
|---|---|---|---|
| 0/7 环境 | Python ≥ 3.9，依赖可导入 | 3.12.15；`auto → simd (prebuilt _simd.so)`；96 核 | PASS |
| 1/7 完整性 | `RESULT : PASS`，无外部字典路径 | 35639 residues / 47.8 MB / licence present / external paths NONE | PASS |
| 2/7 真实数据约束 | `RESULT : PASS`，计数非零 | 684 bonds / 915 angles / 117 planes / 242 torsions；f=7261.923272 | PASS |
| 3/7 单元测试 | `OK`（17 项） | 7 s | PASS |
| 4/7 数值自检 | `SELFTEST PASS`，10 项 | Fc vs gemmi 3.88e-08；X-ray FD 1.57e-08；几何 FD 3.29e-08；内核一致 fused 4e-16 / simd 1e-14 / fft 1.1e-03；64 s | PASS |
| 5/7 短精修 | 打印 `RESULT`，R/Rfree 不升高 | R 0.3830→0.3825，Rfree 0.4116→0.4091，rmsBOND 0.0181→0.0091，7 s | PASS |
| 6/7 产物校验 | `RESULT : PASS` | 15 列 MTZ、空间群/晶胞/反射数不变、无 NaN、R(MTZ)=0.3862 vs report 0.3825（差 0.0037，见 5.2） | PASS |
| 7/7 内核一致（`--kernels`） | R/Rfree 差 < 1e-6 | numpy / fused / simd 全部 `R=0.382630 Rfree=0.409866`，最大差 2.5e-11 | PASS |

总计 126 s（`--checks-only` 为 70 s）。逐阶段 PASS 与耗时另存
`linux_test_out/summary.txt`，环境信息在 `env.txt`。

---

## 4. 脚本之外的额外验证

### 4.1 多周期精修（5 cycle × 10 iter）

修订后（`--kernel simd`，**39.4 s**；修订前 numpy 路径 215 s）：

```
cycle 0  R=0.3830  Rfree=0.4116  FOM=0.682  rmsBOND=0.0181  w=2.59
cycle 1  R=0.3825  Rfree=0.4030  FOM=0.688  rmsBOND=0.0063  w=2.59
cycle 2  R=0.3775  Rfree=0.4017  FOM=0.701  rmsBOND=0.0056  w=3.11
cycle 3  R=0.3740  Rfree=0.4009  FOM=0.709  rmsBOND=0.0050  w=3.74
cycle 4  R=0.3710  Rfree=0.4027  FOM=0.717  rmsBOND=0.0055  w=4.48
cycle 5  R=0.3687  Rfree=0.4047  FOM=0.720  rmsBOND=0.0050  w=4.48
```

**与第一轮 numpy 路径逐周期数值完全相同**（0.3687 / 0.4047），只是快了 5.5 倍
——这正是「所有内核共用同一 G」的验证：换内核不换轨迹。
Rfree 在第 3 周期见底后回升属方法学行为（5 周期已轻微过拟合），与平台无关。

### 4.2 真实规模数据（3056 原子 / 10607 反射 / 3.30 Å）

`/raid0_ssd/AI-XRAY/output_hm/4L6R/4L6R_030075.{mtz,pdb}`：

```
列名自动识别（FOBS / SIGFOBS / R-free-flags）
free set: R-free-flags=1 -> 531 of 10607 observed (5.0 %)   ← auto 判对了
1 cycle × 3 iter：20.6 s（simd），R 0.2785→0.2702，Rfree 0.2781→0.2759
```

第一轮同一份数据需要人工加 `--free-value 1`（否则默认把 95% 数据当 free，
得到无意义的 Rfree 0.3255）；现在 `auto` 直接判对，且启动日志把
`free set: ... (5.0 %)` 打印出来。

### 4.3 内核性能（7enm，1 cycle × 1 iter）

| 内核 | 时间 | 备注 |
|---|---|---|
| numpy | 18.6 s | 参考实现 |
| fused | 18.4 s | 无编译器时的选项 |
| simd | 5.0 s | 3.7× 于 numpy；首次运行自动编译 `_simd.so` |
| fft | 44 s | 网格按精度要求细化后，本机已无速度优势 |

### 4.4 输出文件与 CCP4 生态的互操作

`mtzdump`（CCP4 9.0.017）读取 `refined.mtz` 正常：

```
P 21 21 21, cell 35.245 40.703 66.328 90 90 90, 11057 reflections, 34.692–1.696 Å
15 columns: H K L F SIGF FreeR_flag FWT PHWT DELFWT PHDELWT FOM FC_ALL PHIC_ALL FC PHIC
HISTORY: MINIREFMAC version=0.2.0 kernel=simd / output_scale=absolute / F_abs = F_obs/k_total
         free_column=FreeR_flag free_value=0 n_free=487 n_work=9713 / R_work R_free -LL
```

`FWT/PHWT`、`DELFWT/PHDELWT` 命名符合 CCP4/COOT 习惯，可直接进 COOT 看 2Fo-Fc / Fo-Fc 图。
`refined.pdb` 回读正常：670 原子 / 82 残基 / 2 链，B 5.4–74.7 Å²，无 NaN。

### 4.5 安装 / 入口 / 环境

- `pip install .`（干净 venv）成功，`minirefmac --version`、`minirefmac-selftest` 均可用
- `python -m minirefmac` 与 `python -m minirefmac.cli` 等价
- `run.sh` 现在优先用本地 `.venv/bin/python`，其次 `$PYTHON`，最后 `python3`
- 从任意工作目录调用、`PYTHONPATH=/tmp` 干扰下均正常
- `pack_release.sh` 产物：`oasis-refine-0.2.0.tar.gz`（46 MB，35 个文件），
  解包后 `sha256sum -c MANIFEST.sha256` 全 OK，`import minirefmac` 正常

---

## 5. 两处曾经会误导使用者的问题（第二轮已修）

### 5.1 free 标志约定：默认值会把 95% 的数据当 free（已修）

旧行为默认 `--free-value 0`（CCP4 惯例 0 = free）。但 AI-XRAY 生成的 MTZ 里
`R-free-flags` 是 **0 = work、1 = free**（10076 / 531）。默认跑不报错，只在日志里写成
`work 531 / free 10076` —— 于是 Rfree 是在 95% 数据上算的，完全失去意义（0.3255 不能用）。

现在的行为：`--free-value auto`（默认）按直方图判定，4L6R 自动得到
`R-free-flags=1 -> 531 / 10607 (5.0 %)`；free 集 >30% 或落在 1–25% 之外会告警；
没有 FreeR 列时明确写 `Rfree unavailable` 而不是把所有反射当 free。

### 5.2 从输出 MTZ 直接复算 R，会比日志略高约 0.004（已写进文档与自检）

日志 R = 0.3825 / Rfree 0.4091；直接拿输出 MTZ 的 `F` 和 `FC_ALL` 算得 0.3862 / 0.4106。
不是错，是刻度约定：日志的 R 在观测刻度上算（`Σ|Fo − k·|Fc||/ΣFo`，逐反射 k 加权），
输出文件是绝对刻度（`F_abs = F/k`、`FC_ALL` 也是绝对刻度）。用 `k·FC_ALL` 复算即得
0.3825 / 0.4091，代数上完全一致——自检第 8 项就是在验证这一点，
`check_outputs.py` 则用 0.01 的容差做产物层面的兜底。

---

## 6. 重新打包（第二轮已自动化）

1. 用 `./pack_release.sh`（内部 `tar` + 重生成 MANIFEST + 解包复验），**不要**用
   Finder 压缩 / `zip -j`——第一轮就是被拍平的。
2. 打包会自动排除 `.venv/`、`__pycache__/`、`*.so`（含 `_simd.so`，目标机自动重编）、
   `linux_test_out/`、`dist/`、`.git/`。
3. 打完包在 Mac 上跑一遍 `shasum -a 256 -c MANIFEST.sha256`，再传。
4. 目标机：`tar xzf oasis-refine-0.2.0.tar.gz && cd oasis-refine-0.2.0` →
   `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` →
   `./run_linux_test.sh`。

---

## 7. 本机已验证清单

- [x] 部署完整性（自包含字典，无外部 CIF / 无预编译二进制）
- [x] 单体字典约束构建（684/915/117/242）
- [x] 数值自检 10 项（Fc、-LL、X-ray 梯度 FD、几何梯度 FD、ML 参数、短精修、
      输出 MTZ、绝对刻度、**内核一致**、**FreeR 约定**）
- [x] 单元测试 17 项（free 标志 / FFT 网格 / 内核 / R 代数 / 列名识别 / 无 free 列）
- [x] 短精修端到端（读 MTZ+PDB → 精修 → 写 PDB/MTZ/JSON）
- [x] 5 周期精修（39.4 s，与 numpy 路径逐周期一致）
- [x] 四内核（numpy / fused / simd / fft），三精确内核 R 差 2.5e-11
- [x] `pip install .` + `minirefmac` / `minirefmac-selftest` 入口
- [x] 产物校验（`check_outputs.py`）：MTZ 11 必需列、无 NaN、R 复算吻合、PDB 有限
- [x] 输出 MTZ 被 CCP4 mtzdump 正常读取（15 列，P212121，1.70–34.69 Å）
- [x] 输出 PDB 可回读（无 NaN，B 合理）
- [x] 3056 原子真实数据规模（20.6 s / 周期，auto free 判定正确）
- [x] 任意工作目录 / PYTHONPATH 干扰下运行
- [x] `pack_release.sh` 打包 → 解包 → MANIFEST 校验 → import 全链路
