# minirefmac · oasis-refine

从零实现的**最大似然（ML）大分子晶体学精修引擎**，接口与行为对齐 CCP4 `refmac5`，
但完全自包含：不需要 CCP4、不需要 Phenix、不需要任何外部单体字典或预编译二进制。
唯一的运行时依赖是 `gemmi` / `numpy` / `scipy`。

- 版本：0.2.0（Python ≥ 3.9；已在 Linux + Python 3.12 上完整验证）
- 单体库：`minirefmac/data/monlib_bundle.zip`（35639 个残基，47.8 MB，LGPL-3.0）
- 测试数据：`test_data/7enm/`（P 21 21 21，1.70–34.69 Å，11057 反射 / 670 原子）
- Linux 验收与完整说明书：[`README_LINUX.md`](README_LINUX.md)

---

## 快速开始

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 一键验收（环境 → 完整性 → 约束 → 单元测试 → 数值自检 → 短精修 → 输出校验）
./run_linux_test.sh                 # 约 2 分钟
./run_linux_test.sh --kernels       # 额外验证三种加速内核结果一致

# 正式精修
./run.sh --mtz input.mtz --pdb input.pdb --out-dir out --ncyc 5 --maxiter 10
```

输出：`out/refined.pdb`、`out/refined.mtz`（F/SIGF/FC/PHIC/FC_ALL/FWT/PHWT/DELFWT/
PHDELWT/FOM，绝对刻度）、`out/report.json`、`out/refine.log`。

---

## 引擎做了什么

| 环节 | 实现 |
|---|---|
| 目标函数 | 最大似然（Pannu & Read 1996）：`-LL = Σ [ \|Fo - k·D·\|Fc\|\|² / (2σ²) + ln σ ]`，按分辨率分箱估计 D、S（σ_A 型） |
| 相角信息 | 可选 MLHL：`PHIB/FOM` 或 Hendrickson–Lattman `HLA..HLD`，`--phase-prior` 打开 |
| 刻度 | 整体 k、整体 B、各向异性 B（无迹张量）、体溶剂 k_sol/B_sol（mask 或 Babinet） |
| 体溶剂 | 分子表面 mask → 溶剂贡献，`FC_ALL = FC·(1 - k_sol·exp(-B_sol s²))` |
| 几何约束 | 键长/键角/平面/手性/扭转（来自随包单体字典）+ 对称展开的 VDW 排斥 |
| 优化器 | L-BFGS（`scipy`），坐标 + 各向同性 B 同时优化，带最大位移限制 |
| 权重 | 自动：按 X-ray 与几何梯度范数比平衡（`--weight-scale`，默认 0.05），逐周期更新 |
| 结构因子 | 直接求和（numpy / numpy 融合 / AVX-512 C）或 FFT（`--kernel fft`），四者共享同一套 ML 系数 |
| 输出 | REFMAC 风格的 MTZ 列与 `report.json`（逐周期 R/Rfree/-LL/FOM/rmsd/权重） |

### 三种加速内核，结果必须一致

`--kernel auto`（默认）在本机支持 AVX-512F/DQ/VL 且有 C 编译器时用 `simd`，
否则用 `numpy`。**所有内核在同一个 ML 系数 G 下计算 Fc 与梯度**，
因此精修轨迹相同（`--kernels` 会实测验证：R/Rfree 差 < 1e-6，实测 2.5e-11）。

| 内核 | 7enm 单周期（1 cycle × 1 iter） | 说明 |
|---|---|---|
| `numpy` | 18.6 s | 参考实现，永远可用 |
| `fused` | 18.4 s | numpy 内融合相位计算，无编译器时略省时间 |
| `simd` | 5.0 s | AVX-512 C 内核，首次运行自动用 clang/gcc 编译 `_simd.so` |
| `fft` | 44 s | FFT 结构因子，Fc 误差 ~0.1%，只在反射数极大时才有速度优势 |

---

## 目录

```
run.sh                  精修入口（POSIX sh）
check_install.py        部署完整性自检（依赖 / 包路径 / 字典 / 许可证）
check_outputs.py        精修产物自检（MTZ 列 / PDB 有限性 / R 复算 / report 一致性）
run_linux_test.sh       一键 Linux 验收（7 阶段）
pack_release.sh         打包含正确目录布局 + 重生成 MANIFEST + 解包复验
MANIFEST.sha256         全量校验和
requirements.txt        gemmi / numpy / scipy
pyproject.toml          可选：pip install . 后可用 minirefmac / minirefmac-selftest 命令
tests/                  单元测试（stdlib unittest，17 项，约 7 s）
test_data/7enm/         随包测试输入
minirefmac/             引擎源码
  data/monlib_bundle.zip  随包单体字典（运行时只从这里读）
  _simd.c                 AVX-512 内核源码（按需编译）
  tools/build_monlib_bundle.py  构建期脚本，重建字典时才用
README_LINUX.md         Linux 验收说明书（中文，含故障排查）
CHANGELOG.md            版本变更
LINUX_TEST_REPORT.md    macOS→Linux 交付验收报告
```

---

## 已知限制

1. **不做水分子自动加建/删除**，也不做 TLS 精修；输入模型里有什么就精修什么。
2. **B 因子只做各向同性**（含整体各向异性刻度），不做逐原子各向异性 B。
3. **不输出自由 R 的交叉验证曲线**，`best_rfree` 只记录历史最低点。
4. `--kernel fft` 的 Fc 有约 0.1% 的网格误差（R 偏高 ~0.0005），仅建议在超大反射集上使用。
5. 单体字典覆盖标准氨基酸/核酸/常见配体；未知残基回退 jelly-body 约束（`--no-monlib` 可整体回退）。

---

## 许可

代码为作者自有；`minirefmac/data/monlib_bundle.zip` 是 CCP4 单体字典的派生数据，
按 LGPL-3.0 随包再分发，全文见 `minirefmac/data/LICENSE-CCP4-MONOMERS.txt`。
