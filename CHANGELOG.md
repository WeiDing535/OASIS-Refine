# 变更记录

## 0.2.0 — Linux 验收后的工程化修订

本轮修订的前提：交付包在 Linux 上「能跑」，但把验证过程中暴露的若干**静默错误**
和工程缺口补上，并把测试从「人工看日志」变成「可重复的一键判据」。
所有改动都在 Linux（Ubuntu 26.04 / x86_64 / 96 核 / Python 3.12.15）上实测过。

### 正确性（会静默给出错误结果的）

- **FreeR 标志约定**：`--free-value` 默认由 `0` 改为 `auto`。旧行为在
  `0=work, 1=free` 的文件上（如 4L6R）会把 95% 的反射当 free 集，
  给出无意义的 Rfree（实测 0.3255），且**不报错**。新规则：标志 0 占 1–25% →
  按 CCP4 惯例 0=free；否则取占比 1–25% 且最接近 5% 的少数标志；都不满足则回退 0 并告警。
  free 集超过 30% 时额外告警。启动日志、`report.json` 的 `free` 字段、
  MTZ history 都会写明实际采用的值与占比。
- **无 FreeR 列**的 MTZ：不再当成「全部 free」，而是 `mode=absent`、
  `Rfree` 不可用、并 `warnings.warn` 提示。
- **四个内核共用同一个 ML 系数 G**：此前 fused / SIMD 路径复用 warm-start
  上一轮的 G，梯度不精确，轨迹与 numpy 参考路径分叉（R/Rfree 漂移约 4e-4）。
  现在 numpy / fused / simd / fft 都吃同一个 G，实测 R/Rfree 差 2.5e-11。
- **FFT 网格过粗**：`auto_spacing` 原来固定 0.5 Å，Fc 相对误差实测 **3.5%**
  （不是文档里写的 0.1–0.2%），R 因此偏高约 0.004。改为自适应
  （典型蛋白 B 因子下 0.234 Å；B < 3 Å² 时按 3 处理，避免为单个锐原子把网格推到
  1000³），Fc 误差降到 **1.1e-3**，R 与直接求和一致到小数点后 4 位。
  内存上限 `vmax` 保留，运行日志会打印网格尺寸与内存估算。
- **输出 MTZ 的 `FC/PHIC` 改为原子贡献**（`FC_ALL` 含体溶剂），与 REFMAC5 列语义对齐
  ——已用本机 `refmac5` 输出逐列核对。
- **非法 sigma** 用中位数替换并在 `data_stats` 里计数；缺失/非正 F、
  重复 HKL、altloc、氢原子也都有计数，启动日志按需提示。
- 内核入口对 `fc_atom_factor` 显式取实部，避免 `ComplexWarning` 静默截断。

### 接口

- 新增 `--kernel {auto,numpy,fused,simd,fft}`（默认 `auto`：CPU 有 AVX-512F/DQ/VL
  且有 C 编译器就用 simd，否则 numpy）。旧的 `--simd-kernel` / `--fused-kernel` /
  `--fft-kernel` 保留为**已弃用别名**，行为等价。非法值直接报错，探测失败静默回落。
- 新增 `--version`、`python -m minirefmac`（`minirefmac/__main__.py`）。
- `cli` 参数按组重排；错误分成「载入失败」（exit 2）与「精修失败」（exit 3），
  不再抛裸 traceback。
- `pyproject.toml`：`requires-python >= 3.9`，补 readme/keywords/classifiers/license，
  新增 `minirefmac-selftest` 控制台入口。
- 版本号集中到 `minirefmac.__version__`，`report.json` 与 MTZ history 都引用它。

### 测试与工具（新增）

- `tests/test_units.py`：17 项 stdlib unittest（7 s），覆盖 free 标志三种约定、
  FFT 网格规则、内核一致性、R 代数（完美 Fc → R=0）、MTZ 列名自动识别、
  无 free 列的行为。
- `minirefmac/selftest.py`：8 项 → **10 项**，新增「四内核在同一 G 下一致」与
  「FreeR 约定判定」。
- `check_outputs.py`（新）：不看程序日志，独立校验产物——MTZ 11 个必需列、
  空间群/晶胞/反射数不变、无 NaN、`|FWT|>0`、相位在 [0,360]、
  **从文件复算 R 与 report.json 相差 < 0.01**、PDB 坐标/B/occupancy 有限且
  B 在 `[bmin,bmax]` 内、report 周期数与 `ncyc` 一致。
- `run_linux_test.sh`：4 阶段 → **7 阶段**（环境 → 完整性 → 约束 → 单元测试 →
  数值自检 → 短精修 → 产物校验，可选第 8 步内核一致性），新增
  `--checks-only` / `--kernels` / `--python PATH`，逐阶段 PASS 与耗时写入
  `summary.txt`，环境信息写入 `env.txt`，失败即停并回显日志。
- `pack_release.sh`（新）：按正确布局收集文件（排除 `.venv/`、`__pycache__/`、
  `*.so`、`linux_test_out/`、`dist/`、`.git/`）→ 检查关键文件 → 重生成
  `MANIFEST.sha256` → 打包 → **重新解包并校验 MANIFEST** → 试 `import minirefmac`。
  目的是彻底避免 `zip -j` / Finder「压缩」把包结构拍平（那会导致
  `No module named 'minirefmac'`）。

### 性能（7enm，1 cycle × 1 iter，本机）

| 内核 | 时间 | 说明 |
|---|---|---|
| numpy | 18.6 s | 参考实现 |
| fused | 18.4 s | 无编译器时的省时选项 |
| simd | 5.0 s | 3.7× 于 numpy；首次运行自动编译 `_simd.so` |
| fft | 44 s | 网格细化到 0.234 Å 后，本机已无速度优势，仅在反射数极大时才有意义 |

### 文档

- `README.md` 重写为项目总览（引擎构成、内核对比、目录、已知限制）。
- `README_LINUX.md` 重写：7 阶段判据、10 项自检逐条说明、
  「为什么从输出 MTZ 复算 R 高 0.004」、「FreeR 约定」、「打包禁忌」、故障排查表。
- 新增本 `CHANGELOG.md`。

---

## 0.1.0 — 初始交付

- 自包含 ML 精修引擎：直接求和结构因子、按分辨率分箱的 D/S、体溶剂 mask/Babinet、
  单体字典约束、L-BFGS、自动权重、REFMAC 风格 MTZ 输出。
- 可选 AVX-512 C 内核、numpy 融合内核、FFT 结构因子三条加速路径。
- `check_install.py`、`run_linux_test.sh`（4 阶段）、`minirefmac.selftest`（8 项）。
