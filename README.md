# oasis-refine

自包含的 minirefmac 最大似然精修引擎部署包。

- **Linux 测试与完整说明书**：[`README_LINUX.md`](README_LINUX.md)
- 一键测试：`./run_linux_test.sh`
- 完整性自检：`python3 check_install.py`
- 精修入口：`./run.sh --mtz X.mtz --pdb X.pdb --out-dir out`

运行时只需 PyPI 的 `gemmi / numpy / scipy`；单体库约束固定从
`minirefmac/data/monlib_bundle.zip` 读取，不读外部 CIF、不加载预编译二进制。
