# ActiveContactGS 开源代码

这是论文 **Contact Geometry Is a Latent Variable: Active Bayesian Identification of Gaussian Physical Worlds** 的轻量开源仓库。

本目录只包含：

- 方法、训练、评测、统计与绘图源码；
- 26 项 CUDA 测试；
- 三种子、消融、视频观测、YCB 和 PIN-WM 的复现脚本与说明；
- Python/Conda 环境配置；
- GitHub CI、Issue/PR 模板；
- MIT License、引用文件和第三方资产说明。

本目录不包含 checkpoint、生成数据、实验 JSON、运行日志或论文图片。

## 快速开始

```bash
python -m venv .venv
python -m pip install -e ".[analysis,assets,dev]"
python -c "import torch, active_contact_gs; print(torch.__version__)"
python -m pytest --collect-only -q
```

全部实验入口均强制使用 CUDA；GPU smoke 可运行 `bash scripts/smoke_test.sh`。完整训练与正式评测命令见 [复现说明](docs/REPRODUCIBILITY.md)。正式公开前需要在 `CITATION.cff` 中填写最终作者名单、论文链接和仓库链接。
