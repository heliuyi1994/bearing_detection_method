# AGENTS.md

本文件为 AI 编码代理（ZCode / Claude Code 等）在本仓库中工作时的指导。面向对项目零了解的读者。
根目录另有 `CLAUDE.md`，内容与本文件保持一致，可互为参照。

## 项目概述

本仓库实现**基于振动信号的滚动轴承故障诊断**（传统信号处理 + 经典机器学习），
唯一方法产物是统一包 `bearing_unified_method/`：

| 目录 | 内容 | 包名（src 布局） | CLI |
|---|---|---|---|
| `bearing_unified_method/` | 统一包：阶段一 DSP（v2 盲梳检规则：Kurtogram 选带 + 零相位解调 + 梳存在性盲检 + 频带落入定位）+ 阶段二 ML（两阶段 3 类）+ 阶段三验收交付（`docs/` principles 13 章 + design D1–D22、`notebooks/01_统一诊断流水线示例.ipynb`、`analysis/reports/unified_acceptance/统一包验收报告.md`） | `bearing_unified` | `bearing-unified` |
| `analysis/` | 真实数据验收评测工作区（脚本 + `reports/` 历史产物） | — | — |

包**自包含**，不依赖任何外部诊断包。历史上的两个姊妹项目（glm 方案
`bearing_glm_method`、kimi 方案 `bearing_kimi_method`）与合并原型
`merged_proposal/` 已完成合并使命并从仓库删除，代码与文档历史见 git 仓库。

所有文档、报告和面向用户的输出均为**中文**——请保持一致。

## 环境

**规范方式（文档口径）**：根目录 `environment.yml` 定义 conda 环境 `bearing-detection`（Python 3.12），以可编辑方式安装统一包：

```bash
conda env create -f environment.yml   # 仓库根目录执行
conda env update -f environment.yml   # 依赖变更后执行
```

**依赖版本以根目录 `requirements.txt` 为单一事实来源**（固定版本：numpy 2.5.3 / scipy 1.18.1 / matplotlib 3.11.2 / scikit-learn 1.9.1 / pytest 9.1.1 / nbclient / ipykernel）；项目 `pyproject.toml` 只保留兼容范围（>= 下限），供独立 pip 安装场景。

**本机实际状态（2026-09 验证）**：conda 环境 `bearing-detection` 当前**不存在**；可用环境是仓库根的 `.venv/`（Python 3.13.12），统一包已可编辑安装、版本与 `requirements.txt` 一致，全部测试在该环境通过。本机上直接用：

```bash
VENV=/home/dayin/git/bearing_detection_method/.venv/bin
$VENV/python -m pytest          # 在 bearing_unified_method/ 下执行
$VENV/bearing-unified ...       # CLI 也在 .venv/bin/ 下
```

## 构建与测试命令

无构建步骤（纯 Python，setuptools src 布局）。测试与常用 CLI：

```bash
# 测试——unified 176 项（约 150 s），2026-09-28 全量复验通过
cd bearing_unified_method && <环境python> -m pytest
# 单文件 / 关键字过滤：追加 tests/test_kurtogram.py 或 -k kurtogram

# 统一包：仿真 → 分析（v2 规则，转速自估、匹配不依赖转速）
bearing-unified simulate --fault BPFO --snr 5 --seed 42 --out sim_bpfo.csv
bearing-unified analyze sim_bpfo.csv --fs 16000 --bearing FIELD7 --out report/
bearing-unified bearings --rpm 1150          # 内置轴承（含现场 FIELD7）及特征频率
bearing-unified bands --bearing FIELD7       # 转速窗×几何推出的四个定位频带
bearing-unified ml-dataset --n-base 15 --windows 8 --out data/ds.npz
bearing-unified ml-train --dataset data/ds.npz --out models/   # 监督后端
bearing-unified ml-train --dataset data/ds.npz --stage1 novelty --out models_nov/
bearing-unified predict data.csv --fs 16000 --model models/model.joblib --compare
```

`ml-train`、`ml-dataset` 及分析脚本会重新生成产物（`models*/`、`*.npz`、`sim_*.csv`、`report/`）——它们是输出而非源文件，已被 `.gitignore` 排除。

## 代码组织

`bearing_unified_method/src/bearing_unified/`：

`fault_freqs`（轴承库与特征频率，含现场 FIELD7 注册）、`indicators`（9 项时域指标）、`preprocessing`（去直流/去趋势/零相位带通）、`spectrum`（幅值谱/包络/包络谱/倒频谱）、`kurtogram`（谱峭度自适应选带）、`matcher`（谐波匹配，局部邻域中值显著度基线 + 整数倍混淆消解；v2 中仅作理论复核）、`combdet`（盲梳检 v2-A/B：显著峰拾取、梳扫、倒谱互证、干扰防护）、`localize`（频带落入定位 + 理论复核 + 3 类置信度合成）、`baseline`（马氏距离健康基线 + 域内标定，辅助判据）、`simulate`（恒速故障仿真）、`loaders`（csv/txt/npy/npz/mat，含 CWRU）、`diagnose`（v2 流水线编排）、`report`（四联图 + 中文文本 + JSON）、`cli`、`ml/`（`features` 51 维六组特征 / `dataset` 防泄漏分组 / `train` 两阶段监督 + 新奇检测后端 / `evaluate` GroupKFold/LOSO/工作点/DSP 对比 / `infer`）。

- 改设计前先读 `docs/design.md`（决策 D1–D22，含"为什么"）与 `docs/principles.md`（13 章原理 + 附录）。
- v2 判别仅 3 类（外圈 BPFO / 内圈 BPFI / 滚珠 BSF，滚珠 2×BSF 主峰带归入滚珠），不做保持架 FTF 判别（特征频率计算能力保留）。
- 判别主依据是**盲梳检存在性**（不依赖理论频率、不依赖转速），定位用频带落入法；马氏距离仅作无梳时的辅助判据。
- 变速工况（阶次跟踪/角域重采样）与时频分析（STFT/CWT）为设计决策 D22 明确**不做**的范围。

## 真实数据评测工作区 `analysis/`

历史背景：评测所用真实数据集 `labeled_dataset_4class/`（81 条四类台架 WAV）与音频资产库（仓库外 `../audio-labeling/`）已完成验收使命；数据集本体与历史评测图片（`analysis/reports/figures/`）及引用它们的过期报告已清出仓库并从 git 历史中重写清除（2026-09 仓库瘦身），文本结论报告保留于 `analysis/reports/`（含 `统一包验收报告.md`）。

保留脚本（从 `analysis/` 目录下运行，直接 `python <脚本>.py`）：

| 脚本 | 作用 | 主要输出（`analysis/reports/`） |
|---|---|---|
| `common.py` | 公共层：中文绘图配置、manifest/WAV 加载、几何断言（`verify_geometry()`）、JSON 落盘 | — |
| `run_eval_unified.py` | 统一包正式验收复测（audio-assets 1349 条；`--smoke` / 全量 / `--aggregate-only`） | `audio_assets/unified/`（results.csv、eval_detail.json、figures/） |
| `run_eval_unified_4class.py` | 统一包 81 条四类台架复测（正常 0 误报门槛） | `unified_4class/`（results.csv、eval_detail.json） |
| `make_acceptance_report.py` | 汇总两批复测产物生成统一包验收报告（纯标准库） | `unified_acceptance/统一包验收报告.md` |

注意：完整复测需要原始数据（四类数据集已从仓库及 git 历史清除，如需复测须由仓库外音频资产库 `../audio-labeling/` 重新整理；音频资产库在仓库外），`--aggregate-only` 模式仅由已有 results.csv 重出聚合与图。

## 代码与开发约定

- **语言**：代码注释、文档、报告、CLI 输出一律中文；源码文件多为 `# -*- coding: utf-8 -*-` 声明 + `from __future__ import annotations`。
- **依赖**：只加 `requirements.txt` 里有的东西；版本改动只动 `requirements.txt`，`pyproject.toml` 保持兼容范围。
- **绘图**：一律 `Agg` 无头后端（统一包在 `report.py`、analysis 在 `common.py` 均已强制）——绝不依赖显示器；中文图用 WenQuanYi Micro Hei 字体（见 `analysis/common.py:setup_matplotlib`）。
- **产物**：模型、`*.npz`、仿真 CSV、报告目录均为可重新生成的输出，不入库。

## 测试约定

测试承载物理真值，是 DSP 改动的仲裁标准：高斯信号峭度 ≈ 3、正弦峭度 = 1.5、特征频率与理论值/CWRU 公布值核对、SNR 定量验证、盲梳检检出与干扰防护、频带定位正确性、健康信号零误报、数据加载器往返一致、CLI 冒烟测试；ML 测试覆盖特征尺度不变性、防泄漏分组、两阶段架构行为、新奇检测后端、模型序列化 roundtrip。**改 DSP/特征代码必须跑对应测试**。

## 注意事项（正确性与安全）

- **三个输入绝不可凭猜测填写**：采样率 fs、转速 rpm/轴频、轴承几何。缺失会导致全部特征频率错位、结论完全无效。转速未提供时 v2 盲梳检判别仍有效（不依赖转速），但 T2 物理特征与理论复核需要转速（约定：用户给定 > 包络谱自估 > 回退 1150 rpm）。
- 纯离线库，无联网需求；无密钥/凭据管理问题。
- ML 模型在**仿真数据**上训练：仿真准确率只证明管线正确，不等于现场性能，部署前需用现场数据复验（`--compare` 与 v2 DSP 交叉对比即为此设计；两法不一致时必须提示人工复核）。
- 真实数据集标签为工厂声称值（非拆解确认缺陷部位），结论对外表述时保留这一口径。
