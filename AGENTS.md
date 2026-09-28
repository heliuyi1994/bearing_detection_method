# AGENTS.md

本文件为 AI 编码代理（ZCode / Claude Code 等）在本仓库中工作时的指导。面向对项目零了解的读者。
根目录另有 `CLAUDE.md`（只覆盖两个软件包，内容与本文件前部一致，可互为参照；本文件为最新超集）。

## 项目概述

本仓库围绕**基于振动信号的滚动轴承故障诊断**（传统信号处理 + 经典机器学习），包含六个部分：

| 目录 | 内容 | 包名（src 布局） | CLI |
|---|---|---|---|
| `bearing_glm_method/` | 姊妹项目 A（"glm"方案） | `bearing_diag` | `bearing-diag` |
| `bearing_kimi_method/` | 姊妹项目 B（"kimi"方案） | `bearing_fault` | `bearing-fault` |
| `bearing_unified_method/` | 统一包（阶段一 DSP + 阶段二 ML + 阶段三验收）：glm 骨架 + kimi 门控的内生化合并，**v2 盲梳检规则的正式实现**（蓝本 `merged_proposal/unified_prototype.py`，自包含不 import 两姊妹包）；ML 为 glm 两阶段监督式骨架 + kimi 新奇检测 Stage1 可选后端，含 T6 盲梳检特征组；阶段三交付 `docs/`（principles 13 章 + design D1–D22）、`notebooks/01_统一诊断流水线示例.ipynb`、两批真实数据验收（`analysis/reports/unified_acceptance/统一包验收报告.md`） | `bearing_unified` | `bearing-unified` |
| `labeled_dataset_4class/` | 真实台架标注数据集（81 条 WAV，四类） | — | — |
| `analysis/` | 用真实数据集对两方案做对比评测的脚本与报告 | — | — |
| `merged_proposal/` | 双方案合并评审材料（`双方案优缺点汇总与合并建议.md`）与合并方案原型（`unified_prototype.py`，作为第三方组合两包的实现；`rules="v1"` 理论锚定级联门控 / `rules="v2"` 盲梳检存在性+频带落入定位，两版路径并存） | — | — |

两个软件项目实现**同一个任务**、各自完备、分开开发以便并排对比。**它们之间没有共享代码——绝不要交叉导入**；仅在被明确要求时才保持修复/功能同步。`analysis/` 是两个包的**第三方使用者**：它分别 import 两个包的公共 API，但不构成两项目之间的依赖。

所有文档、报告和面向用户的输出均为**中文**——请保持一致。

## 环境

**规范方式（文档口径）**：两项目共用根目录 `environment.yml` 定义的 conda 环境 `bearing-detection`（Python 3.12），以可编辑方式安装两个包，两套 CLI 同时可用：

```bash
conda env create -f environment.yml   # 仓库根目录执行
conda env update -f environment.yml   # 依赖变更后执行
```

**依赖版本以根目录 `requirements.txt` 为单一事实来源**（固定版本：numpy 2.5.3 / scipy 1.18.1 / matplotlib 3.11.2 / scikit-learn 1.9.1 / pytest 9.1.1 / nbclient / ipykernel）；各项目 `pyproject.toml` 只保留兼容范围（>= 下限），供独立 pip 安装场景。

**本机实际状态（2026-09 验证）**：conda 环境 `bearing-detection` 当前**不存在**；可用环境是仓库根的 `.venv/`（Python 3.13.12），两个包均已可编辑安装、版本与 `requirements.txt` 一致，全部测试在该环境通过。本机上直接用：

```bash
VENV=/home/dayin/git/bearing_detection_method/.venv/bin
$VENV/python -m pytest          # 在各自项目目录下执行
$VENV/bearing-diag ...          # CLI 也在 .venv/bin/ 下
```

## 构建与测试命令

无构建步骤（纯 Python，setuptools src 布局）。测试与常用 CLI：

```bash
# 测试——glm 100 个（约 63 s）/ kimi 88 个（约 11 s）/ unified 176 个（约 150 s），均已验证通过
cd bearing_glm_method && <环境python> -m pytest
cd bearing_kimi_method && <环境python> -m pytest -q
cd bearing_unified_method && <环境python> -m pytest
# 单文件 / 关键字过滤：追加 tests/test_kurtogram.py 或 -k kurtogram

# glm：仿真 → 分析（需采样率 + 转速 + 轴承型号）；ML 训练约 5 分钟
bearing-diag simulate --fault BPFO --snr 5 --seed 42 --out sim_bpfo.csv
bearing-diag analyze sim_bpfo.csv --fs 12000 --rpm 1797 --bearing SKF6205 --out report/
bearing-diag bearings --rpm 1797          # 内置轴承及特征频率
bearing-diag ml-train --out models/       # 600 个仿真样本
bearing-diag predict data.csv --fs 12000 --model models/model.joblib --rpm 1797 --bearing SKF6205

# kimi：仿真 + 完整诊断；ML 数据集/训练/预测
bearing-fault diagnose --fault inner --snr -6 --out out_diag
bearing-fault ml-dataset --out data/ml_dataset.npz     # 360 样本
bearing-fault ml-train --dataset data/ml_dataset.npz --out models/
bearing-fault ml-predict --model models/two_stage.joblib --fault inner
```

`ml-train`、`ml-dataset` 及演示脚本会重新生成产物（`models*/`、`*.npz`、`sim_*.csv`、`report/`、`out_*`）——它们是输出而非源文件，已被 `.gitignore` 排除。

## 代码组织

### 共同领域流水线（两个包都有的模块）

1. **轴承几何 → 故障特征频率**：BPFO（外圈）、BPFI（内圈）、BSF（滚动体）、FTF（保持架），均与轴频 fr = rpm/60 成正比。
2. **仿真器**：Randall 冲激响应模型（冲击序列 + 衰减共振 + 滑差 + 幅值调制 + 加性噪声），SNR 定量可控——生成真值测试数据与 ML 训练数据。
3. **时域指标**：RMS、峭度、峰值/脉冲/裕度因子等。
4. **包络解调（共振解调）**：带通滤波 → Hilbert 变换 → 包络谱。
5. **谐波匹配**：包络谱中找候选故障频率的谐波族 → 判定 + 置信度 → 图表/文本/JSON 报告。

### glm：`bearing_glm_method/src/bearing_diag/`

`fault_freqs`（轴承库与特征频率）、`indicators`（9 项时域指标）、`spectrum`（FFT/包络谱/倒频谱）、`kurtogram`（谱峭度自适应选带）、`simulate`、`matcher`（谐波匹配 + 整数倍混淆消歧）、`loaders`（csv/txt/npy/npz/mat，含 CWRU）、`diagnose`（流水线编排）、`report`（四联图 + 中文文本 + JSON）、`cli`、`ml/`（`features` 48 维特征 / `dataset` 防泄漏分组 / `train` / `evaluate` / `infer`）。

- 改设计前先读 `docs/design.md`（设计决策 D1–D13，含"为什么"）与 `docs/principles.md`（11 章原理 + 附录）。
- 独有：Kurtogram 选带、倒频谱；谐波匹配用局部邻域中值基线；低 SNR 策略**宁漏检不错判**。
- ML：48 维双轨特征（物理 + 统计）；两阶段分层——Stage1 正常/异常检测器（阈值可调，报告给出 0.3/0.5/0.7 工作点漏报/误报率），Stage2 条件化 4 分类，联合置信度 = p_abnormal × P(fault|abnormal)；GroupKFold（防泄漏）+ LOSO 跨工况评估；消融开关 `--arch flat`、`--no-physical`。
- `SKILL.md` 是 AI 助手技能入口。注意：其中写死的项目路径 `/home/dayin/git/bearing_detection` 已过期，仓库现位于 `/home/dayin/git/bearing_detection_method`。

### kimi：`bearing_kimi_method/src/bearing_fault/`

`bearing`、`simulator`（支持变速工况）、`features`、`envelope`、`spectrum`、`order_analysis`（角域重采样阶次谱）、`timefreq`（STFT + 自实现复 Morlet CWT）、`diagnosis`（谱峰匹配 + 马氏距离健康基线）、`pipeline`、`preprocessing`、`plotting`、`cli`、`ml/`（26 维特征；IsolationForest/EllipticEnvelope 检测 + SVM/RF/kNN 分类两级）。

- 理论文档：`docs/` 八章（`docs/README.md` 是索引，每章对应一个模块）；教程：`notebooks/` 7 个 Jupyter；ML 子系统设计规格：`docs/superpowers/specs/2026-09-19-ml-fault-classification-design.md`。

### 命名对照

| 概念 | glm | kimi |
|---|---|---|
| 故障标签 | `BPFO` / `BPFI` / `BSF` / `FTF` / `normal` | `outer` / `inner` / `ball` / `cage` / `healthy` |
| CWRU 6205 轴承 | `"SKF6205"`（字符串，宽松匹配） | `Bearing.sk6205()` |
| 完整诊断入口 | `bearing-diag analyze` / `diagnose()` | `bearing-fault diagnose` / `DiagnosisPipeline.run()` |

## 真实数据集与对比分析

### `labeled_dataset_4class/`（真实台架数据）

81 条 WAV（PCM int16 单声道，**16 kHz**，约 3 s），工厂标签四类：正常 10 / 外圈 20 / 内圈 29 / 滚珠 22，采集批次 9-16 / 9-17 / 9-18。

- 入口文件：`README.md`（先读）、`manifest.csv`（每条录音一行：路径、标签、批次、session、理论频率）、`labels.csv`（精简三列）、`geometry.json`（轴承几何机器可读副本）、`wav/{类别}/`。
- 轴承几何（自定义，**不是** CWRU 6205）：n=7，d=3.969 mm，D=15.0 mm，α=5.6°，外圈固定内圈随轴。阶次（×fr）：BPFO 2.578320 / BPFI 4.421680 / BSF 1.758605 / **滚珠主峰用 2×BSF = 3.517209** / FTF 0.368331。恒等式 BPFO + BPFI = n·fr 可自检。
- 标称转速窗 1100–1200 rpm；72 条有既有转速估计 `fr_est`（18.33–19.99 Hz），9 条（9-18 上午）缺失。
- **标签是工厂声称值，不是拆解确认的缺陷部位**——所有"准确率"应理解为与工厂标签的一致率；匹配容差建议 ±2%（滑差 1–2%）。
- `pack_dataset.py`（纯标准库）可从原始录音重建本包；原始数据在仓库外（`../绿智轴承分析/`，见 manifest 的 `source_path` 列）。

### `analysis/`（双方案评测工作区）

脚本（从 `analysis/` 目录下用共用环境运行，直接 `python <脚本>.py`）：

| 脚本 | 作用 | 主要输出（`analysis/reports/`） |
|---|---|---|
| `common.py` | 公共层：manifest/WAV 加载、两包轴承对象构造与几何断言（`verify_geometry()`）、统一 Kurtogram 选带、glm/kimi 诊断包装、标签映射 | — |
| `eda.py` | 数据质量 / 时域频域统计 / 选带分布 / 转速证据 | `eda_*.json`、`figures/eda_*.png` |
| `run_eval.py` | 主评测：81 文件 × 两方案 × 两轮次 | `results.csv`、`eval_detail.json` |
| `analyze_results.py` | 分层准确率、混淆矩阵、两方案一致性（Cohen's kappa）、错误归因 | `stats_summary.json`、`figures/confusion_*.png` |
| `detect_report.py` | 不以标签为基准的故障检出分析（检出率、置信排名、头部样本谐波证据） | `detect_stats.json`、`table_matched.md` |
| `make_figures.py` | 分歧样本深挖多联图 + 逐样本主表 | `figures/divergence/*.png`、`table_samples.md` |
| `run_eval_audio_assets.py` | 用合并方案原型（`merged_proposal/unified_prototype.py`）评测 audio-labeling 库音频（product_id=1，正常 566 + 异常 783；`--rules v1/v2`、`--smoke` / 全量 / `--aggregate-only`） | `audio_assets/`（v1：results.csv、eval_detail.json、音频资产库评测报告.md、figures/；v2：`rules_v2/` 下同构产物 + 规则V2对照报告.md） |
| `run_eval_unified.py` | 统一包正式验收复测（1349 条，引擎 `bearing_unified.diagnose`；`--smoke` / 全量 / `--aggregate-only`，口径与原型 v2 同） | `audio_assets/unified/`（results.csv、eval_detail.json、figures/） |
| `run_eval_unified_4class.py` | 统一包 81 条四类台架复测（正常 0 误报门槛 + 与 glm/kimi 轮次 B 对照） | `unified_4class/`（results.csv、eval_detail.json） |
| `make_acceptance_report.py` | 汇总两批复测产物生成统一包验收报告 | `unified_acceptance/统一包验收报告.md` |

评测口径（`common.py` 头部有决策记录）：轮次 A 统一 1150 rpm，轮次 B 用逐文件 `fr_est`（缺失回退 1150）；两方案统一用 glm 的 Kurtogram 自适应选带（kimi 侧注入其 `DiagnosisPipeline(band=...)`，贴近奈奎斯特的频带会夹取）；FTF/cage 判决归入"其他"（计错，不并入滚珠）。最终报告：`reports/labeled_dataset_4class_分析报告.md`（主报告）与 `reports/故障检出与最高置信匹配报告.md`。

## 代码与开发约定

- **语言**：代码注释、文档、报告、CLI 输出一律中文；源码文件多为 `# -*- coding: utf-8 -*-` 声明 + `from __future__ import annotations`。
- **依赖**：只加 `requirements.txt` 里有的东西；版本改动只动 `requirements.txt`，`pyproject.toml` 保持兼容范围。
- **绘图**：一律 `Agg` 无头后端（glm 在 `report.py`、kimi 在 `cli.py`、analysis 在 `common.py` 均已强制）——绝不依赖显示器；中文图用 WenQuanYi Micro Hei 字体（见 `analysis/common.py:setup_matplotlib`）。
- **两项目隔离**：不交叉导入；对称修复只在被明确要求时做。
- **产物**：模型、`*.npz`、仿真 CSV、报告目录均为可重新生成的输出，不入库。

## 测试约定

测试承载物理真值，是 DSP 改动的仲裁标准：高斯信号峭度 ≈ 3、正弦峭度 = 1.5、特征频率与 CWRU 公布值核对、SNR 定量验证、每种故障在两个 SNR 水平下端到端识别、健康信号零误报、数据加载器往返一致、CLI 冒烟测试。glm 侧另有 ML 测试（特征尺度不变性、防泄漏分组、两级架构行为、模型序列化 roundtrip）。**改 DSP/特征代码必须跑对应项目的测试**。注意 `bearing_glm_method/docs/design.md` 验证一节写的"69 项"是 ML 加入前的旧数，现实际为 100 项。

## 注意事项（正确性与安全）

- **三个输入绝不可凭猜测填写**：采样率 fs、转速 rpm/轴频、轴承几何。缺失会导致全部特征频率错位、结论完全无效（详见 `bearing_glm_method/SKILL.md`）。
- 纯离线库，无联网需求；无密钥/凭据管理问题。
- ML 模型在**仿真数据**上训练：仿真准确率只证明管线正确，不等于现场性能，部署前需用现场数据复验（glm 的 LOSO 报告、`--compare` 两法交叉对比即为此设计；两法不一致时必须提示人工复核）。
- 真实数据集标签为工厂声称值，结论对外表述时保留这一口径。
