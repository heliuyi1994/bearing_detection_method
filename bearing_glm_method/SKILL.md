---
name: bearing-fault-diagnosis
description: 滚动轴承故障诊断技能：当用户需要分析轴承振动信号、诊断轴承故障部位（内圈/外圈/滚动体/保持架）、计算轴承特征频率（BPFO/BPFI/BSF/FTF）、做包络谱/共振解调/峭度/时域指标分析、训练或使用机器学习故障分类模型，或处理 CWRU 等轴承数据集时使用。含传统信号处理全流程与 ML 双路线，无需训练数据（传统路线）。
---

# 轴承故障诊断（传统信号处理）

本技能通过 `bearing-diag` CLI（Python 项目 `bearing_diag`）对轴承振动信号执行
经典信号处理诊断流程，输出故障部位判定、置信度与可视化报告。

**项目目录（PROJECT_DIR）**：`/home/dayin/git/bearing_detection/bearing_glm_method`
（安装本技能到其他机器时，请把下述命令中的路径改为项目实际位置）

环境为 conda 统一管理（在仓库根目录
`/home/dayin/git/bearing_detection` 执行 `conda env create -f environment.yml`
创建，名称 `bearing-detection`，与 `bearing_kimi_method` 共用）。两种调用方式：

```bash
BD="/home/dayin/git/bearing_detection/bearing_glm_method"
BDIAG="conda run -n bearing-detection bearing-diag"   # 无需激活；或先 conda activate
# 等价：$BD/.venv/bin/bearing-diag 仅当机器上还留有旧 venv，优先用 conda
```

## 何时使用

- 用户给出振动数据（csv/txt/npy/npz/mat，含 CWRU 数据集），问"轴承是否有故障/什么故障"；
- 用户问轴承特征频率计算（提供转速与轴承型号时可直接 `$BDIAG bearings --rpm <rpm>`）；
- 用户要求包络谱、峭度、Kurtogram、时域指标等传统信号分析；
- 用户想体验/演示：用 `simulate` 生成已知答案的仿真信号再诊断。

不适用：变转速工况（需阶次跟踪）、纯磨损类分布式故障、已标注数据的机器学习分类。

## 使用前必须向用户确认的三项输入

1. **采样率 fs**（Hz）——数据文件或用户提供；CSV 首列为时间时可自动推断；
2. **转速 rpm 或轴频 fr**（fr = rpm/60）——诊断必需，特征频率全部正比于它；
3. **轴承几何参数**——三选一：内置型号（`$BDIAG bearings` 列表，含
   SKF6205/ER16K 等 CWRU 常用型号）、自定义 JSON、或用户直接给出特征频率。

三项不全时**不要猜**：fs 或 rpm 缺失会导致所有特征频率错位、结论完全无效。
轴承型号未知时，可请用户提供节圆直径 D、滚动体直径 d、滚动体数 n（铭牌或手
册可查），写成 JSON 传入。

## 标准流程

```bash
# 分析（Kurtogram 自动选带，全流程自动）
$BDIAG analyze <数据文件> --fs <Hz> --rpm <rpm> --bearing <型号> --out report/
#   或 --shaft-freq <Hz> 代替 --rpm
#   或 --bearing-json <params.json> / --fault-freqs '{"BPFO":..,...}' 代替 --bearing
#   可选: --band LOW:HIGH 手动指定解调频带; --column N 多列数据选列;
#         --kurtogram-level 4; --harmonics 5

# 仿真体验（用户没有数据但想看效果时）
$BDIAG simulate --fault BPFO --snr 5 --out sim.csv
$BDIAG analyze sim.csv --fs 12000 --rpm 1797 --bearing SKF6205
```

### 机器学习路线（可选）

```bash
# 训练（两级架构默认：先正常/异常检测，再四类故障分类；首次约 5 分钟）
$BDIAG ml-train --out models/
#   --arch flat       单级 5 分类（A/B 对比用）
#   --no-physical     消融：纯统计盲特征（部署时无需转速/轴承参数）
#   --n-base/--windows 调节数据规模

# 预测：两级输出（p_abnormal + 故障条件概率 + 联合置信度）
$BDIAG predict <数据> --fs 12000 --model models/model.joblib \
    --rpm 1797 --bearing SKF6205 --compare
#   --threshold 0.3   Stage1 报警阈值：调低→更灵敏（漏报↓误报↑），调高→保守
```

ML 使用要点：
- **物理特征模型（默认）预测时必须提供 --rpm 与 --bearing**，与训练配置一致；
  盲特征模型（--no-physical 训练）则不需要；
- **两级输出解读**：`p_abnormal` 是报警依据，`Stage2 条件概率` 只在判异常时
  有意义；训练报告（training_report.json 的 hierarchical_cv）给出 0.3/0.5/0.7
  阈值工作点的漏报率/误报率，帮用户按业务代价选 `--threshold`（漏报=故障判
  正常，通常代价最高；误报=正常判异常）；
- 模型在**仿真数据**上训练：准确率指标证明管线正确，**不等于现场性能**，
  部署前应用现场数据复验（LOSO 报告见 training_report.json）；
- `--compare` 两法不一致时**必须提示用户人工复核**，不一致常意味着分布外
  样本（未知工况或非轴承振源）；
- 真实标注数据可按目录约定混入重训：`root/<标签>/*.csv` + `manifest.json`
  （含 fs/rpm/bearing），用 `ml.dataset.load_dataset_dir` 加载。

stdout 是完整中文文本报告；`report/` 下另有 `diagnosis_report.png`（四联图：
时域波形+指标表 / FFT谱+解调频带 / Kurtogram / 包络谱+谐波标注）与
`diagnosis_report.json`（机器可读结果，含每个候选的得分、命中谐波明细）。

## 结果解读要点（向用户解释时）

- **判定与置信度**：`verdict` ∈ normal/BPFO/BPFI/BSF/FTF；置信度 ≥0.5 为
  "明确"，0.25~0.5 为"疑似"（建议复核）。normal 仅表示"未发现特征频率谐
  波族"，不等于轴承健康；
- **四个判定分别对应**：BPFO 外圈、BPFI 内圈、BSF 滚动体、FTF 保持架；
- **时域峭度**是辅助证据：>6 有显著冲击；但峭度正常而包络谱匹配出谐波族、
  或峭度高而匹配不到特征频率时，报告的"提示"部分会给出矛盾说明——照实转述；
- **最终裁决看包络谱图**：判定故障的谐波族（红色竖线）应当等间隔、高出邻域
  明显。人工复核 PNG 是流程的一部分，不要跳过；
- **低信噪比下**本库宁可漏报（判 normal）也不错报部位；报告 notes 里会说明
  数据质量问题（时长不足、分辨率不够等）。

## 常见问题排查

| 现象 | 处理 |
|---|---|
| 未知轴承型号报错 | 用 `--bearing-json` 传自定义参数，或让用户提供特征频率 |
| 提示"低于频率分辨率可检测下限" | 采集时长不足（判 FTF 需 ≥4 s 数据），建议加长 |
| 判 normal 但峭度很高 | 冲击可能来自轴承之外（齿轮/松动/碰摩），转述报告提示 |
| CWRU 的 .mat 加载 | 直接传入即可自动找 `*_DE_time`；fs 必须显式给（12k 文件 12000，48k 文件 48000） |
| 多列 CSV 取错列 | 用 `--column N` 指定（默认第 0 列，首列为时间时自动取第 1 列） |

## 深入原理

公式推导（特征频率几何推导、Hilbert 变换、谱峭度、谐波匹配判据设计等）与
方法局限详见 `$BD/docs/principles.md`。回答用户"为什么/怎么算"类问题时以
该文档为准，不要凭记忆推导。
