# CLAUDE.md

本文件为 Claude Code(claude.ai/code)在本仓库中处理代码时提供指导。

## 仓库结构

Git 仓库根目录即本目录,包含两个相互独立、各自完备的姊妹项目,实现**同一个任务**(基于振动信号的滚动轴承故障诊断,采用传统信号处理 + 机器学习方法),二者分开开发以便并排对比。没有共享代码——绝不要在两者之间交叉导入;仅在被明确要求时才保持修复/功能同步。

| 目录 | 包名 | CLI |
|---|---|---|
| `bearing_glm_method/` | `bearing_diag`(src 布局) | `bearing-diag` |
| `bearing_kimi_method/` | `bearing_fault`(src 布局) | `bearing-fault` |

所有文档、报告和面向用户的输出均为中文——请保持一致。

## 环境与命令

两个项目共用一个 conda 环境;版本锁定写在根目录的 `requirements.txt` 中(唯一事实来源;各项目的 `pyproject.toml` 只保留兼容范围)。环境由根目录的 `environment.yml` 定义,会以可编辑方式 pip 安装两个包——两个 CLI 可同时使用:

```bash
conda env create -f environment.yml   # 在仓库根目录执行;创建环境 bearing-detection
conda env update -f environment.yml   # 依赖变更后执行
```

请进入各自项目目录,使用 `bearing-detection` 环境运行命令:

```bash
# 测试——glm(100 个测试)/ kimi(88 个测试)
cd bearing_glm_method && conda run -n bearing-detection python -m pytest
cd bearing_kimi_method && conda run -n bearing-detection python -m pytest -q
# 单个文件 / 关键字:追加 tests/test_kurtogram.py 或 -k kurtogram

# glm:分析真实/仿真数据(需要采样率 + 转速 + 轴承型号)
conda run -n bearing-detection bearing-diag simulate --fault BPFO --snr 5 --seed 42 --out sim_bpfo.csv
conda run -n bearing-detection bearing-diag analyze sim_bpfo.csv --fs 12000 --rpm 1797 --bearing SKF6205 --out report/
conda run -n bearing-detection bearing-diag bearings --rpm 1797     # 内置轴承及故障特征频率
conda run -n bearing-detection bearing-diag ml-train --out models/  # 约 5 分钟,600 个仿真样本
conda run -n bearing-detection python examples/run_demo.py          # 5 种状态演示

# kimi:仿真 + 完整诊断;ML 数据集/训练/预测
conda run -n bearing-detection bearing-fault diagnose --fault inner --snr -6 --out out_diag
conda run -n bearing-detection bearing-fault ml-dataset --out data/ml_dataset.npz
conda run -n bearing-detection bearing-fault ml-train --dataset data/ml_dataset.npz --out models/
```

`ml-train`、`ml-dataset` 和演示脚本会重新生成产物(`models*/`、`data/*.npz`)——它们是输出而非源文件,已被 gitignore。

## 领域流水线(两个项目共有)

核心流程相同;以下每个模块在两个包中都存在:

1. **轴承几何 → 特征故障频率**:BPFO(外圈)、BPFI(内圈)、BSF(滚动体自转)、FTF(保持架)——均与轴频(rpm/60)成正比。
2. **仿真器**:Randall 冲激响应模型(冲击序列 + 衰减共振 + 滑动 + 幅值调制 + 加性噪声),SNR 定量可控——用于生成真值测试数据和 ML 训练数据。
3. **时域指标**:RMS、峭度、峰值因子、脉冲因子、裕度因子等。
4. **包络解调(共振解调)**:带通滤波 → Hilbert 变换 → 包络谱。
5. **谐波匹配**:在包络谱中寻找各候选故障频率处的谐波族 → 结论 + 置信度 → 图表/文本/JSON 报告。

### 项目间差异

**glm(`bearing_diag`)**——设计决策记录于 `docs/design.md`(D1–D13),理论说明在 `docs/principles.md`:
- 谱峭度图(Kurtogram,谱峭度树搜索)用于自适应频带选择;含倒频谱模块。
- 谐波匹配采用局部邻域中值显著度基线,以及整数倍混淆消歧(BPFO ≈ n×FTF 的歧义);包络峭度基线取 2.0 而非 3.0。
- 低信噪比策略:宁可漏检,也不错判故障类型。
- ML(`src/bearing_diag/ml/`):48 维双通道特征(物理 + 统计);**两阶段分层**架构——阶段 1 为正常/异常检测器,阈值可调(`--threshold`,报告给出 0.3/0.5/0.7 下的漏检/误报率),阶段 2 为条件化的 4 分类;联合置信度 = p_abnormal × P(fault|abnormal)。评估方式:GroupKFold(防泄漏分组)+ LOSO 跨工况泛化 + 留出测试集与传统方法对比(`--compare`)。消融实验:`--arch flat`、`--no-physical`(盲特征,推理时无需转速)。
- 项目根目录的 `SKILL.md` 是 AI 助手技能入口(记录了绝不可凭猜测填写的三个输入:fs、rpm/轴频、轴承几何)。

**kimi(`bearing_fault`)**——`docs/` 下的 8 章理论文档(每个模块一章),`notebooks/` 下有教程 Notebook:
- 阶次分析(面向变速工况的角度域重采样)和 STFT + Morlet CWT 时频模块——glm 均不具备。
- 诊断额外增加了马氏距离健康基线。
- ML(`src/bearing_fault/ml/`):26 维特征;两阶段 = IsolationForest/EllipticEnvelope 异常检测 + SVM/RF/kNN 分类。

### 项目间命名对照

| 概念 | glm | kimi |
|---|---|---|
| 故障标签 | `BPFO` / `BPFI` / `BSF` / `FTF` / `normal` | `outer` / `inner` / `ball` / `cage` / `healthy` |
| CWRU 6205 轴承 | `"SKF6205"`(字符串,宽松匹配) | `Bearing.sk6205()` |
| 完整诊断 | `bearing-diag analyze` / `diagnose()` | `bearing-fault diagnose` / `DiagnosisPipeline.run()` |

## 测试约定

测试承载了物理真值,是 DSP 改动的仲裁标准:高斯信号峭度 ≈ 3,正弦信号峭度 = 1.5,特征频率与 CWRU 公布值核对,SNR 定量验证,每种故障在两个 SNR 水平下进行端到端故障类型识别,健康信号零误报,数据加载器往返一致,CLI 冒烟测试。生成的图表走 `Agg` matplotlib 后端(两个 CLI 中均已强制指定)——绝不依赖显示器。
