# 机器学习轴承故障识别子系统 — 设计规格

日期：2026-09-19
状态：已获用户批准（2026-09-19）
所属项目：bearing-fault-diagnosis（传统信号处理轴承故障诊断包）

## 1. 目标与范围

在现有传统信号处理项目基础上，新增**经典机器学习**子系统 `bearing_fault.ml`，
实现"检测 + 分类"两级轴承故障识别：

- **检测级**：判断信号是否异常（健康 vs 故障），只用健康样本训练；
- **分类级**：对异常样本判定故障类型（内圈 / 外圈 / 滚动体 / 保持架）。

技术路线已确认：经典 ML（sklearn），不做深度学习；评估采用常规随机划分
准确率方案（不做跨工况泛化评估）。

## 2. 已确认的需求决策

| 决策点 | 结论 |
|---|---|
| 定位 | 实现为项目新模块 `src/bearing_fault/ml/` |
| 任务目标 | 检测 + 分类两级串联 |
| 总体方案 | 方案 A：IsolationForest 检测 + SVM/RF/kNN 择优分类，全 sklearn Pipeline |
| 模型路线 | 经典 ML，手工特征工程 |
| 评估 | 分层 70/30 随机划分 + 5 折交叉验证 + 混淆矩阵 |
| 数据来源 | 现有 `BearingSimulator` 按参数网格批量生成 |

## 3. 架构

`ml/` 子包单向依赖现有 DSP 模块（simulator / features / envelope /
spectrum / diagnosis），不修改任何已有代码：

```
参数网格批量仿真 (dataset.py)
      ↓ 每段 3 s 信号
特征提取 26 维 (features.py)
      ↓ X, y, meta → dataset.npz
两级训练 (train.py)
      ├─ Stage1: Pipeline(StandardScaler → IsolationForest)  仅健康样本
      └─ Stage2: Pipeline(StandardScaler → SVM/RF/kNN 择优)  4 类故障样本
      ↓ joblib 保存
推理 (model.py): TwoStageModel.predict_one(signal, fs, rpm)
      ↓
评估 (evaluate.py): 级联 5 类混淆矩阵、classification report、
                    检测级误报率/检出率、RF 特征重要性
```

级联推理逻辑：stage1 判 `-1`（异常）→ 进入 stage2 输出故障类型；
stage1 判 `+1`（正常）→ 直接输出 healthy。

## 4. 模块设计

### 4.1 `ml/dataset.py` — 数据集生成

- `DatasetConfig`：faults（5 类）、rpms、snrs、seeds 网格参数；
  默认 5 × {1200, 1500, 1797} rpm × {-6, 0, 6} dB × 8 seeds = **360 样本**；
  每段 fs=12000 Hz、duration=3 s、cage 故障 3 s 内约 35 次冲击，足够提取特征。
- `generate(config) -> (X, y, meta)`：逐样本仿真 + 特征提取；
  meta 记录每样本的 fault/rpm/snr/seed。
- `save_npz / load_npz`：数据集落盘与读取 round-trip。

### 4.2 `ml/features.py` — 特征提取（26 维）

`FeatureExtractor(fs=12000.0, band=None).extract(x, rpm) -> dict[str, float]`：

| 组 | 维数 | 内容 | 来源 |
|---|---|---|---|
| 时域 | 10 | rms, peak, peak_to_peak, std, kurtosis, skewness, crest_factor, impulse_factor, shape_factor, clearance_factor | `features.time_domain_features` |
| 包络谱 | 13 | 4 类故障频率 × 1–3 次谐波峰地比（peak/median floor）+ 最大峰地比 | `envelope.envelope_analysis` + `spectrum.find_peaks_near` |
| 频域 | 3 | 谱质心、谱峭度、共振带能量占比 | `spectrum.amplitude_spectrum` |

- 特征名固定顺序常量 `FEATURE_NAMES`（26 项），训练与推理共用，防止特征错位。
- 包络谱峰地比与 `diagnosis.fault_scores` 同一定义（峰值/幅值中位数），
  但输出原始特征值而非聚合得分。

### 4.3 `ml/train.py` — 训练

- `train_two_stage(X, y, meta, test_size=0.3, seed=42) -> TrainResult`：
  - 分层随机划分 70/30（全工况混合，常规评估）；
  - 检测级：训练集健康样本拟合 `Pipeline(StandardScaler, IsolationForest(n_estimators=200, contamination=0.01, random_state))`，
    **仅使用检测特征子集** `DETECTION_FEATURE_NAMES = (kurtosis, crest_factor, clearance_factor, env_max_ratio)`。
    检测级算法通过 `train_two_stage(..., stage1=...)` 与 CLI `--stage1` 可选：
    `elliptic_envelope`（**默认**，稳健协方差/马氏距离，距离感知）或
    `isolation_forest`（备选）。
    设计原因（实现期两轮实测修正）：
    1. 全 26 维下健康样本有近零方差特征
    （rms/std/kurtosis/shape_factor/spec_centroid/spec_kurtosis/band_energy_ratio），
    标准化后被放大为虚假高方差，IsolationForest 在 26 维稀疏空间过拟合，
    健康误报率高达 0.73；改用 4 维冲击性特征子集 + contamination=0.01 后
    误报率降为 0（故障在 env_max_ratio 上为 80–300 倍地板 vs 健康 5–12 倍，
    收紧阈值不损失检出率）；
    2. 进一步网格扫描（2 转速 × 2 SNR × {4,6,8} 种子 × 2 时长）发现
    IsolationForest 在一半配置下检出率塌缩为 0（分裂界由训练数据 min/max
    决定、不随极端程度外推 + 阈值标定随样本数漂移），而 EllipticEnvelope
    在全部 6 种配置下检出率 1.0。经用户批准，默认检测级改为
    EllipticEnvelope，IF 保留为备选并在文档中记录风险。
  - 分类级：训练集故障样本上对 SVM(RBF, C=1) / RandomForest(200 树) /
    kNN(k=5) 做 5 折分层交叉验证，取平均准确率最高者，全量故障训练样本重训；
  - `TrainResult`：模型对象、交叉验证对比表、测试集预测结果。
- 模型保存：joblib dump `{stage1, stage2, feature_names, config_meta}` 单文件。

### 4.4 `ml/model.py` — 推理

- `TwoStageModel(stage1, stage2, feature_names, fs=12000.0)`：
  - `predict_one(x, rpm) -> {"verdict": "healthy"|故障类型, "anomaly": bool, "stage2_label": ...}`；
  - `predict_batch(X) -> labels`；
  - `load(path)` 类方法从 joblib 恢复。

### 4.5 `ml/evaluate.py` — 评估

- 级联 5 类指标：混淆矩阵（matplotlib 出图）、sklearn classification_report；
- 检测级单列：健康误报率（FP rate）、故障检出率（recall@fault）；
- 分类级为 RF 时输出特征重要性 Top-10 表；
- `evaluate(result, out_dir)` 落盘：confusion_matrix.png + metrics.json。

### 4.6 CLI 扩展（`cli.py` 增加三个子命令）

- `bearing-fault ml-dataset --out data/ml_dataset.npz`（网格参数可覆盖）
- `bearing-fault ml-train --dataset data/ml_dataset.npz --out models/`
  → 保存 `models/two_stage.joblib` + `models/metrics.json` + 混淆矩阵图
- `bearing-fault ml-predict --model models/two_stage.joblib [--fault outer 或 --input signal.npz]`

## 5. 依赖变更

`pyproject.toml` dependencies 增加 `scikit-learn>=1.3`（joblib 随附其中）。
不引入 pandas 硬依赖（数据集用 numpy structured array / npz 即可）。

## 6. 测试（`tests/test_ml.py`）

- 特征提取：26 维、名称与 FEATURE_NAMES 一致、健康/故障信号特征可区分；
- 数据集：小网格生成形状正确、save/load round-trip 一致；
- 训练冒烟：缩小网格（5 类 × 1 转速 × 2 SNR × 8 种子 = 80 样本、
  duration 1.5 s）完整跑通 train_two_stage + evaluate。
  注意：健康样本需 ≥ 约 16 个，IsolationForest 检测级阈值标定才稳定
  （实测 5–6 个健康样本时极端故障点被误判为健康——IF 的分裂界不随
  极端程度外推，是其固有盲区；需要距离感知的场景可用备选检测级
  `stage1="elliptic_envelope"`）。
- 推理正确性：易分条件（rpm=1797、snr=0 dB、固定 seed）下，
  healthy → healthy、outer → outer；
- CLI 冒烟：ml-dataset / ml-train / ml-predict 依次跑通且产物存在。

## 7. Notebook 与文档

- `notebooks/07_machine_learning.ipynb`：全流程演示——数据集生成、
  特征分布（健康 vs 故障对比图）、两级训练、混淆矩阵、特征重要性。
  用 nbclient 无头执行验证。
- `docs/08_machine_learning.md`：ML 章——为什么这些特征有效
  （与前 7 章 DSP 原理的对应关系）、IsolationForest / SVM / 随机森林原理
  简述、两级架构的工程 rationale、评估方法与结果解读；
  `docs/README.md` 索引追加第 08 章。

## 8. 验收标准

1. 默认网格（360 样本）上级联 5 类准确率 > 95%，混淆矩阵近似对角；
2. 检测级：健康误报率 < 10%，故障检出率 > 95%；
3. `pytest` 全部通过（含既有 74 项无回归）；
4. CLI 三个新子命令端到端跑通；
5. notebook 07 无头执行无异常；
6. `docs/08` 公式与代码实现一致。

## 9. 非目标（YAGNI）

- 不做深度学习 / CNN 基线；
- 不做跨工况泛化评估（留作后续增强）；
- 不做超参数自动搜索（GridSearchCV 大规模调参）——仅三模型交叉验证择优；
- 不支持真实数据集加载（CWRU 等），仅仿真数据。
