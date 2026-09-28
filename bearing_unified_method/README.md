# bearing-unified：统一轴承故障诊断（v2 盲梳检规则）

统一合并方案的正式实现（阶段一：DSP 主干）。以 glm 方案代码为迁移起点，
吸收 kimi 方案的零相位预处理与马氏距离健康基线，并内生化评审裁定的
**v2 判别规则**（蓝本：合并原型 unified_prototype.py）。本包**自包含**：
不依赖任何外部诊断包。两个姊妹项目与合并原型已从仓库删除，其代码与文档
历史见 git 仓库。

## v2 流水线

```
原始信号
  → 预处理（去直流 + 去趋势；转速自估/触发特征经宽带零相位带通）
  → 转速自估（包络谱 1×fr 峰，仅记录，失败回退窗中点；匹配不依赖它）
  → Kurtogram 自适应选带 → 零相位带通解调 → 包络谱
  →【盲梳检】不依赖任何理论频率：
     A. 观测峰驱动梳扫：局部邻域中位数显著峰（峰/局部中位 ≥ 5）为候选基频
        f0 ∈ [20,150] Hz，验证 k·f0（k=2..5）谐波族（±2% 或 1.5 谱线，
        w_k=1/k 加权）；下探偏好"能解释最多显著峰的最小基频"，同族一梳
     B. 倒频谱互证：q≈m/f0（m=1..3）显著 quefrency 峰
     → A、B 都确认 = 轴承故障特征存在（可多个梳）
  → 干扰防护：f0 与 k×fr_nominal 或工频 50 Hz 在 1.5 谱线内吻合、以及下探后
    基频 <20 Hz 的轴频族梳，打 suspected_interference 标记并排除（仍记录）
  → 有梳 → 判「异常-轴承」→【定位】频带落入法（转速窗×几何推出四个互不
    重叠频带：滚珠BSF / 外圈BPFO / 滚珠2×BSF / 内圈BPFI）+ 理论复核
    （fr 取窗内最接近者，准入 ±4%、谐波复核窗 ±4% 参数化）→ 外圈/内圈/
    滚珠 3 类置信度；空档走 fr_hyp=f0/(k·阶次) 补救；仍无解判「轴承-未定位」
  → 无梳 且 马氏距离触发（健康基线，域内标定，辅助判据）→「异常-非轴承(摩擦)」
  → 皆无 → 正常；证据皆弱 → 低置信标记
```

判别仅 3 类：外圈 BPFO / 内圈 BPFI / 滚珠 BSF（滚珠 2×BSF 主峰带归入滚珠），
不做保持架 FTF 判别（特征频率计算能力保留）。

## 安装与测试

```bash
pip install -e bearing_unified_method
python -m pytest            # 在 bearing_unified_method/ 下
```

## CLI 速览

```bash
# 仿真内圈故障并用现场台架轴承（FIELD7）分析
bearing-unified simulate --fault BPFI --snr 5 --seed 42 --out sim_bpfi.csv
bearing-unified analyze sim_bpfi.csv --fs 12000 --bearing FIELD7 --out report/

bearing-unified bearings --rpm 1150          # 内置轴承（含现场 FIELD7）及特征频率
bearing-unified bands --bearing FIELD7       # 转速窗×几何推出的四个定位频带
```

`analyze` 默认 v2 规则：输出 3 类置信度、检出梳明细、定位结果；马氏辅助
判据需域内健康样本标定（API：`baseline.HealthBaseline`），单文件 CLI 分析
时未启用则自动跳过并在提示中说明。

## ML 子系统（阶段二）

两阶段分层：Stage1 正常/异常（默认监督式，可选新奇检测后端
`--stage1 novelty`，只训健康样本的 EllipticEnvelope）→ Stage2 条件 3 分类，
联合置信度 = p_abnormal × P(fault|abnormal)。特征六组共 51 维
（T1 时域 9 / T2 物理谐波突出度 3 类×5=15 / T3 谱峭度 6 / T4 频谱形态 11 /
T5 包络形态 2 / T6 盲梳检结构化 8），消融开关 `--arch flat`、`--no-physical`
（去 T2）、`--no-comb`（去 T6）。评估协议：GroupKFold 防泄漏 + LOSO
留一转速 + 阈值工作点扫描 + 与 v2 DSP `--compare`。

```bash
bearing-unified ml-dataset --n-base 15 --windows 8 --out data/ds.npz
bearing-unified ml-train --dataset data/ds.npz --out models/          # 监督后端
bearing-unified ml-train --dataset data/ds.npz --stage1 novelty --out models_nov/
bearing-unified predict data.csv --fs 16000 --model models/model.joblib --compare
```

预测转速约定：**用户给定优先（--rpm/--shaft-freq），否则包络谱 1×fr 峰
自估、失败回退 1150 rpm**（该约定只服务 T2 物理特征组；v2 DSP 与 T6
盲梳检特征均不依赖转速）。

## 模块

| 模块 | 来源 | 职责 |
|---|---|---|
| `fault_freqs` | glm 迁移 + 现场轴承注册 | 轴承库与特征频率（FTF 保留、判别不用） |
| `indicators` | glm 迁移 | 9 项时域指标 |
| `preprocessing` | kimi 移植 | 去直流/去趋势/sosfiltfilt 零相位带通 |
| `spectrum` | glm 迁移 | 幅值谱/包络/包络谱/real_cepstrum |
| `kurtogram` | glm 迁移 | 谱峭度自适应选带 |
| `matcher` | glm 迁移 | 谐波匹配（容差参数化；v2 理论复核用 ±4%） |
| `combdet` | 新增（v2-A/B） | 显著峰拾取、盲梳扫、倒谱互证、干扰防护 |
| `localize` | 新增（v2 定位） | 频带表计算、频带落入定位、fr_hyp 补救、理论复核、3 类置信度合成 |
| `baseline` | kimi 移植 | 马氏距离健康基线 + 域内标定（辅助判据） |
| `simulate` | glm 迁移 | 恒速仿真（真值数据） |
| `loaders` | glm 迁移 | csv/txt/npy/npz/mat（含 CWRU） |
| `diagnose` | 重写 | v2 流水线编排，DiagnosisResult 全中间量 |
| `report` | glm 迁移并适配 | 四联图 + 中文文本 + JSON（v2 信息） |
| `cli` | glm 迁移并适配 | `bearing-unified` |
| `ml/` | glm 迁移并适配 + 新增 | 特征 51 维（含 T6 8 维）/ 数据集（FIELD7 网格防泄漏）/ 训练（监督 zoo + Stage1 新奇检测后端）/ 评估（GroupKFold/LOSO/工作点/v2 DSP 对比）/ 推理 |

ML 合并已完成（阶段二）：glm 两阶段监督式为骨架，kimi 新奇检测
（EllipticEnvelope）为 Stage1 可选后端（只训健康样本），评估协议统一为
GroupKFold + LOSO + 阈值工作点 + v2 DSP `--compare`。
