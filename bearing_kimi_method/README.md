# 轴承故障诊断（传统信号处理）

基于**传统信号处理**方法的滚动轴承故障诊断 Python 项目：内置符合经典
Randall 冲击响应模型的故障信号仿真器，提供从时域特征、共振解调包络分析、
阶次分析、时频分析到故障判别的完整工具链，附命令行工具、Jupyter 教程与
七章中文原理文档。

## 功能总览

| 模块 | 内容 |
|---|---|
| `bearing` | 轴承几何参数与故障特征频率（BPFO / BPFI / BSF / FTF） |
| `simulator` | 故障信号仿真器：冲击串 + 共振响应 + 滑差 + 调制 + 定量 SNR，支持变速工况 |
| `features` | 时域统计特征：RMS、峭度、峰值因子、脉冲因子、裕度因子等 |
| `envelope` | 共振解调包络分析（带通滤波 → Hilbert 变换 → 包络谱） |
| `spectrum` | 幅值谱/功率谱/包络谱、特征频率谐波谱峰拾取 |
| `order_analysis` | 变转速阶次分析（角域重采样 → 阶次谱） |
| `timefreq` | STFT 时频谱、自实现复 Morlet CWT |
| `diagnosis` | 谱峰匹配故障定位 + 马氏距离健康基线判别 |
| `pipeline` / `cli` | 端到端诊断流程与 `bearing-fault` 命令行 |
| `ml` | 机器学习子系统：26 维特征提取、批量数据集生成、检测(IsolationForest/EllipticEnvelope)+分类(SVM/RF/kNN) 两级模型、评估 |

## 安装

推荐使用**仓库根目录**的统一 conda 环境（与 `bearing_glm_method` 共用一个
环境，依赖版本见根目录 [requirements.txt](../requirements.txt)）：

```bash
conda env create -f environment.yml   # 在仓库根目录执行；两项目均已装入
conda activate bearing-detection      # bearing-fault 命令直接可用
```

不用 conda 也可以单独安装本包：

```bash
pip install -e .          # 安装本包（含 bearing-fault 命令）
pip install -e .[dev]     # 附加开发依赖：pytest、nbclient、jupyter
```

## 快速上手

### 命令行

```bash
# 生成一段外圈故障仿真信号（保存 signal.npz 与波形图）
bearing-fault simulate --fault outer --out out_sim

# 对内圈故障信号执行完整诊断（输出图表 + summary.json + 终端报告）
bearing-fault diagnose --fault inner --snr -6 --out out_diag

# 对已有信号执行传统谱峰诊断
bearing-fault diagnose --input out_sim/signal.npz --out out_diag2

# 机器学习：数据集 → 两级训练（检测+分类）→ 推理
bearing-fault ml-dataset --out data/ml_dataset.npz      # 360 样本特征数据集
bearing-fault ml-train --dataset data/ml_dataset.npz --out models/
bearing-fault ml-predict --model models/two_stage.joblib --fault inner
```

### Python API

```python
from bearing_fault import Bearing, BearingSimulator, DiagnosisPipeline

bearing = Bearing.sk6205()                    # CWRU 数据集轴承
print(bearing.characteristic_frequencies(29.95))   # 1797 rpm 下的理论特征频率

sim = BearingSimulator(fs=12000, duration=5.0, rpm=1797, seed=42)
x = sim.signal(fault="inner", snr_db=-6)      # 内圈故障，SNR = -6 dB

report = DiagnosisPipeline(fs=12000).run(x, rpm=1797)
print(report.verdict)                          # -> "inner"
print(report.scores)                           # 各类故障谱峰匹配得分
print(report.summary())                        # 可 JSON 序列化的完整摘要
```

## 教程与文档

- `notebooks/`：7 个循序渐进的 Jupyter 教程（仿真 → 时域特征 → 包络分析
  → 阶次分析 → 时频分析 → 完整诊断流程 → 机器学习两级识别）
- `docs/`：八章原理文档，从故障机理与特征频率推导到 Hilbert 变换、
  共振解调、阶次分析、时频分析、诊断判据与机器学习，见 [docs/README.md](docs/README.md)

## 测试

```bash
python3 -m pytest -q
```

覆盖特征频率公式（对照文献系数）、仿真信号 SNR 定量与周期性、包络谱峰值
命中、阶次谱、时频定位、四类故障分类正确性与 CLI 冒烟测试。
