# bearing-diag：滚动轴承故障诊断（传统信号处理 + 机器学习）

双路线轴承故障诊断工具库与 AI 助手技能：

- **传统路线**：时域指标 → Kurtogram 自适应选带 → Hilbert 包络解调谱 →
  故障特征频率谐波匹配 → 判定 + 置信度 + 可视化报告（物理机理可解释）；
- **机器学习路线（两级层次分类）**：48 维双轨特征（物理判别特征 + 统计形态
  特征）→ **Stage1 正常/异常检测器（阈值可调，漏报/误报分开统计）→ Stage2
  四类故障分类（条件概率）**，防泄漏 GroupKFold + LOSO 跨工况验证，与传统
  方法可交叉对比（`--arch flat` 保留单级 5 分类做 A/B）。

无需联网；传统路线零训练数据，ML 路线用内置仿真增强数据（也可混入真实数据）。

> 深入原理（公式推导、物理意义、方法局限、ML 特征工程与评估协议）见
> **[docs/principles.md](docs/principles.md)**。

## 环境管理（conda）

环境由**仓库根目录**统一管理（与 `bearing_kimi_method` 共用一个环境）。在
仓库根目录执行：

```bash
conda env create -f environment.yml     # 创建环境 bearing-detection 并安装（首次）
conda activate bearing-detection        # 日常使用
conda env update -f environment.yml     # 依赖变更后更新
conda env remove -n bearing-detection   # 删除（可再 create 重建）
```

依赖版本以根目录 [requirements.txt](../requirements.txt) 为单一来源，两个项目
均以可编辑方式装入同一环境，两套 CLI（`bearing-diag` / `bearing-fault`）同时
可用。

不用 conda 也可以：`python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`。

依赖：numpy、scipy、matplotlib、scikit-learn（Python ≥ 3.10）；开发附加
`pytest`。下文所有 `bearing-diag` / `python` 命令均默认在激活环境中执行。

## 快速上手

```bash
# 1) 生成一个已知答案的仿真信号（外圈故障，SNR 5 dB）
bearing-diag simulate --fault BPFO --snr 5 --seed 42 --out sim_bpfo.csv

# 2) 诊断：需要数据 + 采样率 + 转速 + 轴承型号
bearing-diag analyze sim_bpfo.csv --fs 12000 --rpm 1797 --bearing SKF6205 --out report/
#    输出: report/diagnosis_report.{png,txt,json}

# 3) 查看内置轴承参数与任意转速下的特征频率
bearing-diag bearings --rpm 1797

# 4) 一键演示五种状态（正常 + 四种故障）的仿真与诊断
python examples/run_demo.py
```

分析真实数据时把 `--bearing` 换成实际型号（或 `--bearing-json` 自定义几何参
数、或 `--fault-freqs` 直接给特征频率），`--fs` 与 `--rpm` 按采集条件给定。
支持 csv/txt/npy/npz/mat（含 CWRU 命名的 `*_DE_time` 变量自动识别）。

## 机器学习路线

```bash
# 1) 训练（两级架构默认：检测器+分类器各自选优，分级评估+LOSO+传统方法对比）
bearing-diag ml-train --out models/          # ~5 分钟，600 样本
#    --arch flat       单级 5 分类（A/B 对比）
#    --no-physical     消融：纯统计盲特征（部署时无需转速信息）

# 2) 预测（两级输出 + 概率 + 贡献特征；--compare 并列传统方法）
bearing-diag predict data.csv --fs 12000 --model models/model.joblib \
    --rpm 1797 --bearing SKF6205 --compare
#    --threshold 0.3   Stage1 报警阈值（调低→更灵敏，调高→更保守）
```

特征五组共 48 维：时域指标 9 + 特征频率谐波突出度 20（物理轨）+ 谱峭度 6 +
频谱形态 11 + 包络谱形态 2。分类为两级层次架构：**Stage1 正常/异常检测器**
（阈值可调，训练报告给出 0.3/0.5/0.7 三个工作点的漏报率与误报率）→
**Stage2 四类故障分类**（条件概率），联合置信度 = p_abnormal × P(fault|abnormal)。
评估三层：分级 GroupKFold 防泄漏交叉验证、两级 + 端到端 LOSO、独立测试集与
传统方法对比。真实数据按目录约定混入训练：`root/<标签>/*.csv` +
`manifest.json`（fs/rpm/bearing），详见 `ml.dataset.load_dataset_dir`。

本仓库默认规模训练结果：Stage1 三个阈值工作点漏报率/误报率均为 0，Stage2 与
端到端 GroupKFold、LOSO 均 1.0，独立测试集与传统方法一致率 100%（仿真数据
信息充分；真实场景性能需用现场数据检验——这正是两法交叉对比设计的原因）。

## Python API（机器学习）

```python
from bearing_diag import simulate, diagnose
from bearing_diag.report import write_reports

sig, info = simulate("BPFI", fs=12000, duration=2.0, snr_db=5, seed=1)
result = diagnose(sig, fs=info["fs"], shaft_freq=info["shaft_freq"],
                  bearing="SKF6205")

print(result.verdict)        # 'BPFI'
print(result.verdict_cn)     # '内圈故障'
print(result.confidence)     # 0.93
print(result.indicators)     # 9 项时域指标
write_reports(result, "report/")
```

模块一览：

| 模块 | 职责 |
|---|---|
| `fault_freqs` | 轴承参数库、BPFO/BPFI/BSF/FTF 特征频率 |
| `indicators` | 9 项时域统计指标、去趋势 |
| `spectrum` | FFT 幅值谱、包络谱、倒频谱 |
| `kurtogram` | 谱峭度选带（层级树搜索） |
| `simulate` | 共振衰减冲击串故障仿真 |
| `matcher` | 谐波匹配、整数倍混淆消解、置信度 |
| `loaders` | CSV/TXT/NPY/NPZ/MAT(CWRU) 加载 |
| `diagnose` | 流水线编排 |
| `ml/features` | ML 特征工程（T1~T5 五组 48 维） |
| `ml/dataset` | 仿真增强数据集（防泄漏分组）、真实数据目录加载 |
| `ml/train` / `ml/evaluate` / `ml/infer` | 两级训练管线 / 分级评估与 LOSO / 两级推理 |
| `report` | 四联图 + 中文文本报告 + JSON |
| `cli` | 命令行入口 `bearing-diag` |

## 测试

```bash
python -m pytest
```

100 项测试，包括：指标理论值校验（高斯峭度≈3、正弦峭度=1.5）、特征频率与
CWRU 官方公布值逐项对照、四种故障 × 两档 SNR 端到端判型、正常信号无误报、
FTF/BPFO 整数倍混淆消解、加载器 round-trip、CLI 冒烟；ML 侧：特征尺度不变
性与物理语义、数据集防泄漏分组结构、两级架构（分级指标、阈值门控行为、
联合概率归一、LOSO、旧模型包兼容）、模型序列化 roundtrip、预测与传统方法
一致性。

## 作为 AI 助手技能使用

项目根目录的 [SKILL.md](SKILL.md) 是技能入口。安装到用户级技能目录：

```bash
mkdir -p ~/.agents/skills/bearing-fault-diagnosis
cp SKILL.md ~/.agents/skills/bearing-fault-diagnosis/
# 编辑其中 PROJECT_DIR 指向本项目的绝对路径
```

此后 AI 助手遇到"轴承故障诊断/振动信号分析"类请求时会按 SKILL.md 指引调用
本项目 CLI。

## 局限

变转速工况（需阶次跟踪）、分布式磨损故障、多故障耦合与强齿轮干扰场景超出
本库设计范围（传统方法见原理文档第 10.3 节，ML 方法见第 11.8 节）。判定
结果应结合包络谱图人工复核；ML 模型在仿真数据上训练，部署前需用现场数据
复验。
