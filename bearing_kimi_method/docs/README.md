# 原理文档目录

本系列文档系统讲解基于传统信号处理的滚动轴承故障诊断原理，与
`src/bearing_fault/` 中的代码实现一一对应。建议按顺序阅读。

| 章节 | 内容 | 对应代码 |
|---|---|---|
| [01 轴承故障机理与特征频率](01_bearing_faults.md) | 轴承结构、四类局部故障机理、特征频率运动学推导、SKF 6205 算例、滑差 | `bearing.py` |
| [02 振动信号数学模型与仿真器](02_signal_model.md) | Randall 冲击响应模型：冲击串、衰减振荡、幅值调制、滑差、信噪比定义 | `simulator.py` |
| [03 时域统计指标](03_time_domain.md) | RMS、峰值因子、峭度、裕度因子等指标的定义、物理意义与算例 | `features.py` |
| [04 包络分析与共振解调](04_envelope_analysis.md) | 共振解调的必要性、Hilbert 变换推导、包络谱、频带选择 | `envelope.py`、`spectrum.py` |
| [05 阶次分析](05_order_analysis.md) | 变转速下的频率涂抹、角域重采样原理与实现、阶次谱 | `order_analysis.py` |
| [06 时频分析](06_time_frequency.md) | STFT 与不确定性原理、复 Morlet CWT、尺度-频率映射、两者对比 | `timefreq.py` |
| [07 故障判别与诊断流程](07_diagnosis.md) | 谱峰匹配得分、谐波巧合与交叉项、马氏距离健康基线、完整流程 | `diagnosis.py`、`pipeline.py` |
| [08 机器学习故障识别](08_machine_learning.md) | 特征工程、IsolationForest/EllipticEnvelope 检测级、SVM/RF/kNN 分类级、两级级联与评估 | `ml/` |

配套的可执行教程见 `notebooks/` 目录。
