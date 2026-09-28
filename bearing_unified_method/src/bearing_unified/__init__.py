"""bearing_unified: 统一轴承故障诊断工具库（v2 盲梳检规则，阶段一 DSP 主干）。

模块总览
--------
- fault_freqs : 轴承几何与特征频率（内置库 + 现场台架轴承 FIELD7 注册）
- indicators  : 时域统计指标（RMS、峭度、峰值因子等 9 项）
- preprocessing: 去直流/去趋势/零相位带通（移植 kimi，sosfiltfilt）
- spectrum    : FFT 频谱、Hilbert 包络谱、实倒频谱
- kurtogram   : 谱峭度与 Kurtogram 自适应解调频带选择
- matcher     : 谐波匹配（容差参数化，v2 理论复核用）
- combdet     : 盲梳检（显著峰拾取/梳扫/倒谱互证/干扰防护）
- localize    : 频带落入定位与 3 类置信度合成
- baseline    : 马氏距离健康基线 + 域内标定（辅助判据）
- simulate    : 轴承故障振动信号仿真（恒速）
- loaders     : CSV/TXT/NPY/NPZ/MAT 数据加载（含 CWRU）
- diagnose    : v2 诊断流水线编排
- report      : 图表与文本报告输出
- cli         : 命令行入口 bearing-unified
"""

__version__ = "0.2.0"
