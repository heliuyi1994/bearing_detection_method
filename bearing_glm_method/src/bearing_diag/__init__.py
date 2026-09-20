"""bearing_diag: 基于传统信号处理的滚动轴承故障诊断工具库。

模块总览
--------
- fault_freqs : 轴承几何参数与故障特征频率（BPFO/BPFI/BSF/FTF）计算
- indicators  : 时域统计指标（RMS、峭度、峰值因子等 9 项）
- spectrum    : FFT 频谱、Hilbert 包络谱、实倒频谱
- kurtogram   : 谱峭度与 Kurtogram 自适应解调频带选择
- simulate    : 轴承故障振动信号仿真（共振衰减冲击模型）
- matcher     : 包络谱谐波匹配与故障部位判别
- loaders     : CSV/TXT/MAT 数据加载（含 CWRU 格式解析）
- diagnose    : 诊断流水线编排
- report      : 图表与文本报告输出
- cli         : 命令行入口
"""

__version__ = "0.1.0"
