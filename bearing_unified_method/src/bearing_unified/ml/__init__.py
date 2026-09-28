# -*- coding: utf-8 -*-
"""bearing_unified.ml：机器学习故障识别子包（阶段二，3 类 + normal）。

- features : 特征工程（T1 时域 / T2 物理谐波突出度(3 类×5) / T3 谱峭度 /
             T4 频谱形态 / T5 包络谱形态 / T6 盲梳检结构化）
- dataset  : 仿真增强数据集（FIELD7 恒速窗网格，防泄漏分组）与真实数据目录加载
- train    : 训练管线（监督 zoo 对比选优 / Stage1 新奇检测后端 / 模型序列化）
- evaluate : 评估报告（GroupKFold / LOSO / 阈值工作点 / v2 DSP 对比）
- infer    : 推理（flat/two_stage，转速约定：用户给定优先否则自估回退）
"""
