"""bearing_diag.ml：机器学习故障识别子包。

- features : 特征工程（时域指标 / 特征频率突出度 / 谱峭度 / 频谱形态 / 包络谱形态）
- dataset  : 仿真增强数据集（防泄漏分组）与真实数据目录加载
- train    : 训练管线（多分类器 GroupKFold 对比选优、模型序列化）
- evaluate : 评估报告（混淆矩阵、留一转速 LOSO、与传统方法对比）
- infer    : 推理（模型加载 → 特征 → 类别 + 概率）

原理见 docs/principles.md 第 11 章。
"""
