"""推理：加载模型包 → 提取特征 → 类别 + 概率（可选对比传统方法）。

支持两种模型架构（按 bundle["arch"] 分派）：
- flat：单级 5 分类，直接输出类别与概率；
- two_stage：Stage1 检测器给出 p_abnormal，低于阈值判正常；高于阈值进入
  Stage2 故障分类，输出条件概率与联合置信度 p_abnormal × P(fault|abnormal)。
"""

from __future__ import annotations

import numpy as np

from ..fault_freqs import get_bearing
from .features import extract_features
from .train import ABNORMAL, NORMAL, THRESHOLD_DEFAULT, load_model


def predict_features(
    bundle: dict,
    fv: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """对单个特征向量做模型预测（flat / two_stage 通用）。

    返回 {label, confidence, probabilities, model_conf}：
    - flat: confidence = 最大类概率；
    - two_stage: label=normal 时 confidence = 1−p_abnormal；label=故障时
      confidence = 联合置信度；probabilities 为归一化的联合分布；
      model_conf 附 {p_abnormal, threshold, stage2_conditional}。
    """
    arch = bundle.get("arch", "flat")
    fv = np.asarray(fv, dtype=float).reshape(1, -1)
    if fv.shape[1] != len(bundle["feature_names"]):
        raise ValueError(
            f"特征维度不匹配：模型期望 {len(bundle['feature_names'])}，"
            f"实际 {fv.shape[1]}（检查 use_physical 与预测端参数是否与训练一致）"
        )

    if arch == "flat":
        proba = bundle["model"].predict_proba(fv)[0]
        classes = list(bundle["classes"])
        order = np.argsort(proba)[::-1]
        return {
            "label": str(classes[int(order[0])]),
            "confidence": float(proba[order[0]]),
            "probabilities": {str(classes[i]): float(proba[i]) for i in order},
            "model_conf": None,
        }

    # two_stage
    thr = float(bundle.get("threshold", THRESHOLD_DEFAULT) if threshold is None
                else threshold)
    s1 = bundle["stage1"]["model"]
    classes1 = list(s1.classes_)
    p1 = s1.predict_proba(fv)[0]
    p_abn = float(p1[classes1.index(ABNORMAL)])
    if p_abn < thr:
        return {
            "label": NORMAL,
            "confidence": 1.0 - p_abn,
            "probabilities": {NORMAL: 1.0 - p_abn},
            "model_conf": {"p_abnormal": p_abn, "threshold": thr,
                           "stage2_conditional": None},
        }
    s2 = bundle["stage2"]["model"]
    classes2 = list(s2.classes_)
    p2 = s2.predict_proba(fv)[0]
    joint = {c: p_abn * float(p2[i]) for i, c in enumerate(classes2)}
    joint[NORMAL] = 1.0 - p_abn
    best = max(joint, key=joint.get)
    return {
        "label": str(best),
        "confidence": float(joint[best]),
        "probabilities": dict(sorted(joint.items(), key=lambda kv: -kv[1])),
        "model_conf": {
            "p_abnormal": p_abn, "threshold": thr,
            "stage2_conditional": {c: float(p2[i]) for i, c in enumerate(classes2)},
        },
    }


def _resolve_shaft_freq(x: np.ndarray, fs: float, shaft_freq: float | None,
                        rpm: float | None,
                        rpm_window: tuple[float, float]) -> float:
    """转速解析约定：**用户给定优先，否则转速自估，再回退窗中点**。

    优先级：--shaft-freq > --rpm/60 > 包络谱 1×fr 峰自估（失败回退
    rpm_window 中点，如 1100–1200 → 1150 rpm）。v2 DSP 不依赖转速，
    本约定只服务于 T2 物理特征组（T6 盲梳检特征不需要转速）。
    """
    if shaft_freq is not None:
        return float(shaft_freq)
    if rpm is not None:
        return float(rpm) / 60.0
    from ..diagnose import estimate_rpm
    from ..preprocessing import bandpass, detrend, remove_dc

    x_dt = detrend(remove_dc(np.asarray(x, dtype=float)))
    band = (min(500.0, 0.25 * fs), min(7500.0, 0.47 * fs))  # 与 diagnose 同口径
    x_pre = bandpass(x_dt, fs, band[0], band[1])
    info = estimate_rpm(x_pre, fs, rpm_window=rpm_window)
    return float(info["fr_hz"])


def predict_array(
    x: np.ndarray,
    fs: float,
    bundle: dict,
    shaft_freq: float | None = None,
    rpm: float | None = None,
    bearing_name: str | None = None,
    threshold: float | None = None,
) -> dict:
    """对一段信号做 ML 预测。返回 {label, confidence, probabilities, top_features}。

    use_physical 模型的转速按 _resolve_shaft_freq 的约定解析（用户给定优先、
    否则自估回退）；use_comb 模型需要 bearing_name（定位频带由几何现算），
    缺省取模型包 config["bearing"]（训练时记录，默认 FIELD7）。
    """
    use_physical = bool(bundle["use_physical"])
    use_comb = bool(bundle.get("use_comb", True))
    cfg = bundle.get("config") or {}
    bearing_name = bearing_name or cfg.get("bearing")
    rpm_window = tuple(cfg.get("rpm_window") or (1100.0, 1200.0))
    if use_comb and bearing_name is None:
        raise ValueError(
            "该模型使用 T6 盲梳检特征（定位频带需要轴承几何），预测时必须提供 "
            "--bearing（或在模型包 config 中记录）"
        )
    if use_physical:
        shaft_freq = _resolve_shaft_freq(x, fs, shaft_freq, rpm, rpm_window)
    bearing = get_bearing(bearing_name) if (use_comb or use_physical) and bearing_name else None
    fv = extract_features(x, fs=fs, shaft_freq=shaft_freq,
                          bearing=bearing, use_physical=use_physical,
                          use_comb=use_comb, rpm_window=rpm_window)
    out = predict_features(bundle, fv, threshold=threshold)
    out["top_features"] = _top_features(fv, bundle, label=out["label"])
    return out


def _top_features(fv: np.ndarray, bundle: dict, n: int = 8,
                  label: str | None = None) -> list[dict]:
    """按 |z-score| 列出该样本贡献最大的特征（可解释性辅助）。

    flat 用主模型 scaler；two_stage 判故障时用 Stage2 的 scaler（故障判别
    特征）——判 normal 时 Stage2 输出无意义，退化为原始值排序。
    """
    names = bundle["feature_names"]
    model = bundle.get("model")
    if bundle.get("arch") == "two_stage" and label != NORMAL:
        model = bundle["stage2"]["model"]
    elif bundle.get("arch") == "two_stage":
        model = None
    scaler = None
    if model is not None and hasattr(model, "named_steps") and "scaler" in model.named_steps:
        scaler = model.named_steps["scaler"]
    if scaler is not None:
        z = scaler.transform(fv.reshape(1, -1))[0]
    else:
        z = fv
    order = np.argsort(np.abs(z))[::-1][: min(n, z.size)]
    return [{"feature": names[i], "value": float(fv[i]), "z": float(z[i])}
            for i in order]


def predict_file(
    path,
    bundle: dict,
    fs: float | None = None,
    shaft_freq: float | None = None,
    rpm: float | None = None,
    bearing_name: str | None = None,
    column: int | None = None,
    compare_traditional: bool = False,
    threshold: float | None = None,
) -> dict:
    """预测数据文件。compare_traditional=True 时同时跑 v2 DSP 并附结论。

    统一包的"传统方法"是 v2 盲梳检流水线（bearing_unified.diagnose）——
    它不依赖转速，因此 --compare 不再需要提供转速/轴承（对比口径与 glm 的
    matcher 不同，结论不一致时同样提示人工复核）。
    """
    from ..loaders import load_signal

    x, fs_loaded, _ = load_signal(path, fs=fs, column=column)
    fs = fs or fs_loaded
    out = predict_array(x, fs=fs, bundle=bundle,
                        shaft_freq=shaft_freq, rpm=rpm, bearing_name=bearing_name,
                        threshold=threshold)
    out["fs"] = fs
    if compare_traditional:
        from ..diagnose import diagnose

        cfg = bundle.get("config") or {}
        b_name = bearing_name or cfg.get("bearing") or "FIELD7"
        res = diagnose(x, fs=fs, bearing=get_bearing(b_name))
        out["traditional"] = {
            "verdict": res.verdict, "verdict_cn": res.verdict_cn,
            "confidence": res.confidence,
            "rules": "v2",
        }
        out["agreement"] = out["label"] == res.verdict
    return out


def format_prediction(result: dict) -> str:
    """把预测结果渲染为中文文本（CLI stdout 用）。"""
    lines = ["—— 机器学习模型预测 ——"]
    lines.append(f"  判定类别 : {result['label']}")
    lines.append(f"  置信度   : {result['confidence']:.3f}")
    mc = result.get("model_conf")
    if mc:  # 两级架构：展示检测器门控信息
        lines.append(f"  [两级] Stage1 检测器 p_abnormal={mc['p_abnormal']:.3f} "
                     f"(阈值 {mc['threshold']:.2f})")
        if mc.get("stage2_conditional"):
            cond = "  ".join(f"{k}:{v:.3f}" for k, v in
                             sorted(mc["stage2_conditional"].items(),
                                    key=lambda kv: -kv[1]))
            lines.append(f"  [两级] Stage2 故障条件概率: {cond}")
    lines.append("  各类概率（联合分布）:")
    for k, v in result["probabilities"].items():
        lines.append(f"    {k:8s} {v:.3f}")
    if result.get("top_features"):
        lines.append("  该样本贡献最大的特征（|z| 排序）:")
        for f in result["top_features"][:5]:
            lines.append(f"    {f['feature']:22s} value={f['value']:.3f} z={f['z']:+.2f}")
    if "traditional" in result:
        t = result["traditional"]
        lines.append("—— v2 DSP（盲梳检流水线）——")
        lines.append(f"  判定    : {t['verdict']} ({t['verdict_cn']}) "
                     f"置信度 {t['confidence']:.2f}")
        lines.append(f"  两法一致: {'是' if result['agreement'] else '否——建议人工复核'}")
    return "\n".join(lines)
