"""评估报告：分级指标（两级架构）、阈值工作点、LOSO、与传统方法对比。

两级架构的落地导向设计（docs/principles.md 第 11.3 节）：
- **漏报**（故障被判正常）与**误报**（正常被判异常）分开统计——现场代价不对称；
- 阈值工作点扫描：同一份 Stage1 交叉验证概率上比较多个阈值的漏报/误报，
  供现场按业务策略选择（宁漏报不误报 或 相反）；
- 端到端层次准确率：normal 判对且（若故障）类型判对才计对；
- 交叉验证用**未拟合的配置化估计器**（make_estimator 重建），避免用全量重
  拟合的模型评自己训练过的数据。
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict

from ..fault_freqs import get_bearing
from ..simulate import simulate
from .dataset import LABELS
from .features import extract_features
from .train import ABNORMAL, NORMAL, THRESHOLD_DEFAULT, make_estimator

FAULT_LABELS = [l for l in LABELS if l != NORMAL]
THRESHOLD_SWEEP = (0.3, 0.5, 0.7)


def _binary_labels(y: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(y) == NORMAL, NORMAL, ABNORMAL)


def _stage1_metrics(y_bin: np.ndarray, p_abn: np.ndarray,
                    threshold: float) -> dict:
    """漏报率 / 误报率 / 二分类准确率与 F1。"""
    pred = np.where(p_abn >= threshold, ABNORMAL, NORMAL)
    is_fault = y_bin == ABNORMAL
    miss = float(np.mean(pred[is_fault] == NORMAL)) if is_fault.any() else 0.0
    false_alarm = float(np.mean(pred[~is_fault] == ABNORMAL)) if (~is_fault).any() else 0.0
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_bin, pred)),
        "f1": float(f1_score(y_bin, pred, pos_label=ABNORMAL)),
        "miss_rate": miss,          # 漏报：故障 → 判正常
        "false_alarm_rate": false_alarm,  # 误报：正常 → 判异常
    }


def _end_to_end_accuracy(y: np.ndarray, p_abn: np.ndarray,
                         fault_pred: np.ndarray, threshold: float) -> float:
    """层次准确率：normal 判对，故障需门控通过且类型判对。"""
    is_fault = np.asarray(y) != NORMAL
    gated = p_abn >= threshold
    correct = np.zeros(len(y), dtype=bool)
    correct[~is_fault] = ~gated[~is_fault]           # 正常样本：未被门控
    correct[is_fault] = gated[is_fault] & (
        fault_pred[: int(is_fault.sum())] == y[is_fault])
    return float(np.mean(correct))


def hierarchical_cv_report(
    dataset,
    s1_spec: dict,
    s2_spec: dict,
    thresholds: tuple[float, ...] = THRESHOLD_SWEEP,
    n_splits: int = 5,
) -> dict:
    """两级架构的 GroupKFold 防泄漏交叉验证报告。

    s1_spec / s2_spec: {"model": 分类器名, "best_params": {...}}（训练阶段产物）。
    """
    y_bin = _binary_labels(dataset.y)
    is_fault = dataset.y != NORMAL
    n_splits1 = min(n_splits, max(2, len(np.unique(dataset.groups)) // 4))
    cv1 = GroupKFold(n_splits=n_splits1)
    p1 = cross_val_predict(
        make_estimator(s1_spec["model"], s1_spec["best_params"],
                       feature_names=dataset.feature_names_),
        dataset.X, y_bin, groups=dataset.groups, cv=cv1,
        method="predict_proba")
    classes1 = np.unique(y_bin)
    p_abn = p1[:, int(np.where(classes1 == ABNORMAL)[0][0])]

    # Stage2 的交叉验证预测（仅故障样本上有定义）
    ds2 = dataset.subset(is_fault)
    n_splits2 = min(n_splits, max(2, len(np.unique(ds2.groups)) // 4))
    p2_pred = cross_val_predict(
        make_estimator(s2_spec["model"], s2_spec["best_params"],
                       feature_names=dataset.feature_names_),
        ds2.X, ds2.y, groups=ds2.groups, cv=GroupKFold(n_splits=n_splits2))

    sweep = {f"{t:.2f}": _stage1_metrics(y_bin, p_abn, t) for t in thresholds}
    end_to_end = {f"{t:.2f}": _end_to_end_accuracy(dataset.y, p_abn, p2_pred, t)
                  for t in thresholds}
    return {
        "stage1": sweep,
        "stage2": classification_report_dict(ds2.y, p2_pred, FAULT_LABELS),
        "end_to_end_accuracy": end_to_end,
        "default_threshold": THRESHOLD_DEFAULT,
    }


def loso_evaluate_two_stage(
    dataset,
    s1_spec: dict,
    s2_spec: dict,
    threshold: float = THRESHOLD_DEFAULT,
    verbose: bool = True,
) -> dict:
    """留一转速验证（两级架构）：每档留出转速下分别评估两级与端到端。"""
    if dataset.rpm.size != dataset.n_samples:
        raise ValueError("数据集缺少 rpm 元数据，无法做 LOSO 评估")
    speeds = np.unique(dataset.rpm)
    if len(speeds) < 2:
        raise ValueError("rpm 只有一档，LOSO 需要至少两档转速")
    y_bin = _binary_labels(dataset.y)
    results: dict[str, dict] = {}
    for held in speeds:
        test_m = dataset.rpm == held
        train_m = ~test_m
        est1 = make_estimator(s1_spec["model"], s1_spec["best_params"],
                              feature_names=dataset.feature_names_)
        est1.fit(dataset.X[train_m], y_bin[train_m])
        classes1 = np.asarray(est1.classes_)
        p_abn = est1.predict_proba(dataset.X[test_m])[
            :, int(np.where(classes1 == ABNORMAL)[0][0])]

        tr_fault = train_m & (dataset.y != NORMAL)
        te_fault = test_m & (dataset.y != NORMAL)
        est2 = make_estimator(s2_spec["model"], s2_spec["best_params"],
                              feature_names=dataset.feature_names_)
        est2.fit(dataset.X[tr_fault], dataset.y[tr_fault])
        fault_pred = est2.predict(dataset.X[te_fault])

        y_te = dataset.y[test_m]
        rep = {
            "stage1": _stage1_metrics(y_bin[test_m], p_abn, threshold),
            "stage2": classification_report_dict(
                dataset.y[te_fault], fault_pred, FAULT_LABELS),
            "end_to_end": _end_to_end_accuracy(y_te, p_abn, fault_pred, threshold),
        }
        results[f"{held:.0f}rpm"] = rep
        if verbose:
            s1 = rep["stage1"]
            print(f"  LOSO 留出 {held:.0f} rpm: 端到端 acc={rep['end_to_end']:.4f} "
                  f"漏报率={s1['miss_rate']:.3f} 误报率={s1['false_alarm_rate']:.3f} "
                  f"Stage2 acc={rep['stage2']['accuracy']:.4f}")
    e2e = [r["end_to_end"] for r in results.values()]
    summary = {"mean_end_to_end": float(np.mean(e2e)),
               "min_end_to_end": float(np.min(e2e)),
               "max_miss_rate": max(r["stage1"]["miss_rate"] for r in results.values())}
    if verbose:
        print(f"  LOSO 汇总: 平均端到端 acc={summary['mean_end_to_end']:.4f} "
              f"最差档={summary['min_end_to_end']:.4f} "
              f"最大漏报率={summary['max_miss_rate']:.3f}")
    return {"per_speed": results, "summary": summary}


def classification_report_dict(y_true, y_pred, labels) -> dict:
    """总体指标 + 每类 precision/recall/F1 + 混淆矩阵。"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": {
            lbl: {
                "precision": float(precision_score(y_true, y_pred, labels=labels,
                                                   average=None, zero_division=0)[i]),
                "recall": float(recall_score(y_true, y_pred, labels=labels,
                                             average=None, zero_division=0)[i]),
                "f1": float(f1_score(y_true, y_pred, labels=labels,
                                     average=None, zero_division=0)[i]),
            }
            for i, lbl in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(labels),
    }


def loso_evaluate(dataset, estimator, verbose: bool = True) -> dict:
    """留一转速验证。dataset.rpm 必须非空。返回 {留出转速: 报告}。"""
    if dataset.rpm.size != dataset.n_samples:
        raise ValueError("数据集缺少 rpm 元数据，无法做 LOSO 评估")
    speeds = np.unique(dataset.rpm)
    if len(speeds) < 2:
        raise ValueError("rpm 只有一档，LOSO 需要至少两档转速")
    results: dict[str, dict] = {}
    for held in speeds:
        test_m = dataset.rpm == held
        train_m = ~test_m
        est = clone(estimator)
        est.fit(dataset.X[train_m], dataset.y[train_m])
        pred = est.predict(dataset.X[test_m])
        rep = classification_report_dict(dataset.y[test_m], pred, LABELS)
        results[f"{held:.0f}rpm"] = rep
        if verbose:
            print(f"  LOSO 留出 {held:.0f} rpm: "
                  f"acc={rep['accuracy']:.4f}  macroF1={rep['macro_f1']:.4f}")
    accs = [r["accuracy"] for r in results.values()]
    f1s = [r["macro_f1"] for r in results.values()]
    summary = {"mean_accuracy": float(np.mean(accs)),
               "min_accuracy": float(np.min(accs)),
               "mean_macro_f1": float(np.mean(f1s))}
    if verbose:
        print(f"  LOSO 汇总: 平均 acc={summary['mean_accuracy']:.4f} "
              f"最差档 acc={summary['min_accuracy']:.4f}")
    return {"per_speed": results, "summary": summary}


def compare_with_traditional(
    bundle: dict,
    fs: float = 16000.0,
    window_s: float = 2.0,
    n_per_class: int = 4,
    rpm_grid: tuple[float, ...] = (1100.0, 1133.0, 1167.0, 1200.0),
    snr_range: tuple[float, float] = (-5.0, 15.0),
    resonance_range: tuple[float, float] = (1000.0, 3000.0),
    bearing_name: str = "FIELD7",
    seed: int = 99,
    verbose: bool = True,
) -> dict:
    """在同一批独立测试信号上对比 ML 模型（flat/two_stage 均可）与 v2 DSP。

    注意：统一包的"传统方法"是 v2 盲梳检流水线（bearing_unified.diagnose），
    不是 glm 的 matcher 四候选匹配。测试协议与训练数据同分布但独立种子。
    v2 判决含 friction/bearing_unlocalized——与标签不一致时如实计错。
    """
    from ..diagnose import diagnose
    from .infer import predict_features

    use_physical = bool(bundle["use_physical"])
    use_comb = bool(bundle.get("use_comb", True))
    rng = np.random.default_rng(seed)
    bearing = get_bearing(bearing_name)
    n_win = int(round(window_s * fs))

    ml_pred, trad_pred, truth = [], [], []
    for label in LABELS:
        for _ in range(n_per_class):
            rpm = float(rng.choice(rpm_grid))
            snr = float(rng.uniform(*snr_range))
            sig, info = simulate(label, fs=fs, duration=window_s, rpm=rpm,
                                 bearing=bearing, snr_db=snr,
                                 resonance_hz=float(rng.uniform(*resonance_range)),
                                 jitter=float(rng.uniform(0.005, 0.02)),
                                 seed=int(rng.integers(0, 2**31 - 1)))
            shaft_freq = info["shaft_freq"]
            fv = extract_features(sig, fs=fs, shaft_freq=shaft_freq,
                                  bearing=bearing, use_physical=use_physical,
                                  use_comb=use_comb)
            ml_pred.append(predict_features(bundle, fv)["label"])
            res = diagnose(sig, fs=fs, bearing=bearing)  # v2 DSP（无需转速）
            trad_pred.append(res.verdict)
            truth.append(label)
    ml_acc = float(np.mean([m == t for m, t in zip(ml_pred, truth)]))
    trad_acc = float(np.mean([m == t for m, t in zip(trad_pred, truth)]))
    agreement = float(np.mean([m == t for m, t in zip(ml_pred, trad_pred)]))
    if verbose:
        print(f"  独立测试集对比（n={len(truth)}）: "
              f"ML acc={ml_acc:.3f}  传统方法 acc={trad_acc:.3f}  一致率={agreement:.3f}")
    return {"ml_accuracy": ml_acc, "traditional_accuracy": trad_acc,
            "agreement": agreement, "n": len(truth),
            "ml_predictions": ml_pred, "traditional_predictions": trad_pred,
            "truth": truth}
