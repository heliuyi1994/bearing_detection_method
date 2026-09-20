"""评估：级联 5 类指标、检测级误报/检出率、特征重要性、混淆矩阵图。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .dataset import HEALTHY_LABEL
from .train import TrainResult


def cascade_metrics(result: TrainResult) -> dict:
    """计算级联 5 类指标与检测级（健康/异常二分类）指标。"""
    y_true, y_pred = result.y_test, result.y_pred
    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(
            y_true, y_pred, target_names=result.classes, output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(result.classes)))
        ).tolist(),
    }
    # 检测级：把预测折叠为 健康/异常 二分类
    true_anom = y_true != HEALTHY_LABEL
    pred_anom = y_pred != HEALTHY_LABEL
    fp = int(np.sum(~true_anom & pred_anom))   # 健康误报为异常
    tn = int(np.sum(~true_anom & ~pred_anom))
    tp = int(np.sum(true_anom & pred_anom))    # 故障检出
    fn = int(np.sum(true_anom & ~pred_anom))
    metrics["detection"] = {
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "detection_rate": tp / (tp + fn) if (tp + fn) else 0.0,
    }
    return metrics


def plot_confusion_matrix(
    cm: NDArray[np.int64],
    classes: list[str],
    title: str = "级联诊断混淆矩阵",
) -> tuple[plt.Figure, plt.Axes]:
    """混淆矩阵热力图（行=真实，列=预测）。"""
    from ..plotting import FAULT_LABELS

    labels = [FAULT_LABELS.get(c, c) for c in classes]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), labels)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "0.2")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig, ax


def feature_importances(result: TrainResult, top: int = 10) -> list[tuple[str, float]]:
    """分类级为随机森林时返回特征重要性 Top-N，否则返回空列表。"""
    clf = result.model.stage2.named_steps.get("clf")
    if not hasattr(clf, "feature_importances_"):
        return []
    pairs = sorted(
        zip(result.model.feature_names, clf.feature_importances_),
        key=lambda kv: kv[1], reverse=True,
    )
    return pairs[:top]


def evaluate(result: TrainResult, out_dir: str | Path | None = None) -> dict:
    """汇总评估指标；给定 out_dir 时落盘 metrics.json 与混淆矩阵图。"""
    metrics = cascade_metrics(result)
    metrics["best_stage2"] = result.best_stage2
    metrics["cv_scores"] = result.cv_scores
    imp = feature_importances(result)
    if imp:
        metrics["feature_importances_top10"] = [
            {"feature": n, "importance": float(v)} for n, v in imp
        ]
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        cm = np.array(metrics["confusion_matrix"])
        fig, _ = plot_confusion_matrix(cm, result.classes)
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
    return metrics
