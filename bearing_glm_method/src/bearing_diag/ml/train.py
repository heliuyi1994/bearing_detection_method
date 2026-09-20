"""训练管线：分类器对比选优 + 小网格调参 + 模型序列化。

支持两种架构（docs/principles.md 第 11.3 节）：

- flat（单级）：一个 5 分类器直接输出 normal/BPFO/BPFI/BSF/FTF；
- two_stage（两级，默认落地架构）：
    Stage 1 检测器  normal vs abnormal 二分类（报警决策，阈值可调）
    Stage 2 分类器  四类故障分类（仅故障样本训练，输出条件概率）

流程：每级独立执行"四分类器 GroupKFold 对比（防泄漏）→ macro-F1 选优 →
小网格 GridSearch → 全量重拟合"。所有分类器支持 predict_proba
（LinearSVM/RBF-SVM 经 CalibratedClassifierCV 校准）。
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

MODEL_FILENAME = "model.joblib"
NORMAL = "normal"
ABNORMAL = "abnormal"
THRESHOLD_DEFAULT = 0.5


def build_model_zoo(random_state: int = 0) -> dict[str, dict]:
    """候选分类器及其小网格。返回 {名称: {"estimator", "param_grid"}}。"""
    scaler = StandardScaler()
    return {
        "LinearSVM": {
            "estimator": Pipeline([
                ("scaler", scaler),
                ("clf", CalibratedClassifierCV(
                    LinearSVC(C=1.0, dual="auto", max_iter=20000),
                    method="sigmoid", cv=3)),
            ]),
            "param_grid": {"clf__estimator__C": [0.1, 1.0, 10.0]},
        },
        "RBF-SVM": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", CalibratedClassifierCV(
                    SVC(C=10.0, gamma="scale", random_state=random_state),
                    method="sigmoid", cv=3, ensemble=False)),
            ]),
            "param_grid": {"clf__estimator__C": [1.0, 10.0, 100.0],
                           "clf__estimator__gamma": ["scale", 0.01]},
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                n_estimators=300, random_state=random_state, n_jobs=-1),
            "param_grid": {"max_depth": [None, 12]},
        },
        "HistGB": {
            "estimator": HistGradientBoostingClassifier(
                max_iter=200, random_state=random_state),
            "param_grid": {"max_iter": [100, 200]},
        },
    }


def compare_classifiers(dataset, n_splits: int = 5) -> list[dict]:
    """四分类器 GroupKFold 对比。返回按 macro-F1 降序的结果表。"""
    from sklearn.metrics import accuracy_score, f1_score

    if dataset.n_samples < n_splits * 4:
        n_splits = max(2, dataset.n_samples // 8)
    cv = GroupKFold(n_splits=n_splits)
    rows: list[dict] = []
    for name, spec in build_model_zoo().items():
        est = spec["estimator"]
        pred = cross_val_predict(est, dataset.X, dataset.y,
                                 groups=dataset.groups, cv=cv)
        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(dataset.y, pred)),
            "macro_f1": float(f1_score(dataset.y, pred, average="macro")),
        })
    rows.sort(key=lambda r: r["macro_f1"], reverse=True)
    return rows


def tune_and_fit(dataset, model_name: str, random_state: int = 0, n_splits: int = 5):
    """对指定分类器做小网格搜索并返回最优拟合（refit 于全量数据）。"""
    spec = build_model_zoo(random_state)[model_name]
    cv = GroupKFold(n_splits=min(n_splits, max(2, len(np.unique(dataset.groups)) // 4)))
    gs = GridSearchCV(
        spec["estimator"], spec["param_grid"], scoring="f1_macro",
        cv=cv, refit=True, n_jobs=-1,
    )
    gs.fit(dataset.X, dataset.y, groups=dataset.groups)
    return gs


def train_pipeline(dataset, random_state: int = 0, verbose: bool = True):
    """完整训练：对比 → 选优 → 调参 → 重拟合。返回 (model, report)。"""
    cmp_rows = compare_classifiers(dataset)
    best_name = cmp_rows[0]["model"]
    if verbose:
        print("分类器对比（GroupKFold 防泄漏交叉验证）：")
        for r in cmp_rows:
            print(f"  {r['model']:12s} acc={r['accuracy']:.4f}  macroF1={r['macro_f1']:.4f}")
        print(f"选优: {best_name}，进入小网格调参…")
    gs = tune_and_fit(dataset, best_name, random_state)
    if verbose:
        print(f"最优超参: {gs.best_params_}  CV macroF1={gs.best_score_:.4f}")
    report = {
        "comparison": cmp_rows,
        "best_model": best_name,
        "best_params": {k: v for k, v in gs.best_params_.items()},
        "cv_macro_f1": float(gs.best_score_),
    }
    return gs.best_estimator_, report


def make_estimator(name: str, params: dict | None = None, random_state: int = 0):
    """按名称与超参重建未拟合的估计器（供评估端做交叉验证，避免用全量
    重拟合模型评自己训练的数据）。"""
    est = clone(build_model_zoo(random_state)[name]["estimator"])
    if params:
        est.set_params(**params)
    return est


def train_pipeline_two_stage(dataset, random_state: int = 0, verbose: bool = True):
    """两级层次分类训练。

    Stage 1: normal vs abnormal（标签二值化，全样本）。
    Stage 2: 四类故障（仅故障样本）。
    每级独立执行对比选优 + 小网格调参 + 全量重拟合。

    返回 (stage1_model, stage2_model, report)，report 含各级选型与最优超参，
    供评估端用 make_estimator 重建未拟合估计器做诚实交叉验证。
    """
    y1 = np.where(dataset.y == NORMAL, NORMAL, ABNORMAL)
    from dataclasses import replace

    ds1 = replace(dataset, y=y1)
    if verbose:
        print("── Stage 1 检测器（normal vs abnormal）──")
    m1, r1 = train_pipeline(ds1, random_state, verbose)
    if verbose:
        print("── Stage 2 分类器（四类故障）──")
    ds2 = dataset.subset(dataset.y != NORMAL)
    m2, r2 = train_pipeline(ds2, random_state, verbose)
    report = {"stage1": r1, "stage2": r2, "arch": "two_stage"}
    return m1, m2, report


def save_model(
    path: str | Path,
    model=None,
    *,
    feature_names_: list[str] | None = None,
    classes: list[str] | None = None,
    use_physical: bool = True,
    config: dict | None = None,
    metrics: dict | None = None,
    stage1=None,
    stage2=None,
    stage1_spec: dict | None = None,
    stage2_spec: dict | None = None,
    threshold: float = THRESHOLD_DEFAULT,
) -> Path:
    """序列化模型包。

    单级：model；两级：stage1/stage2（模型）+ stage*_spec（名称与超参，供评估
    端重建）+ threshold。两类包都带 arch 字段，推理端按 arch 分派。
    """
    if model is None and stage1 is None:
        raise ValueError("save_model 需要 model（单级）或 stage1/stage2（两级）")
    if feature_names_ is None or classes is None:
        raise ValueError("feature_names_ 与 classes 为必填")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "arch": "two_stage" if stage1 is not None else "flat",
        "model": model,
        "stage1": {"model": stage1, "spec": stage1_spec} if stage1 is not None else None,
        "stage2": {"model": stage2, "spec": stage2_spec} if stage2 is not None else None,
        "threshold": float(threshold),
        "feature_names": feature_names_,
        "classes": classes,
        "use_physical": use_physical,
        "config": config or {},
        "metrics": metrics or {},
        "format_version": 2,
    }
    joblib.dump(bundle, p)
    return p


def load_model(path: str | Path) -> dict:
    """加载模型包并做基本校验（旧版无 arch 字段的包按 flat 兼容）。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"模型文件不存在: {p}")
    bundle = joblib.load(p)
    if bundle.get("arch") == "two_stage":
        for stage_key in ("stage1", "stage2"):
            stage = bundle.get(stage_key)
            if not stage or "model" not in stage:
                raise ValueError(f"两级模型包缺少 {stage_key}.model（文件损坏？）")
    elif "model" not in bundle:
        raise ValueError("模型包缺少 model 字段（文件可能损坏或版本不兼容）")
    for key in ("feature_names", "classes", "use_physical"):
        if key not in bundle:
            raise ValueError(f"模型包缺少字段 {key}（文件可能损坏或版本不兼容）")
    return bundle
