"""两级模型训练：检测级 IsolationForest + 分类级交叉验证择优。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .dataset import CLASS_NAMES, HEALTHY_LABEL, Dataset
from .features import DETECTION_FEATURE_NAMES
from .model import TwoStageModel

#: 检测级候选：EllipticEnvelope（默认，稳健协方差/马氏距离思路，距离感知）
#: 或 IsolationForest（孤立森林）。实测教训：IF 的分裂界由训练数据 min/max
#: 决定、不随极端程度外推，健康样本数或网格配置稍变即可能把全部故障判为
#: 正常（检出率塌缩为 0）；EE 基于马氏距离，同批实验全配置检出率 1.0。
STAGE1_CANDIDATES = {
    "elliptic_envelope": lambda seed: EllipticEnvelope(
        contamination=0.01, random_state=seed
    ),
    "isolation_forest": lambda seed: IsolationForest(
        n_estimators=200, contamination=0.01, random_state=seed
    ),
}

#: 分类级候选模型（5 折交叉验证择优）。
#: SVC 经 CalibratedClassifierCV 包装以提供 predict_proba
#: （sklearn ≥1.9 起 SVC(probability=True) 已弃用）。
STAGE2_CANDIDATES = {
    "svm_rbf": lambda seed: CalibratedClassifierCV(
        SVC(kernel="rbf", C=1.0, random_state=seed), ensemble=False
    ),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=200, random_state=seed
    ),
    "knn": lambda seed: KNeighborsClassifier(n_neighbors=5),
}


@dataclass
class TrainResult:
    """一次训练的完整产出。"""

    model: TwoStageModel
    cv_scores: dict[str, float]  # 分类级候选模型 5 折平均准确率
    best_stage2: str
    y_test: NDArray[np.int64]
    y_pred: NDArray[np.int64]
    meta_test: NDArray
    classes: list[str] = field(default_factory=lambda: list(CLASS_NAMES))


def _make_stage2_pipelines(seed: int) -> dict[str, Pipeline]:
    return {
        name: Pipeline([("scaler", StandardScaler()), ("clf", make(seed))])
        for name, make in STAGE2_CANDIDATES.items()
    }


def train_two_stage(
    dataset: Dataset,
    test_size: float = 0.3,
    seed: int = 42,
    stage1: str = "elliptic_envelope",
) -> TrainResult:
    """训练两级模型并在留出测试集上级联预测。

    划分：按 5 类标签分层随机划分；检测级只用训练集中健康样本的检测特征
    子集（DETECTION_FEATURE_NAMES）拟合，分类级只用训练集中的故障样本
    （标签保持全局整数 1–4）。stage1 可选 "elliptic_envelope"（默认）或
    "isolation_forest"。
    """
    if stage1 not in STAGE1_CANDIDATES:
        raise ValueError(f"未知检测级 {stage1!r}，可选 {tuple(STAGE1_CANDIDATES)}")
    X, y = dataset.X, dataset.y
    det_cols = [list(dataset.feature_names).index(n) for n in DETECTION_FEATURE_NAMES]
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y
    )
    X_train, y_train = X[idx_train], y[idx_train]

    # 检测级：仅健康样本、仅检测特征子集；contamination 取低值收紧边界，
    # 健康与故障在冲击性特征上相距多个数量级，收紧不会损失检出率
    stage1_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", STAGE1_CANDIDATES[stage1](seed)),
    ])
    stage1_pipe.fit(X_train[y_train == HEALTHY_LABEL][:, det_cols])

    # 分类级：仅故障样本，候选模型交叉验证择优
    fault_mask = y_train != HEALTHY_LABEL
    X_fault, y_fault = X_train[fault_mask], y_train[fault_mask]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pipelines = _make_stage2_pipelines(seed)
    cv_scores = {
        name: float(cross_val_score(pipe, X_fault, y_fault, cv=cv).mean())
        for name, pipe in pipelines.items()
    }
    best = max(cv_scores, key=cv_scores.get)
    stage2 = pipelines[best]
    stage2.fit(X_fault, y_fault)

    model = TwoStageModel(
        stage1=stage1_pipe, stage2=stage2,
        feature_names=list(dataset.feature_names),
        classes=list(dataset.classes), fs=dataset.fs, band=dataset.band,
    )
    y_pred = model.predict_batch(dataset.X[idx_test])
    return TrainResult(
        model=model, cv_scores=cv_scores, best_stage2=best,
        y_test=dataset.y[idx_test], y_pred=y_pred, meta_test=dataset.meta[idx_test],
        classes=list(dataset.classes),
    )
