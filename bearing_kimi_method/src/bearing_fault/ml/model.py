"""两级串联模型：检测级（异常与否）→ 分类级（故障类型）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray

from .dataset import CLASS_NAMES, HEALTHY_LABEL
from .features import DETECTION_FEATURE_NAMES, FEATURE_NAMES, FeatureExtractor


@dataclass
class TwoStageModel:
    """检测 + 分类两级模型。

    - stage1：Pipeline(StandardScaler → EllipticEnvelope)（默认；可选
      IsolationForest），只用健康样本的检测特征子集训练，
      预测 +1 为正常、-1 为异常；
    - stage2：Pipeline(StandardScaler → 分类器)，只用四类故障样本训练，
      输出为全局整数标签（对应 classes 的下标，训练时已保持全局编号，
      见 train.py）。
    - detection_features：检测级特征子集名称（feature_names 的子集）。
      全维下健康样本的近零方差特征经标准化放大后会让 IsolationForest
      高维过拟合、误报率高，故检测级只用 4 维冲击性指标。
    """

    stage1: object
    stage2: object
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    classes: list[str] = field(default_factory=lambda: list(CLASS_NAMES))
    fs: float = 12000.0
    band: tuple[float, float] | None = None
    detection_features: list[str] = field(
        default_factory=lambda: list(DETECTION_FEATURE_NAMES)
    )

    def _detection_columns(self) -> list[int]:
        return [self.feature_names.index(n) for n in self.detection_features]

    def predict_batch(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """对特征矩阵级联预测，返回全局整数标签（0=healthy）。"""
        X = np.asarray(X, dtype=np.float64)
        pred = np.full(X.shape[0], HEALTHY_LABEL, dtype=np.int64)
        anomaly = self.stage1.predict(X[:, self._detection_columns()]) == -1
        if np.any(anomaly):
            pred[anomaly] = self.stage2.predict(X[anomaly])
        return pred

    def predict_one(self, x: NDArray[np.float64], rpm: float) -> dict:
        """对一段原始信号做完整推理，返回可序列化结果字典。"""
        extractor = FeatureExtractor(fs=self.fs, band=self.band)
        feats = extractor.extract(x, rpm)
        vec = np.array([[feats[n] for n in self.feature_names]])
        anomaly = bool(
            self.stage1.predict(vec[:, self._detection_columns()])[0] == -1
        )
        label = HEALTHY_LABEL
        fault_scores = None
        if anomaly:
            label = int(self.stage2.predict(vec)[0])
            if hasattr(self.stage2, "predict_proba"):
                proba = self.stage2.predict_proba(vec)[0]
                fault_scores = {
                    self.classes[c]: float(p)
                    for c, p in zip(self.stage2.classes_, proba)
                }
        return {
            "verdict": self.classes[label],
            "anomaly": anomaly,
            "fault_proba": fault_scores,
            "features": feats,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TwoStageModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"{path} 不是 TwoStageModel")
        return model
