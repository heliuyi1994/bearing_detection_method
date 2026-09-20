"""机器学习子系统：数据集生成、特征提取、两级训练、评估与推理。"""

from .dataset import (
    CLASS_NAMES,
    CLASS_TO_LABEL,
    HEALTHY_LABEL,
    Dataset,
    DatasetConfig,
    generate,
    load_npz,
    save_npz,
)
from .evaluate import cascade_metrics, evaluate, feature_importances, plot_confusion_matrix
from .features import DETECTION_FEATURE_NAMES, FEATURE_NAMES, FeatureExtractor
from .model import TwoStageModel
from .train import STAGE2_CANDIDATES, TrainResult, train_two_stage

__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_LABEL",
    "HEALTHY_LABEL",
    "Dataset",
    "DatasetConfig",
    "generate",
    "save_npz",
    "load_npz",
    "FEATURE_NAMES",
    "DETECTION_FEATURE_NAMES",
    "FeatureExtractor",
    "TwoStageModel",
    "train_two_stage",
    "TrainResult",
    "STAGE2_CANDIDATES",
    "cascade_metrics",
    "evaluate",
    "feature_importances",
    "plot_confusion_matrix",
]
