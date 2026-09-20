"""ML 数据集生成：按参数网格批量仿真并提取特征。

默认网格：5 类故障 × 3 转速 {1200, 1500, 1797} rpm × 3 信噪比 {-6, 0, 6} dB
× 8 个随机种子 = 360 样本。样本间唯一的随机性来自种子，网格确定即可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..simulator import FAULT_TYPES, BearingSimulator
from .features import FEATURE_NAMES, FeatureExtractor

#: 类别名 → 整数标签（固定顺序，训练/推理/评估共用）
CLASS_NAMES: tuple[str, ...] = tuple(FAULT_TYPES)  # healthy 在最前，标签 0
CLASS_TO_LABEL = {name: i for i, name in enumerate(CLASS_NAMES)}
HEALTHY_LABEL = CLASS_TO_LABEL["healthy"]


@dataclass(frozen=True)
class DatasetConfig:
    """数据集网格配置。"""

    fs: float = 12000.0
    duration: float = 3.0
    faults: tuple[str, ...] = CLASS_NAMES
    rpms: tuple[float, ...] = (1200.0, 1500.0, 1797.0)
    snrs: tuple[float, ...] = (-6.0, 0.0, 6.0)
    n_seeds: int = 8
    band: tuple[float, float] | None = None

    def grid_size(self) -> int:
        return len(self.faults) * len(self.rpms) * len(self.snrs) * self.n_seeds


@dataclass
class Dataset:
    """特征数据集：X (n_samples, n_features)、y (n_samples,)、逐样本元数据。"""

    X: NDArray[np.float64]
    y: NDArray[np.int64]
    meta: NDArray  # structured array: fault(str), rpm, snr, seed
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    classes: list[str] = field(default_factory=lambda: list(CLASS_NAMES))
    fs: float = 12000.0
    band: tuple[float, float] | None = None


def generate(config: DatasetConfig | None = None, verbose: bool = False) -> Dataset:
    """按网格生成数据集。"""
    config = config or DatasetConfig()
    extractor = FeatureExtractor(fs=config.fs, band=config.band)
    rows: list[NDArray[np.float64]] = []
    labels: list[int] = []
    meta_rows: list[tuple[str, float, float, int]] = []
    counter = 0

    for fault in config.faults:
        if fault not in CLASS_TO_LABEL:
            raise ValueError(f"未知故障类型 {fault!r}")
        for rpm in config.rpms:
            for snr in config.snrs:
                for k in range(config.n_seeds):
                    # 每个网格格点用独立种子：健康信号不含故障分量，若种子
                    # 只随 k 变化，则同一 (rpm, k) 的健康样本在不同 SNR 下是
                    # 完全相同的行——重复样本既造成训练/测试泄漏，又会使
                    # EllipticEnvelope 的稳健协方差退化
                    seed = 10_000 + counter
                    counter += 1
                    sim = BearingSimulator(
                        fs=config.fs, duration=config.duration, rpm=rpm, seed=seed,
                    )
                    x = sim.signal(fault, snr_db=snr)
                    rows.append(extractor.extract_vector(x, rpm))
                    labels.append(CLASS_TO_LABEL[fault])
                    meta_rows.append((fault, rpm, snr, seed))
                    if verbose:
                        print(f"  {fault}/{rpm:.0f}rpm/{snr:.0f}dB/seed={seed} 完成")

    meta = np.array(
        meta_rows,
        dtype=[("fault", "U8"), ("rpm", "f8"), ("snr", "f8"), ("seed", "i8")],
    )
    return Dataset(
        X=np.vstack(rows), y=np.array(labels, dtype=np.int64), meta=meta,
        fs=config.fs, band=config.band,
    )


def save_npz(path: str | Path, dataset: Dataset) -> Path:
    """保存数据集到 npz。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        X=dataset.X,
        y=dataset.y,
        meta=dataset.meta,
        feature_names=np.array(dataset.feature_names),
        classes=np.array(dataset.classes),
        fs=np.float64(dataset.fs),
        band=np.array(dataset.band if dataset.band else [0.0, 0.0]),
    )
    return path


def load_npz(path: str | Path) -> Dataset:
    """从 npz 加载数据集。"""
    data = np.load(path, allow_pickle=False)
    band = data["band"]
    return Dataset(
        X=data["X"],
        y=data["y"],
        meta=data["meta"],
        feature_names=[str(s) for s in data["feature_names"]],
        classes=[str(s) for s in data["classes"]],
        fs=float(data["fs"]),
        band=tuple(float(b) for b in band) if band[1] > 0 else None,
    )
