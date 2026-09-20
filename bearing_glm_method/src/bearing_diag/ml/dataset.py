"""数据集构建：仿真增强（防泄漏分组）+ 真实数据目录约定。

防泄漏协议（docs/principles.md 第 11.2 节）：
    每条"基录音"（base recording）是一段独立的仿真长信号（独立种子、独立增强
    参数），切窗后同源的窗共享同一个 group id。GroupKFold/LOSO 按 group 划分，
    保证训练/测试集来自完全独立的信号实现——避免"同一噪声实现的两个窗分别落
    在训练与测试"造成的准确率虚高。

真实数据目录约定（可选混入训练）：
    root/
      manifest.json   {"fs": 12000, "rpm": 1797, "bearing": "SKF6205",
                       "window_s": 1.0}          # rpm 全局缺省，可被文件级覆盖
      normal/  a.csv b.mat ...
      BPFO/    c.csv ...
    每个文件切窗为一个或多个样本，group = 文件名。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..fault_freqs import Bearing, get_bearing
from ..simulate import simulate

LABELS = ("normal", "BPFO", "BPFI", "BSF", "FTF")
DEFAULT_RPM_GRID = (1730.0, 1750.0, 1772.0, 1797.0)  # CWRU 四个负载档转速
DEFAULT_SNR_RANGE = (-5.0, 15.0)
DEFAULT_RESONANCE_RANGE = (2000.0, 5000.0)


@dataclass
class Dataset:
    """特征已提取完成的数据集（X 与元数据等长对齐）。"""

    X: np.ndarray                        # (n_samples, n_features)
    y: np.ndarray                        # (n_samples,) 标签
    groups: np.ndarray                   # (n_samples,) 防泄漏分组 id
    feature_names_: list[str]
    rpm: np.ndarray = field(default_factory=lambda: np.array([]))
    meta: list[dict] = field(default_factory=list)
    use_physical: bool = True

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def label_counts(self) -> dict[str, int]:
        u, c = np.unique(self.y, return_counts=True)
        return {str(a): int(b) for a, b in zip(u, c)}

    def subset(self, mask: np.ndarray) -> "Dataset":
        return Dataset(
            X=self.X[mask], y=self.y[mask], groups=self.groups[mask],
            feature_names_=self.feature_names_, rpm=self.rpm[mask] if self.rpm.size else self.rpm,
            meta=[m for m, keep in zip(self.meta, mask) if keep],
            use_physical=self.use_physical,
        )


def make_sim_dataset(
    n_base_per_class: int = 15,
    windows_per_base: int = 8,
    window_s: float = 1.0,
    fs: float = 12000.0,
    bearing: Bearing | str = "SKF6205",
    rpm_grid: tuple[float, ...] = DEFAULT_RPM_GRID,
    snr_range: tuple[float, float] = DEFAULT_SNR_RANGE,
    resonance_range: tuple[float, float] = DEFAULT_RESONANCE_RANGE,
    jitter_range: tuple[float, float] = (0.005, 0.02),
    seed: int = 2026,
    use_physical: bool = True,
    verbose: bool = False,
) -> Dataset:
    """生成仿真增强数据集并提取特征。

    每条基录音独立抽取 rpm/snr/共振频率/滑移参数与随机种子；切窗共享 group。
    返回的 Dataset 可直接送入 train.evaluate_models / train_final。
    """
    from .features import extract_features, feature_names

    if isinstance(bearing, str):
        bearing = get_bearing(bearing)
    rng = np.random.default_rng(seed)
    n_win = max(2, int(round(window_s * fs)))

    X: list[np.ndarray] = []
    y: list[str] = []
    groups: list[str] = []
    rpms: list[float] = []
    metas: list[dict] = []
    gid = 0
    for label in LABELS:
        for base_i in range(n_base_per_class):
            gid += 1
            # 转速确定性轮转：保证 LOSO 每档转速都有样本（协议要求）
            rpm = float(rpm_grid[(gid - 1) % len(rpm_grid)])
            snr = float(rng.uniform(*snr_range))
            res = float(rng.uniform(*resonance_range))
            jit = float(rng.uniform(*jitter_range))
            base_seed = int(rng.integers(0, 2**31 - 1))
            duration = windows_per_base * window_s
            sig, info = simulate(
                fault=label, fs=fs, duration=duration, rpm=rpm,
                bearing=bearing, snr_db=snr, jitter=jit,
                resonance_hz=res, seed=base_seed,
            )
            starts = np.arange(0, windows_per_base) * n_win
            for wi, s in enumerate(starts):
                seg = sig[s : s + n_win]
                fv = extract_features(
                    seg, fs=fs, shaft_freq=info["shaft_freq"],
                    bearing=bearing, use_physical=use_physical,
                )
                X.append(fv)
                y.append(label)
                groups.append(f"g{gid}")
                rpms.append(rpm)
                metas.append({
                    "label": label, "group": f"g{gid}", "window": wi,
                    "rpm": rpm, "snr_db": snr, "resonance_hz": res,
                    "jitter": jit, "seed": base_seed,
                })
            if verbose:
                print(f"  [dataset] {label:6s} base#{gid:03d} "
                      f"rpm={rpm:.0f} snr={snr:+.1f}dB res={res:.0f}Hz")
    return Dataset(
        X=np.vstack(X), y=np.asarray(y), groups=np.asarray(groups),
        feature_names_=feature_names(use_physical), rpm=np.asarray(rpms),
        meta=metas, use_physical=use_physical,
    )


def load_dataset_dir(
    root: str | Path,
    bearing: Bearing | str | None = None,
    use_physical: bool = True,
    window_s: float | None = None,
) -> Dataset:
    """按目录约定加载真实数据集（子目录名 = 标签）。

    manifest.json 提供全局 fs / rpm / bearing / window_s；文件名形如
    "...rpm1797..." 时不做特殊解析（如需文件级差异，请分目录+manifest）。
    """
    import json

    from ..loaders import load_signal
    from .features import extract_features, feature_names

    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fs = float(manifest.get("fs", 0) or 0)
    if fs <= 0:
        raise ValueError("manifest.json 必须提供 fs（采样率 Hz）")
    rpm = manifest.get("rpm")
    shaft_freq = manifest.get("shaft_freq") or (float(rpm) / 60.0 if rpm else None)
    bearing = bearing or manifest.get("bearing")
    win_s = float(window_s or manifest.get("window_s", 1.0))
    n_win = max(512, int(round(win_s * fs)))

    label_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not label_dirs:
        raise ValueError(f"目录 {root} 下没有标签子目录")
    unknown = [d.name for d in label_dirs if d.name not in LABELS]
    if unknown:
        raise ValueError(f"未知标签子目录 {unknown}，合法标签: {LABELS}")

    X, y, groups, rpms, metas = [], [], [], [], []
    for d in sorted(label_dirs):
        files = sorted(p for p in d.iterdir() if p.suffix.lower()
                       in {".csv", ".txt", ".dat", ".npy", ".npz", ".mat"})
        for f in files:
            x, fs_i, _ = load_signal(f, fs=fs, column=None)
            if fs_i:
                fs = fs_i
            n_seg = x.size // n_win
            if n_seg == 0:
                continue  # 短于一个窗的文件跳过
            for wi in range(n_seg):
                seg = x[wi * n_win : (wi + 1) * n_win]
                fv = extract_features(
                    seg, fs=fs, shaft_freq=shaft_freq, bearing=bearing,
                    use_physical=use_physical,
                )
                X.append(fv)
                y.append(d.name)
                groups.append(f.name)
                rpms.append(float(rpm or 0))
                metas.append({"label": d.name, "group": f.name, "window": wi,
                              "path": str(f)})
    if not X:
        raise ValueError("未加载到任何样本（检查文件长度与格式）")
    return Dataset(
        X=np.vstack(X), y=np.asarray(y), groups=np.asarray(groups),
        feature_names_=feature_names(use_physical), rpm=np.asarray(rpms),
        meta=metas, use_physical=use_physical,
    )
