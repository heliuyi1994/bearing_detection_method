"""振动数据加载：CSV/TXT、NumPy、MAT（含 CWRU 格式解析）。

约定：
- CSV/TXT：每行一个数值或多列数值。单列 → 信号；若首列单调递增且其余列
  数值相同 → 识别为时间列，取下一列为信号；多列且无法判断时用 column 参数指定。
- .npy：一维数组即信号；二维取 column。
- .mat：自动寻找 CWRU 命名的变量（含 'DE_time' 优先，其次任意 '_time'），
  取其第一列为驱动端加速度信号。CWRU 文件不含采样率，必须由 --fs 指定
  （12 kHz 数据用 12000，48 kHz 用 48000）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_EXT_CSV = {".csv", ".txt", ".dat"}


def load_signal(
    path: str | Path,
    fs: float | None = None,
    column: int | None = None,
) -> tuple[np.ndarray, float, dict]:
    """加载信号文件。

    返回 (signal, fs, meta)。fs 为 None 时尝试从数据（时间列）推断，
    推断失败则报错要求显式传入。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在: {p}")
    ext = p.suffix.lower()
    meta: dict = {"path": str(p), "format": ext}

    if ext in _EXT_CSV:
        data, meta_cols = _load_csv(p)
        meta["columns"] = meta_cols
        x, fs = _pick_column(data, fs, column, meta)
    elif ext == ".npy":
        arr = np.load(p)
        if arr.ndim == 1:
            x = arr.astype(float)
        elif arr.ndim == 2:
            x, fs = _pick_column(arr, fs, column, meta)
        else:
            raise ValueError(f"npy 数组维度异常: {arr.shape}")
    elif ext == ".npz":
        z = np.load(p)
        key = "signal" if "signal" in z.files else z.files[0]
        arr = np.asarray(z[key], dtype=float).ravel()
        meta["npz_key"] = key
        x = arr
    elif ext == ".mat":
        x = _load_mat(p, meta)
    else:
        raise ValueError(f"不支持的文件类型 {ext}（支持 csv/txt/dat/npy/npz/mat）")

    x = np.asarray(x, dtype=float).ravel()
    if x.size < 256:
        raise ValueError(f"信号过短（{x.size} 点），至少需要 256 点")
    if not np.all(np.isfinite(x)):
        raise ValueError("信号包含 NaN/Inf，请先清洗数据")
    meta["n_samples"] = int(x.size)
    meta["duration_s"] = float(x.size / fs)
    return x, float(fs), meta


def _load_csv(p: Path) -> tuple[np.ndarray, int]:
    """读取数值表格；跳过非数值头行；返回 (二维数组, 列数)。"""
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    rows: list[list[float]] = []
    skipped = 0
    for ln in lines:
        parts = ln.replace(",", " ").split()
        try:
            rows.append([float(v) for v in parts])
        except ValueError:
            skipped += 1
    if not rows:
        raise ValueError("文件中未找到数值数据")
    ncol = len(rows[0])
    if any(len(r) != ncol for r in rows):
        # 容错：按最常见列数截取
        from collections import Counter

        ncol = Counter(len(r) for r in rows).most_common(1)[0][0]
        rows = [r for r in rows if len(r) == ncol]
    return np.asarray(rows), ncol


def _pick_column(
    data: np.ndarray, fs: float | None, column: int | None, meta: dict
) -> tuple[np.ndarray, float | None]:
    ncol = data.shape[1]
    if ncol == 1:
        sig = data[:, 0]
    else:
        # 时间列自动识别：首列严格单调递增且相邻差接近常数
        first = data[:, 0]
        diffs = np.diff(first)
        is_time = (
            np.all(diffs > 0)
            and np.std(diffs) < 0.2 * np.mean(diffs)
            and np.mean(diffs) > 0
        )
        if column is not None:
            sig = data[:, column]
            meta["column"] = column
        elif is_time and ncol >= 2:
            dt = float(np.median(diffs))
            fs = 1.0 / dt if fs is None else fs
            sig = data[:, 1]
            meta["column"] = 1
            meta["time_column_detected"] = True
        else:
            sig = data[:, 0]
            meta["column"] = 0
            meta["hint"] = "多列数据默认取第 0 列，如非所需请指定 --column"
    if fs is None:
        raise ValueError(
            "无法从数据推断采样率：请用 --fs 显式指定（单位 Hz）"
        )
    return sig, fs


def _load_mat(p: Path, meta: dict) -> np.ndarray:
    """解析 MAT 文件，优先 CWRU 命名约定。"""
    from scipy.io import loadmat

    raw = loadmat(p)
    keys = [k for k in raw if not k.startswith("__")]
    meta["mat_keys"] = keys
    chosen = None
    # CWRU：X123_DE_time / X123_FE_time / X123_BA_time
    de = [k for k in keys if "DE_time" in k]
    time_keys = [k for k in keys if "time" in k.lower()]
    if de:
        chosen = de[0]
    elif time_keys:
        chosen = time_keys[0]
    elif keys:
        arr_keys = [k for k in keys if np.asarray(raw[k]).ndim >= 1]
        if not arr_keys:
            raise ValueError(f"MAT 文件中无可用信号变量: {keys}")
        chosen = arr_keys[0]
    meta["mat_var"] = chosen
    arr = np.asarray(raw[chosen], dtype=float)
    return arr[:, 0] if arr.ndim == 2 else arr.ravel()


def save_signal_csv(
    path: str | Path, x: np.ndarray, fs: float
) -> None:
    """把仿真信号存为单列 CSV（首行为注释头，加载时自动跳过）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"# fs={fs:g}\n")
        for v in np.asarray(x, dtype=float).ravel():
            f.write(f"{v:.9g}\n")
