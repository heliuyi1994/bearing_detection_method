# -*- coding: utf-8 -*-
"""统一包（bearing_unified）评测脚本的公共层。

提供 manifest/WAV 加载、几何断言（评测前置门槛）、中文绘图配置与 JSON 落盘
等工具；由 run_eval_unified.py、run_eval_unified_4class.py 使用。

口径（决策记录）：
- 主评测转速统一 1150 rpm；敏感性轮次用 manifest 的 fr_est_hz（缺失仍用 1150）。
- 真值 = 工厂标签（中文四类：正常/外圈/内圈/滚珠），仅声称值、非拆解确认。
- labeled_dataset_4class 数据集已从仓库移除（commit acd38bf）；历史评测报告
  保留于 reports/，完整复测需先从 git 历史恢复数据目录。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "labeled_dataset_4class"

FS = 16000.0

# geometry.json：n=7, d=3.969 mm, D=15.0 mm, α=5.6°；阶次为设计值
GEOMETRY = {
    "n": 7,
    "d_mm": 3.969,
    "D_mm": 15.0,
    "alpha_deg": 5.6,
    "order_BPFO": 2.578320,
    "order_BPFI": 4.421680,
    "order_BSF": 1.758605,
    "order_2BSF": 3.517209,
    "order_FTF": 0.368331,
}


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 300  # 高清输出（未显式传 dpi 的 savefig 均生效）


def load_manifest() -> list[dict]:
    """读 manifest.csv 并与 labels.csv 交叉核对，返回记录列表。"""
    with open(DATA_DIR / "manifest.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    with open(DATA_DIR / "labels.csv", encoding="utf-8-sig", newline="") as f:
        label_rows = list(csv.DictReader(f))
    assert {r["relpath"] for r in rows} == {r["relpath"] for r in label_rows}, (
        "manifest.csv 与 labels.csv 的 relpath 不一致"
    )
    for r in rows:
        r["fr_est_hz"] = float(r["fr_est_hz"]) if r["fr_est_hz"] else float("nan")
        r["fs_hz"] = float(r["fs_hz"])
        r["duration_s"] = float(r["duration_s"])
        for key in ("order_BPFO", "order_BPFI", "order_BSF", "order_FTF"):
            r[key] = float(r[key])
    return rows


def read_wav(relpath: str) -> tuple[np.ndarray, float]:
    """读 WAV（int16 mono 16 kHz）→ float64（满幅归一）与削波比例。"""
    from scipy.io import wavfile

    sr, raw = wavfile.read(DATA_DIR / relpath)
    assert sr == int(FS), f"{relpath}: 采样率 {sr} != {FS}"
    x = raw.astype(np.float64) / 32768.0
    clip_ratio = float(np.mean(np.abs(raw) >= 32767))
    return x, clip_ratio


def verify_geometry() -> dict[str, float]:
    """断言统一包轴承对象在 fr=1 Hz 的阶次与 geometry.json 一致（评测前置门槛）。"""
    from bearing_unified.fault_freqs import Bearing

    b = Bearing(
        name="labeled_4class_custom",
        n_balls=GEOMETRY["n"],
        pitch_diameter=GEOMETRY["D_mm"],
        ball_diameter=GEOMETRY["d_mm"],
        contact_angle_deg=GEOMETRY["alpha_deg"],
    )
    tol = 2e-4
    ff = b.fault_frequencies(1.0)
    design = {
        "BPFO": GEOMETRY["order_BPFO"],
        "BPFI": GEOMETRY["order_BPFI"],
        "BSF": GEOMETRY["order_BSF"],
        "FTF": GEOMETRY["order_FTF"],
    }
    for name, v in design.items():
        assert abs(ff[name] - v) < tol, f"{name}: {ff[name]} != {v}"
    # 恒等式 BPFO + BPFI = n·fr
    assert abs(ff["BPFO"] + ff["BPFI"] - GEOMETRY["n"]) < 1e-9
    return {k: float(v) for k, v in ff.items()}


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
