# -*- coding: utf-8 -*-
"""labeled_dataset_4class 双方案分析的公共层。

本目录（analysis/）是两个项目的第三方使用者：分别 import 两个包的公共 API，
不建立两项目之间的任何导入关系。

口径（决策记录）：
- 主评测转速统一 1150 rpm；敏感性轮次用 manifest 的 fr_est_hz（缺失仍用 1150）。
- 真值 = 工厂标签（中文四类：正常/外圈/内圈/滚珠），仅声称值、非拆解确认。
- 两方案统一 Kurtogram 自适应选带：glm 走其原生 use_kurtogram=True 路径；
  kimi 由本层调用 bearing_diag.kurtogram 取 best_band 注入 DiagnosisPipeline(band=...)。
- FTF/cage 判决归入"其他"行（计错，不并入滚珠）。
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
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
FIG_DIR = REPORTS_DIR / "figures"

FS = 16000.0
RPM_MAIN = 1150.0
RPM_WINDOW = (1100.0, 1200.0)

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

CLASS_CN = ("正常", "外圈", "内圈", "滚珠")
OTHER_CN = "其他"

# 中文标签 → 两方案判决符号
CN2GLM = {"正常": "normal", "外圈": "BPFO", "内圈": "BPFI", "滚珠": "BSF"}
CN2KIMI = {"正常": "healthy", "外圈": "outer", "内圈": "inner", "滚珠": "ball"}
# 两方案判决符号 → 中文四类 + 其他
GLM2CN = {"normal": "正常", "BPFO": "外圈", "BPFI": "内圈", "BSF": "滚珠", "FTF": OTHER_CN}
KIMI2CN = {"healthy": "正常", "outer": "外圈", "inner": "内圈", "ball": "滚珠", "cage": OTHER_CN}


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


def make_bearings():
    """按 geometry.json 构造两侧轴承对象（glm/kimi 各自的类型）。"""
    from bearing_diag.fault_freqs import Bearing as GBearing
    from bearing_fault.bearing import Bearing as KBearing

    g = GBearing(
        name="labeled_4class_custom",
        n_balls=GEOMETRY["n"],
        pitch_diameter=GEOMETRY["D_mm"],
        ball_diameter=GEOMETRY["d_mm"],
        contact_angle_deg=GEOMETRY["alpha_deg"],
    )
    k = KBearing(
        n=GEOMETRY["n"],
        d=GEOMETRY["d_mm"],
        pitch_d=GEOMETRY["D_mm"],
        contact_angle=GEOMETRY["alpha_deg"],
    )
    return g, k


def verify_geometry() -> dict[str, dict[str, float]]:
    """断言两侧轴承在 fr=1 Hz 的阶次与 geometry.json 一致（评测前置门槛）。"""
    g, k = make_bearings()
    tol = 2e-4
    gf = g.fault_frequencies(1.0)
    kf = k.characteristic_frequencies(1.0)
    checks = {
        "glm": {
            "BPFO": gf["BPFO"],
            "BPFI": gf["BPFI"],
            "BSF": gf["BSF"],
            "FTF": gf["FTF"],
        },
        "kimi": {
            "BPFO": kf["outer"],
            "BPFI": kf["inner"],
            "BSF": kf["ball"],
            "FTF": kf["cage"],
        },
    }
    design = {
        "BPFO": GEOMETRY["order_BPFO"],
        "BPFI": GEOMETRY["order_BPFI"],
        "BSF": GEOMETRY["order_BSF"],
        "FTF": GEOMETRY["order_FTF"],
    }
    for side, freqs in checks.items():
        for name, v in design.items():
            assert abs(freqs[name] - v) < tol, f"{side} {name}: {freqs[name]} != {v}"
    # 恒等式 BPFO + BPFI = n·fr
    assert abs(gf["BPFO"] + gf["BPFI"] - GEOMETRY["n"]) < 1e-9
    assert abs(kf["outer"] + kf["inner"] - GEOMETRY["n"]) < 1e-9
    return checks


def characteristic_freqs_hz(rpm: float) -> dict[str, float]:
    """四类特征频率（Hz），fr = rpm/60。返回统一键名 BPFO/BPFI/BSF/FTF。"""
    fr = rpm / 60.0
    return {k: v * fr for k, v in (
        ("BPFO", GEOMETRY["order_BPFO"]),
        ("BPFI", GEOMETRY["order_BPFI"]),
        ("BSF", GEOMETRY["order_BSF"]),
        ("FTF", GEOMETRY["order_FTF"]),
    )}


def run_glm(x: np.ndarray, fs: float, rpm: float) -> dict:
    """glm 原生完整诊断（use_kurtogram=True）。"""
    from bearing_diag.diagnose import diagnose

    r = diagnose(
        x,
        fs=fs,
        shaft_freq=rpm / 60.0,
        bearing=make_bearings()[0],
        use_kurtogram=True,
    )
    cand = [(c.fault, c.score) for c in r.match.candidates] if r.match else []
    kg = r.kurtogram_result
    return {
        "verdict_raw": r.verdict,
        "verdict_cn": GLM2CN.get(r.verdict, r.verdict),
        "confidence": float(r.confidence),
        "level": r.level,
        "band": tuple(r.demod_band),
        "band_source": r.band_source,
        "best_kurtosis": float(kg.best_kurtosis) if kg else float("nan"),
        "candidates": cand,
        "notes": list(r.notes),
        # 深挖图用中间量
        "_env_f": r.env_freqs,
        "_env_a": r.env_amps,
        "_indicators": dict(r.indicators),
    }


def clamp_band_kimi(band: tuple[float, float], fs: float) -> tuple[tuple[float, float], bool]:
    """kimi 的 bandpass 要求 0 < low < high < fs/2（严格），kurtogram 可能选到
    贴奈奎斯特的频带（如 (6667, 8000)@16k），此处统一夹取并报告是否发生。"""
    hi = min(band[1], fs / 2 - max(10.0, 0.001 * fs))
    lo = min(band[0], hi * 0.9)
    lo = max(lo, 1.0)
    clamped = (lo, hi) != tuple(band)
    return (lo, hi), clamped


def run_kimi(x: np.ndarray, fs: float, rpm: float, band: tuple[float, float]) -> dict:
    """kimi 诊断；band 由本层用 glm 的 kurtogram 预先算好注入（统一选带口径）。"""
    from bearing_fault.pipeline import DiagnosisPipeline

    band, clamped = clamp_band_kimi(band, fs)
    pipe = DiagnosisPipeline(bearing=make_bearings()[1], fs=fs, band=band)
    r = pipe.run(x, rpm)
    scores = {k: float(v) for k, v in r.scores.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "verdict_raw": r.verdict,
        "verdict_cn": KIMI2CN.get(r.verdict, r.verdict),
        "top_score": ranked[0][1] if ranked else 0.0,
        "second_score": ranked[1][1] if len(ranked) > 1 else 0.0,
        "scores": scores,
        "band": tuple(r.band),
        "band_clamped": clamped,
        "peak_error": r.peak_error(),
        # 深挖图用中间量
        "_env_f": r.env_freqs,
        "_env_a": r.env_amps,
        "_indicators": dict(r.features),
    }


def kurtogram_band(x: np.ndarray, fs: float) -> tuple[tuple[float, float], float]:
    """统一选带：glm 的 kurtogram（对去趋势信号），返回 (best_band, best_kurtosis)。"""
    from bearing_diag.indicators import detrend
    from bearing_diag.kurtogram import kurtogram

    kg = kurtogram(detrend(np.asarray(x, dtype=float)), fs, max_level=4)
    return kg.best_band, float(kg.best_kurtosis)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
