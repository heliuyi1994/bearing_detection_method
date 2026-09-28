# -*- coding: utf-8 -*-
"""localize 单元测试：频带表计算 / 频带落入定位 / 补救路径 / 理论复核 / 置信度合成。"""

import numpy as np
import pytest

from bearing_unified.fault_freqs import get_bearing
from bearing_unified.localize import (
    class_bands,
    band_of,
    localize_comb,
    localize_combs,
)

F7 = get_bearing("FIELD7")


def _synth_env(peak_hz: list[float], peak_amp: float = 1.0, noise: float = 0.002,
               fmax: float = 800.0, df: float = 0.25, seed: int = 0):
    """合成包络谱：噪声地板 + 指定频率的强峰。"""
    rng = np.random.default_rng(seed)
    f = np.arange(0, fmax, df)
    a = noise * (1.0 + 0.2 * rng.standard_normal(f.size))
    for hz in peak_hz:
        a[int(round(hz / df))] = peak_amp
    return f, a


class TestClassBands:
    def test_field7_default_window_matches_spec(self):
        """FIELD7 @1100–1200 rpm 的四个频带与合并建议稿 §6.2 表一致。"""
        bands = class_bands(F7)
        got = {b["label"]: (b["lo"], b["hi"]) for b in bands}
        assert got["滚珠(BSF基频带)"] == pytest.approx((32.2, 35.2), abs=0.1)
        assert got["外圈(BPFO带)"] == pytest.approx((47.3, 51.6), abs=0.1)
        assert got["滚珠(2×BSF主峰带)"] == pytest.approx((64.5, 70.3), abs=0.1)
        assert got["内圈(BPFI带)"] == pytest.approx((81.1, 88.4), abs=0.1)

    def test_bands_recompute_with_window(self):
        """转速窗改变时频带随之重算（函数化，不硬编码）。"""
        b1 = class_bands(F7, (1100.0, 1200.0))
        b2 = class_bands(F7, (900.0, 1000.0))
        for x, y in zip(b1, b2):
            assert y["lo"] < x["lo"] and y["hi"] < x["hi"]
        # 900–1000 rpm 的外圈带 = 2.57832×[15, 16.67]
        outer = next(b for b in b2 if "BPFO" in b["label"])
        assert (outer["lo"], outer["hi"]) == pytest.approx(
            (2.5783199 * 15.0, 2.5783199 * 1000 / 60), rel=1e-3)

    def test_bands_not_overlapping(self):
        bands = class_bands(F7)
        spans = sorted((b["lo"], b["hi"]) for b in bands)
        for (l1, h1), (l2, h2) in zip(spans, spans[1:]):
            assert h1 <= l2

    def test_band_of(self):
        bands = class_bands(F7)
        assert "BPFO" in band_of(49.4, bands)
        assert "BSF基频" in band_of(33.7, bands)
        assert "2×BSF" in band_of(67.4, bands)
        assert "BPFI" in band_of(84.7, bands)
        assert band_of(45.0, bands) == ""  # 空档


class TestLocalizeComb:
    def test_outer_band_hit(self):
        ff = F7.fault_frequencies(1150 / 60)
        f, a = _synth_env([ff["BPFO"] * k for k in range(1, 6)])
        loc = localize_comb(f, a, ff["BPFO"], F7)
        assert loc["localized"] and loc["cls"] == "outer"
        assert loc["confs"]["outer"] > 0.3
        assert "BPFO" in loc["band"]
        assert loc["fr_hyp"] == pytest.approx(1150 / 60, rel=0.01)

    def test_inner_band_hit(self):
        ff = F7.fault_frequencies(1150 / 60)
        f, a = _synth_env([ff["BPFI"] * k for k in range(1, 6)])
        loc = localize_comb(f, a, ff["BPFI"], F7)
        assert loc["localized"] and loc["cls"] == "inner"

    def test_ball_bsf_band_hit(self):
        ff = F7.fault_frequencies(1150 / 60)
        f, a = _synth_env([ff["BSF"] * k for k in range(1, 6)])
        loc = localize_comb(f, a, ff["BSF"], F7)
        assert loc["localized"] and loc["cls"] == "ball"

    def test_ball_2bsf_band_hit(self):
        """梳基频落入 2×BSF 主峰带 → 滚珠（k=2 假设）。"""
        ff = F7.fault_frequencies(1150 / 60)
        f2 = 2 * ff["BSF"]
        f, a = _synth_env([ff["BSF"], f2, 3 * ff["BSF"], 4 * ff["BSF"]])
        loc = localize_comb(f, a, f2, F7)
        assert loc["localized"] and loc["cls"] == "ball"
        assert "2×BSF" in loc["band"]

    def test_gap_unlocalized(self):
        """45 Hz（空档，fr_hyp=17.5 出窗，±4% 也够不到）→ 无法定位。"""
        f, a = _synth_env([45.0, 90.0, 135.0, 180.0, 225.0])
        loc = localize_comb(f, a, 45.0, F7)
        assert not loc["localized"] and loc["cls"] is None

    def test_rescue_k2(self):
        """f0=2×BPFO 落空档（98.8 Hz），补救路径 k=2 → 外圈。"""
        ff = F7.fault_frequencies(1150 / 60)
        f, a = _synth_env([ff["BPFO"] * k for k in range(1, 6)])
        loc = localize_comb(f, a, 2 * ff["BPFO"], F7)
        assert loc["localized"] and loc["cls"] == "outer"
        assert loc["band"] == ""  # 98.8 不在任何频带（补救路径定位）

    def test_tol_4pct_admits_slightly_out_of_window(self):
        """fr≈17.97（略低于窗）的真外圈 46.32 Hz 靠 ±4% 容差纳入窗边界。"""
        fr = 46.32 / 2.5783199  # ≈17.96，出窗 ~2%
        ff = F7.fault_frequencies(fr)
        f, a = _synth_env([ff["BPFO"] * k for k in range(1, 6)])
        loc = localize_comb(f, a, ff["BPFO"], F7)
        assert loc["localized"] and loc["cls"] == "outer"
        assert loc["fr_hyp"] == pytest.approx(1100 / 60)  # 夹取到下沿


class TestLocalizeCombs:
    def test_confidence_synthesis_max_per_class(self):
        """多梳合成：逐类取 max；一个外圈梳 + 一个未定位梳 → conf_outer 高。"""
        ff = F7.fault_frequencies(1150 / 60)
        f, a = _synth_env([ff["BPFO"] * k for k in range(1, 6)] + [45.0, 90.0])
        loc = localize_combs(f, a, [ff["BPFO"], 45.0], F7)
        assert loc["n_localized"] == 1 and not loc["unlocalized"]
        assert loc["conf_outer"] > 0.3
        assert loc["conf_inner"] <= loc["conf_outer"]

    def test_all_unlocalized_semantics(self):
        f, a = _synth_env([45.0, 90.0, 135.0])
        loc = localize_combs(f, a, [45.0], F7)
        assert loc["unlocalized"] and loc["n_localized"] == 0

    def test_empty_combs(self):
        f, a = _synth_env([], noise=0.001)
        loc = localize_combs(f, a, [], F7)
        assert loc["n_localized"] == 0 and not loc["unlocalized"]
        assert loc["conf_outer"] == loc["conf_inner"] == loc["conf_ball"] == 0.0
