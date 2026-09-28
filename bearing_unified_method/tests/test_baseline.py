# -*- coding: utf-8 -*-
"""baseline 单元测试：马氏距离健康基线拟合/距离行为 + 域内标定。"""

import numpy as np
import pytest

from bearing_unified.baseline import FEATURE_NAMES, HealthBaseline, calibrate_oof


def _healthy_dicts(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return [{"kurtosis": 3.0 + 0.3 * rng.standard_normal(),
             "crest_factor": 4.2 + 0.3 * rng.standard_normal(),
             "clearance_factor": 6.2 + 0.4 * rng.standard_normal()}
            for _ in range(n)]


class TestHealthBaseline:
    def test_fit_requires_two_samples(self):
        with pytest.raises(ValueError):
            HealthBaseline().fit([{"kurtosis": 3.0, "crest_factor": 4.0,
                                   "clearance_factor": 6.0}])

    def test_distance_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            HealthBaseline().distance({"kurtosis": 3.0, "crest_factor": 4.0,
                                       "clearance_factor": 6.0})

    def test_healthy_close_faulty_far(self):
        bl = HealthBaseline().fit(_healthy_dicts(30))
        d_health = bl.distance({"kurtosis": 3.1, "crest_factor": 4.3,
                                "clearance_factor": 6.3})
        d_fault = bl.distance({"kurtosis": 40.0, "crest_factor": 20.0,
                               "clearance_factor": 36.0})
        assert d_fault > 10 * max(d_health, 1e-9)

    def test_feature_names_order_matters(self):
        bl = HealthBaseline(FEATURE_NAMES).fit(_healthy_dicts(10))
        d = bl.distance({"kurtosis": 3.0, "crest_factor": 4.2,
                         "clearance_factor": 6.2, "rms": 99.0})  # 多余字段被忽略
        assert d >= 0.0


class TestCalibrateOof:
    def test_deterministic_with_seed(self):
        feats = _healthy_dicts(20)
        a = calibrate_oof(feats, seed=42)
        b = calibrate_oof(feats, seed=42)
        assert np.array_equal(a["oof"], b["oof"])
        assert a["thresholds"] == b["thresholds"]

    def test_thresholds_ordered_and_shapes(self):
        feats = _healthy_dicts(20)
        cal = calibrate_oof(feats, seed=1)
        th = cal["thresholds"]
        assert th["p95"] <= th["p99"] <= th["p99.7"]
        assert cal["oof"].shape == (20,)
        assert sum(cal["fold_sizes"]) == 20
        # 基线已拟合可直接计距
        assert cal["baseline"].distance(feats[0]) >= 0.0

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError):
            calibrate_oof(_healthy_dicts(5), n_splits=5)

    def test_fault_distances_exceed_threshold(self):
        # 正常样本标定后，强冲击故障的距离应显著越限（重尾塌陷的反向 sanity）
        feats = _healthy_dicts(20, seed=3)
        cal = calibrate_oof(feats, seed=3)
        bl, thr = cal["baseline"], cal["thresholds"]["p99"]
        d_fault = bl.distance({"kurtosis": 35.0, "crest_factor": 18.0,
                               "clearance_factor": 30.0})
        assert d_fault > thr
        # 正常折外距离绝大多数 ≤ p99（标定内禀性质）
        assert np.mean(cal["oof"] <= thr) > 0.9
