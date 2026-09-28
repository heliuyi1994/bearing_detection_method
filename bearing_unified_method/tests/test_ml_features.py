# -*- coding: utf-8 -*-
"""ML 特征工程单元测试（51/43/36/28 维口径 + T6 语义 + 尺度不变性）。"""

import numpy as np
import pytest

from bearing_unified.fault_freqs import get_bearing
from bearing_unified.ml.features import extract_features, feature_names
from bearing_unified.simulate import simulate

F7 = get_bearing("FIELD7")


@pytest.fixture(scope="module")
def bpfo_signal():
    x, info = simulate("BPFO", fs=16000, duration=2.0, rpm=1150, bearing=F7,
                       snr_db=8, resonance_hz=2200, seed=11)
    return x, info


class TestFeatureShapes:
    @pytest.mark.parametrize("phys,comb,expect", [
        (True, True, 51),    # 9 + 15 + 6 + 11 + 2 + 8
        (True, False, 43),   # 去 T6
        (False, True, 36),   # 去 T2（T6 不依赖转速，保留）
        (False, False, 28),  # 全盲
    ])
    def test_names_match_vector(self, bpfo_signal, phys, comb, expect):
        x, info = bpfo_signal
        fv = extract_features(x, fs=info["fs"], shaft_freq=info["shaft_freq"],
                              bearing=F7, use_physical=phys, use_comb=comb)
        names = feature_names(phys, comb)
        assert fv.shape == (len(names),) == (expect,)

    def test_no_nan_inf(self, bpfo_signal):
        x, info = bpfo_signal
        for phys in (True, False):
            for comb in (True, False):
                fv = extract_features(x, fs=info["fs"], shaft_freq=info["shaft_freq"],
                                      bearing=F7, use_physical=phys, use_comb=comb)
                assert np.all(np.isfinite(fv)), (phys, comb)

    def test_deterministic(self, bpfo_signal):
        x, info = bpfo_signal
        a = extract_features(x, fs=info["fs"], shaft_freq=info["shaft_freq"], bearing=F7)
        b = extract_features(x, fs=info["fs"], shaft_freq=info["shaft_freq"], bearing=F7)
        assert np.array_equal(a, b)

    def test_t2_is_15_dims_three_classes(self):
        """T2 物理组为三候选 × 5 谐波 = 15 维（无 FTF）。"""
        names = feature_names(True, False)
        t2 = [n for n in names if n.startswith("t2_")]
        assert len(t2) == 15
        assert not any("FTF" in n for n in t2)
        assert {n.split("_")[1] for n in t2} == {"BPFO", "BPFI", "BSF"}


class TestScaleInvariance:
    def test_dimensionless_features_scale_invariant(self, bpfo_signal):
        """无量纲/归一化特征（T2 突出度、T3 峭度族、T4 占比、T5、T6 结构化）
        对信号增益不敏感。"""
        x, info = bpfo_signal
        a = extract_features(x, fs=info["fs"], shaft_freq=info["shaft_freq"], bearing=F7)
        b = extract_features(x * 100.0, fs=info["fs"],
                             shaft_freq=info["shaft_freq"], bearing=F7)
        names = feature_names(True)
        for i, n in enumerate(names):
            if n.startswith(("t2_", "t3_", "t4_", "t5_", "t6_")):
                assert b[i] == pytest.approx(a[i], rel=1e-6, abs=1e-6), n
        # T1 有量纲项 log1p 变换：放大后必然增大，且增量不超过 log(增益)
        for i, n in enumerate(names):
            if n.endswith("_log"):
                assert b[i] > a[i], n
                assert b[i] - a[i] <= np.log(100.0) + 1e-9, n


class TestPhysicalFeatureSemantics:
    def test_bpfo_harmonics_dominate(self, bpfo_signal):
        """外圈故障信号的 T2 特征中 BPFO 谐波突出度应显著高于其他候选。"""
        x, info = bpfo_signal
        fv = extract_features(x, fs=info["fs"], shaft_freq=info["shaft_freq"], bearing=F7)
        names = feature_names(True)
        bpfo_proms = [fv[i] for i, n in enumerate(names)
                      if n.startswith("t2_BPFO_")]
        bpfi_proms = [fv[i] for i, n in enumerate(names)
                      if n.startswith("t2_BPFI_")]
        assert max(bpfo_proms) > 5.0            # 命中判据级突出度
        assert max(bpfo_proms) > max(bpfi_proms)


class TestCombFeatureSemantics:
    def test_t6_outer_fault(self, bpfo_signal):
        """外圈仿真：净梳 ≥1、最强梳落带、外圈定位置信度最高。"""
        x, info = bpfo_signal
        fv = extract_features(x, fs=info["fs"], shaft_freq=info["shaft_freq"], bearing=F7)
        names = feature_names(True)
        t6 = {n: fv[i] for i, n in enumerate(names) if n.startswith("t6_")}
        assert t6["t6_n_clean_combs"] >= 1
        assert t6["t6_best_in_band"] == 1.0
        assert t6["t6_conf_outer"] == max(
            t6["t6_conf_outer"], t6["t6_conf_inner"], t6["t6_conf_ball"])
        assert t6["t6_conf_outer"] > 0.3
        assert t6["t6_unlocalized"] == 0.0

    def test_t6_normal_near_zero(self):
        """健康信号：净梳数应为 0 或极少，三类置信度近零。"""
        x, _ = simulate("normal", fs=16000, duration=2.0, rpm=1150, bearing=F7,
                        snr_db=10, resonance_hz=2200, seed=3)
        fv = extract_features(x, fs=16000, bearing=F7, use_physical=False)
        names = feature_names(False)
        t6 = {n: fv[i] for i, n in enumerate(names) if n.startswith("t6_")}
        assert t6["t6_n_clean_combs"] == 0.0
        assert t6["t6_conf_outer"] == t6["t6_conf_inner"] == t6["t6_conf_ball"] == 0.0


class TestErrors:
    def test_physical_requires_shaft_freq(self, bpfo_signal):
        x, info = bpfo_signal
        with pytest.raises(ValueError, match="shaft_freq"):
            extract_features(x, fs=info["fs"], use_physical=True, bearing=F7)

    def test_comb_requires_bearing(self, bpfo_signal):
        x, info = bpfo_signal
        with pytest.raises(ValueError, match="bearing"):
            extract_features(x, fs=info["fs"], use_physical=False, use_comb=True)

    def test_short_signal_raises(self):
        with pytest.raises(ValueError, match="过短"):
            extract_features(np.zeros(100), fs=1000.0, use_physical=False)
