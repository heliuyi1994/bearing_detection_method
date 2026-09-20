"""ML 特征工程单元测试。"""

import numpy as np
import pytest

from bearing_diag.fault_freqs import BUILTIN_BEARINGS
from bearing_diag.ml.features import extract_features, feature_names
from bearing_diag.simulate import simulate

B = BUILTIN_BEARINGS["SKF6205"]


@pytest.fixture(scope="module")
def bpfo_signal():
    x, info = simulate("BPFO", fs=12000, duration=1.0, snr_db=5, seed=11)
    return x, info


class TestFeatureShapes:
    def test_names_match_vector_physical(self, bpfo_signal):
        x, info = bpfo_signal
        fv = extract_features(x, fs=info["fs"], shaft_freq=info["shaft_freq"], bearing=B)
        names = feature_names(use_physical=True)
        assert fv.shape == (len(names),)
        assert len(names) == 48  # 9 + 20 + 6 + 11 + 2

    def test_names_match_vector_blind(self, bpfo_signal):
        x, info = bpfo_signal
        fv = extract_features(x, fs=info["fs"], use_physical=False)
        names = feature_names(use_physical=False)
        assert fv.shape == (len(names),)
        assert len(names) == 28  # 9 + 6 + 11 + 2

    def test_no_nan_inf(self, bpfo_signal):
        x, info = bpfo_signal
        for phys in (True, False):
            fv = extract_features(x, fs=info["fs"], shaft_freq=info["shaft_freq"],
                                  bearing=B, use_physical=phys)
            assert np.all(np.isfinite(fv))

    def test_deterministic(self, bpfo_signal):
        x, info = bpfo_signal
        a = extract_features(x, fs=info["fs"], shaft_freq=info["shaft_freq"], bearing=B)
        b = extract_features(x, fs=info["fs"], shaft_freq=info["shaft_freq"], bearing=B)
        assert np.array_equal(a, b)


class TestScaleInvariance:
    def test_blind_features_scale_invariant(self, bpfo_signal):
        """无量纲/归一化特征（T2 突出度、T4 能量占比等）对信号增益不敏感。"""
        x, info = bpfo_signal
        a = extract_features(x, fs=info["fs"], shaft_freq=info["shaft_freq"], bearing=B)
        b = extract_features(x * 100.0, fs=info["fs"],
                             shaft_freq=info["shaft_freq"], bearing=B)
        names = feature_names(True)
        # 检查 T2/T4/T5（按构造尺度不变）与 T3 峭度族
        for i, n in enumerate(names):
            if n.startswith(("t2_", "t4_", "t5_", "t3_")):
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
        fv = extract_features(x, fs=info["fs"], shaft_freq=info["shaft_freq"], bearing=B)
        names = feature_names(True)
        bpfo_proms = [fv[i] for i, n in enumerate(names)
                      if n.startswith("t2_BPFO_")]
        bpfi_proms = [fv[i] for i, n in enumerate(names)
                      if n.startswith("t2_BPFI_")]
        assert max(bpfo_proms) > 5.0            # 命中判据级突出度
        assert max(bpfo_proms) > max(bpfi_proms)


class TestErrors:
    def test_physical_requires_params(self, bpfo_signal):
        x, info = bpfo_signal
        with pytest.raises(ValueError, match="shaft_freq"):
            extract_features(x, fs=info["fs"], use_physical=True)

    def test_short_signal_raises(self):
        with pytest.raises(ValueError, match="过短"):
            extract_features(np.zeros(100), fs=1000.0, use_physical=False)
