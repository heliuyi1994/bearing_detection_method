"""诊断判别测试：四类故障分类正确性、健康判据、马氏距离基线。"""

import numpy as np
import pytest

from bearing_fault import (
    BearingSimulator,
    DiagnosisPipeline,
    HealthBaseline,
    time_domain_features,
)

FS = 12000.0
RPM = 1797.0
FEATURE_NAMES = ["rms", "kurtosis", "crest_factor", "impulse_factor", "peak_to_peak"]


def _diagnose(fault: str, seed: int = 1, snr_db: float = -6.0) -> str:
    sim = BearingSimulator(fs=FS, duration=5.0, rpm=RPM, seed=seed)
    x = sim.signal(fault, snr_db=snr_db)
    return DiagnosisPipeline(fs=FS).run(x, RPM).verdict


class TestSpectralDiagnosis:
    @pytest.mark.parametrize("fault", ["inner", "outer", "ball", "cage"])
    @pytest.mark.parametrize("seed", [1, 7, 42])
    def test_fault_classified_correctly(self, fault, seed):
        assert _diagnose(fault, seed=seed) == fault

    @pytest.mark.parametrize("seed", [1, 7, 42])
    def test_healthy_not_misdiagnosed(self, seed):
        assert _diagnose("healthy", seed=seed) == "healthy"


class TestPeakError:
    @pytest.mark.parametrize("fault", ["inner", "outer", "ball", "cage"])
    def test_peak_error_below_2_percent(self, fault):
        sim = BearingSimulator(fs=FS, duration=5.0, rpm=RPM, seed=1)
        x = sim.signal(fault, snr_db=-6.0)
        report = DiagnosisPipeline(fs=FS).run(x, RPM)
        err = report.peak_error(fault)
        assert err is not None
        assert err < 0.02


class TestHealthBaseline:
    def _features(self, fault: str, seed: int) -> dict[str, float]:
        sim = BearingSimulator(fs=FS, duration=2.0, rpm=RPM, seed=seed)
        return time_domain_features(sim.signal(fault, snr_db=-6.0))

    def test_faulty_distance_exceeds_healthy(self):
        baseline = HealthBaseline(FEATURE_NAMES).fit(
            [self._features("healthy", s) for s in range(10)]
        )
        d_healthy = [
            baseline.distance(self._features("healthy", s)) for s in (100, 101)
        ]
        d_faulty = [
            baseline.distance(self._features(f, 100))
            for f in ("inner", "outer", "ball", "cage")
        ]
        assert min(d_faulty) > max(d_healthy)

    def test_fit_requires_two_samples(self):
        with pytest.raises(ValueError):
            HealthBaseline(FEATURE_NAMES).fit([self._features("healthy", 0)])

    def test_distance_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            HealthBaseline(FEATURE_NAMES).distance(self._features("healthy", 0))
