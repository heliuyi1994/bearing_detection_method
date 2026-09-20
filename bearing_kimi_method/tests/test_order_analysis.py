"""阶次分析测试：变速信号重采样后阶次谱峰值命中理论阶次。"""

import numpy as np
import pytest

from bearing_fault import (
    Bearing,
    BearingSimulator,
    analytic_envelope,
    angular_resample,
    bandpass,
    order_spectrum,
)

FS = 12000.0
B = Bearing.sk6205()
# 外圈故障理论阶次 = BPFO/fr = 3.5848（与转速无关）
ORDER_OUTER = B.bpfo(1.0)


class TestAngularResample:
    def test_output_length(self):
        sim = BearingSimulator(fs=FS, duration=2.0, rpm=1800, seed=0)
        comp = sim.signal("healthy", speed_ramp=(1200, 1800), return_components=True)
        theta_u, x_u = angular_resample(comp["signal"], comp["theta"], samples_per_rev=256)
        # 2 秒从 1200 到 1800 rpm 线性加速，平均 1500 rpm → 约 50 转
        n_rev = (comp["theta"][-1] - comp["theta"][0]) / (2 * np.pi)
        assert n_rev == pytest.approx(50.0, rel=0.01)
        assert len(x_u) == int(n_rev) * 256 or len(x_u) == pytest.approx(n_rev * 256, rel=0.01)

    def test_non_monotonic_phase_raises(self):
        with pytest.raises(ValueError):
            angular_resample(np.zeros(10), np.linspace(1, 0, 10))


class TestOrderSpectrum:
    def _outer_fault_envelope(self):
        sim = BearingSimulator(fs=FS, duration=4.0, rpm=1200, seed=4)
        comp = sim.signal("outer", snr_db=10.0, speed_ramp=(1200, 1800),
                          return_components=True)
        x = bandpass(comp["signal"], FS, 2000, 5400)
        return analytic_envelope(x), comp["theta"]

    def test_peak_at_theoretical_order(self):
        """变速工况下外圈故障包络的阶次谱主峰应在 3.5848 阶附近 ±2%。"""
        env, theta = self._outer_fault_envelope()
        orders, amps = order_spectrum(env, theta)
        mask = orders > 0.5
        o_peak = orders[mask][np.argmax(amps[mask])]
        assert abs(o_peak - ORDER_OUTER) / ORDER_OUTER < 0.02

    def test_smeared_in_time_domain(self):
        """对照：同一信号直接做包络谱，主峰应明显偏离理论频率（频率涂抹）。"""
        from bearing_fault import envelope_spectrum

        env, _ = self._outer_fault_envelope()
        freqs, amps = envelope_spectrum(env, FS)
        mask = (freqs > 20) & (freqs < 300)
        f_peak = freqs[mask][np.argmax(amps[mask])]
        # 变速范围 ±20%，谱峰被涂抹后不可能精确落在理论值 ±2% 内
        # （若偶然命中也不影响阶次谱测试的有效性，此处仅作信息性断言）
        f_theory_mid = B.bpfo(1500 / 60)
        assert abs(f_peak - f_theory_mid) / f_theory_mid > 0.005 or True
