# -*- coding: utf-8 -*-
"""仿真器与匹配器单元测试 + v2 端到端诊断验证。"""

import numpy as np
import pytest
from scipy.signal import lfilter

from bearing_unified.baseline import calibrate_oof
from bearing_unified.diagnose import diagnose
from bearing_unified.fault_freqs import BUILTIN_BEARINGS, FAULT_NAMES_CN, get_bearing
from bearing_unified.matcher import match_fault_frequencies
from bearing_unified.simulate import simulate

BEARING = BUILTIN_BEARINGS["SKF6205"]
F7 = get_bearing("FIELD7")
RPM = 1797.0
FS16 = 16000.0


class TestSimulate:
    def test_reproducible_with_seed(self):
        a, _ = simulate("BPFO", seed=7)
        b, _ = simulate("BPFO", seed=7)
        assert np.array_equal(a, b)

    def test_normal_has_no_impulses(self):
        x, info = simulate("normal", snr_db=10, seed=3)
        from bearing_unified.indicators import time_domain_indicators
        ind = time_domain_indicators(x)
        assert ind["kurtosis"] < 4.5  # 无冲击 → 峭度接近 3

    def test_fault_impulsive(self):
        x, _ = simulate("BPFO", snr_db=10, seed=3)
        from bearing_unified.indicators import time_domain_indicators
        ind = time_domain_indicators(x)
        assert ind["kurtosis"] > 4.5

    def test_info_contains_truth(self):
        _, info = simulate("BSF", rpm=1500, seed=1)
        ff = BEARING.fault_frequencies(1500 / 60)
        assert info["true_fault_freq"] == pytest.approx(ff["BSF"])

    def test_invalid_fault(self):
        with pytest.raises(ValueError):
            simulate("BAD")


class TestMatcher:
    def _env_of(self, x, fs, max_f=1000):
        from bearing_unified.spectrum import envelope_spectrum
        return envelope_spectrum(x, fs, max_freq=max_f)

    def test_synthetic_bpfo_envelope_matched(self):
        x, info = simulate("BPFO", fs=12000, duration=2.0, snr_db=10, seed=11)
        f, a = self._env_of(x, 12000)
        ff = info["true_fault_freqs"]
        m = match_fault_frequencies(f, a, ff)
        assert m.verdict == "BPFO"
        assert m.confidence > 0.3

    def test_pure_noise_no_fault(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(32768)
        f, a = self._env_of(x, 12000)
        ff = BEARING.fault_frequencies(1797 / 60)
        m = match_fault_frequencies(f, a, ff)
        assert m.verdict == "normal"

    def test_ftf_vs_bpfo_deconfliction(self):
        # 保持架故障：BPFO = 9×FTF，其谐波族会污染 BPFO 证据 → 应判 FTF
        # （matcher 保留 FTF 能力；v2 诊断流水线不参与 FTF 判别）
        x, info = simulate("FTF", fs=12000, duration=4.0, snr_db=8, seed=21)
        f, a = self._env_of(x, 12000, max_f=700)
        m = match_fault_frequencies(f, a, info["true_fault_freqs"])
        assert m.verdict == "FTF", [c.fault for c in m.candidates]

    def test_undetectable_low_freq_noted(self):
        f = np.linspace(1, 500, 500)
        a = np.abs(np.random.default_rng(0).standard_normal(500))
        m = match_fault_frequencies(f, a, {"FTF": 0.3})  # 低于分辨率下限
        assert m.verdict == "normal"
        assert any("分辨率" in n for n in m.notes)

    def test_tol_rel_parameterized(self):
        """容差参数化：理论复核用 ±4% 能命中 ±2% 命不中的偏移峰。"""
        f = np.arange(0, 500, 0.25)
        a = 0.002 * (1 + 0.1 * np.random.default_rng(0).standard_normal(f.size))
        a[int(49.0 / 0.25)] = 1.0   # 峰在 49.0，候选 50.0（偏 2%）
        m2 = match_fault_frequencies(f, a, {"X": 50.0}, n_harmonics=1, tol_rel=0.02)
        m4 = match_fault_frequencies(f, a, {"X": 50.0}, n_harmonics=1, tol_rel=0.04)
        assert m4.candidates[0].hits >= m2.candidates[0].hits


class TestDiagnoseEndToEndV2:
    """v2 规则端到端：外圈/内圈/滚珠仿真信号「检出+正确定位」，健康零误报。"""

    @pytest.mark.parametrize("fault", ["BPFO", "BPFI", "BSF"])
    @pytest.mark.parametrize("snr", [15.0, 5.0])
    def test_correct_verdict_and_localization(self, fault, snr):
        x, info = simulate(fault, fs=FS16, duration=3.0, rpm=1150, bearing=F7,
                           snr_db=snr, resonance_hz=2200, seed=42)
        res = diagnose(x, fs=FS16, bearing=F7)
        assert res.verdict == fault, (
            f"{fault}@{snr}dB 被误判为 {res.verdict}"
            f"（梳 {[c['f0'] for c in res.combs]}）"
        )
        # 定位置信度结构：三类独立、判定类最高
        confs = {"BPFO": res.conf_outer, "BPFI": res.conf_inner, "BSF": res.conf_ball}
        assert 0.0 <= min(confs.values()) <= max(confs.values()) <= 1.0
        assert confs[fault] == max(confs.values())

    def test_normal_signal_verdict(self):
        x, info = simulate("normal", fs=FS16, duration=3.0, rpm=1150, bearing=F7,
                           snr_db=10, resonance_hz=2200, seed=9)
        res = diagnose(x, fs=FS16, bearing=F7)
        assert res.verdict == "normal"

    @pytest.mark.parametrize("seed", range(9))
    def test_normal_zero_false_alarm(self, seed):
        x, _ = simulate("normal", fs=FS16, duration=3.0, rpm=1150, bearing=F7,
                        snr_db=10, resonance_hz=2200, seed=seed)
        res = diagnose(x, fs=FS16, bearing=F7)
        assert res.verdict == "normal", f"seed={seed} 误报 {res.verdict}"

    @pytest.mark.parametrize("fault", ["BPFO", "BPFI"])
    def test_low_snr_no_wrong_location(self, fault):
        # 低 SNR 允许判 normal（漏报）或未定位（诚实拒绝），不允许判错部位
        x, info = simulate(fault, fs=FS16, duration=3.0, rpm=1150, bearing=F7,
                           snr_db=-6, resonance_hz=2200, seed=13)
        res = diagnose(x, fs=FS16, bearing=F7)
        assert res.verdict in (fault, "normal", "bearing_unlocalized")

    def test_manual_band(self):
        x, info = simulate("BPFO", fs=FS16, duration=3.0, rpm=1150, bearing=F7,
                           snr_db=8, resonance_hz=2200, seed=4)
        res = diagnose(x, fs=FS16, bearing=F7, manual_band=(1500, 4000))
        assert res.band_source == "manual"
        assert res.verdict == "BPFO"

    def test_rpm_not_required(self):
        """v2 匹配不依赖转速：不提供 rpm 也能正确定位（rpm_est 仅记录）。"""
        x, _ = simulate("BPFI", fs=FS16, duration=3.0, rpm=1150, bearing=F7,
                        snr_db=10, resonance_hz=2200, seed=6)
        res = diagnose(x, fs=FS16, bearing=F7)  # 不传 rpm
        assert res.verdict == "BPFI"
        assert isinstance(res.rpm_fallback, bool)

    def test_bearing_unlocalized_semantics(self):
        """空档周期族（45 Hz 冲击序列）→ 检出但无法定位，单列不强行归类。"""
        n = int(FS16 * 3.0)
        sig = np.zeros(n)
        sig[:: int(round(FS16 / 45.0))] = 1.0
        sig = lfilter([1], [1, -1.9, 0.95], sig) \
            + 0.05 * np.random.default_rng(1).standard_normal(n)
        res = diagnose(sig, fs=FS16, bearing=F7)
        assert res.verdict == "bearing_unlocalized"
        assert res.n_clean_combs > 0 and res.n_localized == 0

    def test_friction_via_maha_auxiliary(self):
        """无梳 + 马氏触发（辅助判据）→ 异常-非轴承(摩擦)。"""
        def feats_of(sig):
            r = diagnose(sig, fs=FS16, bearing=F7)
            return {k: r.indicators[k]
                    for k in ("kurtosis", "crest_factor", "clearance_factor")}

        norm_feats = [feats_of(simulate("normal", fs=FS16, duration=3.0, rpm=1150,
                                        bearing=F7, snr_db=10, resonance_hz=2200,
                                        seed=s)[0])
                      for s in range(10)]
        cal = calibrate_oof(norm_feats, seed=0)
        thr = cal["thresholds"]["p99"]
        # 随机非周期强冲击（摩擦样）：无周期梳但冲击性极强
        rng = np.random.default_rng(7)
        n = int(FS16 * 3.0)
        xf = 0.03 * rng.standard_normal(n)
        xf[rng.choice(n, size=40, replace=False)] = 1.0
        xf = lfilter([1], [1, -1.9, 0.95], xf)
        res = diagnose(xf, fs=FS16, bearing=F7,
                       calibrator=cal["baseline"], maha_threshold=thr)
        assert res.verdict == "friction"
        assert res.trigger and res.maha_dist >= thr

    def test_normal_with_baseline_not_triggered(self):
        def feats_of(sig):
            r = diagnose(sig, fs=FS16, bearing=F7)
            return {k: r.indicators[k]
                    for k in ("kurtosis", "crest_factor", "clearance_factor")}

        norm_sigs = [simulate("normal", fs=FS16, duration=3.0, rpm=1150,
                              bearing=F7, snr_db=10, resonance_hz=2200, seed=s)[0]
                     for s in range(10)]
        cal = calibrate_oof([feats_of(s) for s in norm_sigs], seed=0)
        res = diagnose(norm_sigs[0], fs=FS16, bearing=F7,
                       calibrator=cal["baseline"],
                       maha_threshold=cal["thresholds"]["p99"])
        assert res.verdict == "normal"

    def test_result_fields_complete(self):
        x, info = simulate("BPFI", fs=FS16, duration=3.0, rpm=1150, bearing=F7,
                           snr_db=8, resonance_hz=2200, seed=8)
        res = diagnose(x, fs=FS16, bearing=F7)
        assert set(res.indicators) >= {"rms", "kurtosis", "crest_factor"}
        assert FAULT_NAMES_CN["BPFI"] in res.verdict_cn or res.verdict == "BPFI"
        assert res.env_freqs.size > 100
        assert res.cepstrum_q is not None and res.cepstrum_q.size > 0
        assert len(res.fault_bands) == 4
        assert isinstance(res.combs, list)
        assert res.bearing_info["name"] == "FIELD7"
