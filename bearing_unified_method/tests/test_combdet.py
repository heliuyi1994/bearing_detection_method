# -*- coding: utf-8 -*-
"""combdet 单元测试：显著峰拾取 / 盲梳扫 / 倒谱互证 / 干扰防护。"""

import numpy as np
import pytest
from scipy.signal import lfilter

from bearing_unified.combdet import (
    cepstrum_confirm,
    comb_match,
    comb_scan,
    mark_interference,
    pick_salient_peaks,
)
from bearing_unified.fault_freqs import get_bearing
from bearing_unified.kurtogram import kurtogram
from bearing_unified.preprocessing import bandpass, clamp_band, detrend, remove_dc
from bearing_unified.simulate import simulate
from bearing_unified.spectrum import envelope_spectrum, real_cepstrum

FS = 16000.0
F7 = get_bearing("FIELD7")


def _synth_env(peak_hz: list[float], peak_amp: float = 1.0, noise: float = 0.002,
               fmax: float = 800.0, df: float = 0.25, seed: int = 0):
    """合成包络谱：噪声地板 + 指定频率的强峰（局部中位数基线下显著）。"""
    rng = np.random.default_rng(seed)
    f = np.arange(0, fmax, df)
    a = noise * (1.0 + 0.2 * rng.standard_normal(f.size))
    for hz in peak_hz:
        j = int(round(hz / df))
        a[j] = peak_amp
    return f, a


def _env_of(x, fs=FS):
    """仿真信号 → Kurtogram 选带 → 零相位解调 → 包络谱 + 倒谱（与诊断同路径）。"""
    x_dt = detrend(remove_dc(x))
    kg = kurtogram(x_dt, fs, max_level=4)
    band = clamp_band(kg.best_band, fs)
    xb = bandpass(x_dt, fs, band[0], band[1])
    ef, ea = envelope_spectrum(xb, fs, max_freq=800.0)
    q, cep = real_cepstrum(xb, fs, max_quefrency=0.2)
    return ef, ea, q, cep


class TestPickSalientPeaks:
    def test_strong_peak_picked_weak_not(self):
        f, a = _synth_env([49.4], peak_amp=1.0)
        a[int(75.0 / 0.25)] = 0.004  # 75 Hz 处放一个不足显著度的小峰
        peaks = pick_salient_peaks(f, a)
        freqs = [p[0] for p in peaks]
        assert any(abs(x - 49.5) < 0.5 for x in freqs)
        assert not any(abs(x - 75.0) < 0.5 for x in freqs)

    def test_range_filter(self):
        f, a = _synth_env([10.0, 49.4, 200.0])
        peaks = pick_salient_peaks(f, a, f0_range=(20.0, 150.0))
        freqs = [p[0] for p in peaks]
        assert all(20.0 <= x <= 150.0 for x in freqs)


class TestCombScan:
    def test_finds_simulated_bpfo_comb(self):
        x, _ = simulate("BPFO", fs=FS, duration=3.0, rpm=1150, bearing=F7,
                        snr_db=10, resonance_hz=2200, seed=11)
        ef, ea, _, _ = _env_of(x)
        combs = comb_scan(ef, ea, dive_floor=18.333 - 2.0)
        assert combs, "BPFO 仿真信号应检出梳"
        ff = F7.fault_frequencies(1150 / 60)
        assert any(abs(c["f0"] - ff["BPFO"]) / ff["BPFO"] < 0.02 for c in combs), \
            f"梳基频应接近真 BPFO={ff['BPFO']:.1f}：{[c['f0'] for c in combs]}"

    def test_dives_to_minimal_fundamental(self):
        """仅 2×BSF 强、BSF 弱（基频缺失）时，应下探到 BSF 基频。"""
        f, a = _synth_env([67.4, 134.8], peak_amp=1.0)
        combs = comb_scan(f, a, dive_floor=16.33)
        assert combs and abs(combs[0]["f0"] - 33.7) < 1.0, \
            f"应下探到 33.7 Hz：{[c['f0'] for c in combs]}"

    def test_rejects_high_k_only_pseudo_comb(self):
        """高次窗扫峰伪梳（最低命中 k≥3）必须拒绝；真族（含 k≤2）不受影响。"""
        # 132.3/162.3 两个强峰：33.08 的 k=4/k=5 窗会扫到它们——伪梳场景；
        # 162.3 带上 2/3 次谐波构成真族（避免被下探规则改写）
        f, a = _synth_env([132.3, 162.3, 324.7, 487.3], peak_amp=1.0)
        combs = comb_scan(f, a, f0_range=(20.0, 200.0), dive_floor=16.33)
        assert any(abs(c["f0"] - 162.3) < 1.5 for c in combs), \
            f"真族 162.3 应检出：{[c['f0'] for c in combs]}"
        assert not any(30.0 < c["f0"] < 36.0 for c in combs), \
            f"伪梳 ~33 不应存在：{[c['f0'] for c in combs]}"
        # 缺少 k≤2 成员的 33.08 直接验证 comb_match 的命中结构
        _, _, _, hit_ks = comb_match(f, a, 33.08)
        assert min(hit_ks) >= 3 if hit_ks else True

    def test_low_f0_marks_shaft_family(self):
        """轴频族（17.1 Hz 冲击序列）下探到 <20 Hz 并打 low_f0 标记。"""
        n = int(FS * 3.0)
        sig = np.zeros(n)
        sig[:: int(round(FS / 17.1))] = 1.0
        sig = lfilter([1], [1, -1.9, 0.95], sig) \
            + 0.05 * np.random.default_rng(0).standard_normal(n)
        ef, ea, _, _ = _env_of(sig)
        combs = comb_scan(ef, ea, dive_floor=18.333 - 2.0)
        assert combs and all(c["low_f0"] for c in combs), \
            f"17.1 Hz 族应收敛为 low_f0 梳：{[(c['f0'], c['low_f0']) for c in combs]}"


class TestCepstrumConfirm:
    def test_fault_signal_confirmed(self):
        x, info = simulate("BPFI", fs=FS, duration=3.0, rpm=1150, bearing=F7,
                           snr_db=10, resonance_hz=2200, seed=5)
        ef, ea, q, cep = _env_of(x)
        ok, hits = cepstrum_confirm(q, cep, info["true_fault_freqs"]["BPFI"])
        assert ok and hits >= 1

    def test_normal_signal_not_confirmed(self):
        x, _ = simulate("normal", fs=FS, duration=3.0, rpm=1150, bearing=F7,
                        snr_db=10, resonance_hz=2200, seed=5)
        _, _, q, cep = _env_of(x)
        ok, _ = cepstrum_confirm(q, cep, 49.4)
        assert not ok


class TestMarkInterference:
    def test_mains_50hz(self):
        assert "工频" in mark_interference(50.13, 0.26, fr_nominal=19.167)

    def test_shaft_harmonic(self):
        reason = mark_interference(38.33, 0.26, fr_nominal=19.167)
        assert "轴频谐波" in reason and "2×" in reason

    def test_true_fault_passes(self):
        # BPFO≈49.4 与 50 Hz 及 k×19.167 均不吻合
        assert mark_interference(49.4, 0.26, fr_nominal=19.167) == ""

    def test_tolerance_boundary(self):
        # 1.5 谱线容差（0.39 Hz @df=0.26）：49.6 距 50 Hz 0.4 Hz，不应标记
        assert mark_interference(49.6, 0.26, fr_nominal=19.167) == ""
