"""数据加载器单元测试。"""

import numpy as np
import pytest
from scipy.io import savemat

from bearing_unified.loaders import load_signal, save_signal_csv


def _sig(fs=1000.0, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * 50 * t) + 0.1 * rng.standard_normal(n)


class TestLoaders:
    def test_single_column_csv(self, tmp_path):
        x = _sig()
        p = tmp_path / "s.csv"
        p.write_text("\n".join(f"{v:.8f}" for v in x), encoding="utf-8")
        y, fs, meta = load_signal(p, fs=1000.0)
        assert fs == 1000.0 and y.size == x.size
        assert np.allclose(y, x, atol=1e-7)

    def test_header_line_skipped(self, tmp_path):
        x = _sig(n=600)
        p = tmp_path / "s.csv"
        p.write_text("time,value\n" + "\n".join(f"{v:.8f}" for v in x),
                     encoding="utf-8")
        y, fs, _ = load_signal(p, fs=512.0)
        assert y.size == x.size

    def test_time_column_autodetect(self, tmp_path):
        x = _sig(fs=1000.0)
        t = np.arange(x.size) / 1000.0
        p = tmp_path / "s.csv"
        p.write_text("\n".join(f"{ti:.8f},{v:.8f}" for ti, v in zip(t, x)),
                     encoding="utf-8")
        y, fs, meta = load_signal(p)  # fs 不给，应从时间列推断
        assert fs == pytest.approx(1000.0)
        assert np.allclose(y, x, atol=1e-6)

    def test_multicolumn_explicit(self, tmp_path):
        x = _sig()
        p = tmp_path / "s.csv"
        p.write_text("\n".join(f"{i},{v:.8f},{v*2:.8f}" for i, v in enumerate(x)),
                     encoding="utf-8")
        y, fs, meta = load_signal(p, fs=1000.0, column=2)
        assert np.allclose(y, x * 2, atol=1e-6)

    def test_save_roundtrip(self, tmp_path):
        x = _sig(n=3000)
        p = tmp_path / "sim.csv"
        save_signal_csv(p, x, 2048.0)
        y, fs, _ = load_signal(p, fs=2048.0)
        assert fs == 2048.0 and np.allclose(y, x, rtol=1e-5, atol=1e-8)

    def test_npy_load(self, tmp_path):
        x = _sig()
        p = tmp_path / "s.npy"
        np.save(p, x)
        y, fs, _ = load_signal(p, fs=1000.0)
        assert np.allclose(y, x)

    def test_mat_cwru_style(self, tmp_path):
        # 模拟 CWRU 命名：X123_DE_time
        x = _sig(n=5000)
        p = tmp_path / "97.mat"
        savemat(str(p), {"X123_DE_time": x.reshape(-1, 1),
                         "X123_FE_time": (x * 2).reshape(-1, 1)})
        y, fs, meta = load_signal(p, fs=12000.0)
        assert meta["mat_var"] == "X123_DE_time"
        assert np.allclose(y, x)

    def test_mat_generic_var(self, tmp_path):
        x = _sig()
        p = tmp_path / "g.mat"
        savemat(str(p), {"vib": x.reshape(-1, 1)})
        y, _, _ = load_signal(p, fs=1000.0)
        assert np.allclose(y, x)

    def test_missing_fs_raises(self, tmp_path):
        x = _sig()
        p = tmp_path / "s.csv"
        p.write_text("\n".join(f"{v:.8f}" for v in x), encoding="utf-8")
        with pytest.raises(ValueError, match="采样率"):
            load_signal(p)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_signal("/nonexistent/x.csv", fs=1000)

    def test_bad_extension(self, tmp_path):
        p = tmp_path / "x.xyz"
        p.write_text("1")
        with pytest.raises(ValueError, match="不支持"):
            load_signal(p, fs=1000)

    def test_nan_rejected(self, tmp_path):
        x = _sig(); x[10] = np.nan
        p = tmp_path / "s.csv"
        p.write_text("\n".join(f"{v:.8f}" for v in x), encoding="utf-8")
        with pytest.raises(ValueError, match="NaN"):
            load_signal(p, fs=1000)
