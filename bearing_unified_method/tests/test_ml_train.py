# -*- coding: utf-8 -*-
"""ML 数据集 / 训练 / 评估 / 推理 / CLI 测试（3 类 + FIELD7 工况）。

数据集与模型用 module 级 fixture 共享，控制总运行时间。
"""

import json

import numpy as np
import pytest

from bearing_unified.fault_freqs import get_bearing
from bearing_unified.loaders import save_signal_csv
from bearing_unified.ml.dataset import (
    LABELS,
    load_dataset_dir,
    make_sim_dataset,
)
from bearing_unified.ml.evaluate import (
    hierarchical_cv_report,
    loso_evaluate,
    loso_evaluate_two_stage,
)
from bearing_unified.ml.infer import predict_array, predict_features, predict_file
from bearing_unified.ml.train import (
    compare_classifiers,
    load_model,
    save_model,
    train_pipeline,
)
from bearing_unified.simulate import simulate

F7 = get_bearing("FIELD7")
N_BASE, N_WIN = 4, 4  # 小规模：4 类 × 16 样本 = 64


@pytest.fixture(scope="module")
def dataset():
    return make_sim_dataset(n_base_per_class=N_BASE, windows_per_base=N_WIN,
                            window_s=0.8, seed=555)


@pytest.fixture(scope="module")
def trained(dataset):
    model, report = train_pipeline(dataset, verbose=False)
    return model, report


@pytest.fixture(scope="module")
def two_stage_trained(dataset):
    from bearing_unified.ml.train import train_pipeline_two_stage

    m1, m2, report = train_pipeline_two_stage(dataset, verbose=False)
    return m1, m2, report


class TestDataset:
    def test_shapes_and_labels(self, dataset):
        assert dataset.n_samples == len(LABELS) * N_BASE * N_WIN
        assert set(dataset.y) == set(LABELS) == {"normal", "BPFO", "BPFI", "BSF"}
        assert dataset.X.shape[1] == dataset.n_features == 51

    def test_group_leakage_structure(self, dataset):
        """同一条基录音的所有窗必须共享 group；不同基录音不同 group。"""
        assert dataset.groups.size == dataset.n_samples
        n_groups = len(set(dataset.groups))
        assert n_groups == len(LABELS) * N_BASE
        for g in set(dataset.groups):
            m = dataset.groups == g
            assert m.sum() == N_WIN
            assert len(set(dataset.y[m])) == 1

    def test_reproducible(self):
        a = make_sim_dataset(n_base_per_class=2, windows_per_base=2,
                             window_s=0.5, seed=7)
        b = make_sim_dataset(n_base_per_class=2, windows_per_base=2,
                             window_s=0.5, seed=7)
        assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)

    def test_blind_mode(self):
        ds = make_sim_dataset(n_base_per_class=2, windows_per_base=2,
                              window_s=0.5, seed=3, use_physical=False)
        assert ds.n_features == 36 and ds.use_physical is False

    def test_no_comb_mode(self):
        ds = make_sim_dataset(n_base_per_class=2, windows_per_base=2,
                              window_s=0.5, seed=3, use_comb=False)
        assert ds.n_features == 43 and ds.use_comb is False

    def test_field7_rpm_grid(self, dataset):
        """FIELD7 恒速窗四点网格：LOSO 各档都有样本。"""
        assert set(np.unique(dataset.rpm)) == {1100.0, 1133.0, 1167.0, 1200.0}


class TestTrain:
    def test_comparison_table(self, dataset):
        rows = compare_classifiers(dataset)
        assert len(rows) == 4
        assert all(0.0 <= r["accuracy"] <= 1.0 for r in rows)
        f1s = [r["macro_f1"] for r in rows]
        assert f1s == sorted(f1s, reverse=True)

    def test_pipeline_accuracy(self, trained):
        model, report = trained
        assert report["cv_macro_f1"] >= 0.95
        assert report["best_model"] in {"LinearSVM", "RBF-SVM",
                                        "RandomForest", "HistGB"}

    def test_save_load_roundtrip(self, trained, dataset, tmp_path):
        from bearing_unified.ml.features import extract_features

        model, report = trained
        p = save_model(tmp_path / "m.joblib", model,
                       feature_names_=dataset.feature_names_,
                       classes=list(model.classes_), use_physical=True,
                       config={"fs": 16000.0}, metrics={"cv": report})
        bundle = load_model(p)
        assert bundle["feature_names"] == dataset.feature_names_
        assert bundle["use_comb"] is True
        x, info = simulate("BPFI", fs=16000, duration=2.0, rpm=1150, bearing=F7,
                           snr_db=5, resonance_hz=2200, seed=8)
        fv = extract_features(x, fs=16000, shaft_freq=info["shaft_freq"],
                              bearing=F7)
        p1 = bundle["model"].predict_proba(fv.reshape(1, -1))
        p2 = model.predict_proba(fv.reshape(1, -1))
        assert np.allclose(p1, p2)

    def test_load_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/model.joblib")


class TestLOSO:
    def test_loso_runs(self, dataset, trained):
        model, _ = trained
        out = loso_evaluate(dataset, model, verbose=False)
        assert set(out["per_speed"]) == {"1100rpm", "1133rpm",
                                         "1167rpm", "1200rpm"}
        assert out["summary"]["mean_accuracy"] >= 0.9


class TestInfer:
    def _bundle(self, trained):
        model, _ = trained
        from bearing_unified.ml.features import feature_names
        return {"arch": "flat", "model": model, "classes": list(model.classes_),
                "feature_names": feature_names(True), "use_physical": True,
                "use_comb": True, "config": {"bearing": "FIELD7"}}

    def test_predict_array_correct(self, trained):
        bundle = self._bundle(trained)
        for fault in ("BPFO", "BSF"):
            x, info = simulate(fault, fs=16000, duration=2.0, rpm=1150,
                               bearing=F7, snr_db=10, resonance_hz=2200, seed=21)
            r = predict_array(x, fs=16000, bundle=bundle,
                              shaft_freq=info["shaft_freq"], bearing_name="FIELD7")
            assert r["label"] == fault, f"{fault} → {r['label']}"
            assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-9

    def test_predict_rpm_convention(self, trained):
        """转速约定：用户给定优先、否则自估回退——不传 rpm/shaft_freq 也能预测。"""
        bundle = self._bundle(trained)
        x, info = simulate("BPFI", fs=16000, duration=2.0, rpm=1150,
                           bearing=F7, snr_db=10, resonance_hz=2200, seed=23)
        r = predict_array(x, fs=16000, bundle=bundle, bearing_name="FIELD7")
        assert r["label"] in ("BPFI", "normal")

    def test_comb_model_requires_bearing(self, trained):
        bundle = self._bundle(trained)
        bundle["config"] = {}
        with pytest.raises(ValueError, match="bearing"):
            predict_array(np.zeros(4096), fs=16000, bundle=bundle)

    def test_predict_file_with_compare(self, trained, tmp_path):
        bundle = self._bundle(trained)
        x, info = simulate("BSF", fs=16000, duration=2.0, rpm=1150,
                           bearing=F7, snr_db=8, resonance_hz=2200, seed=13)
        p = tmp_path / "s.csv"
        save_signal_csv(p, x, 16000)
        r = predict_file(p, bundle, fs=16000, bearing_name="FIELD7",
                         compare_traditional=True)
        assert r["label"] == "BSF"
        assert r["traditional"]["verdict"] == "BSF"  # v2 DSP 对照
        assert r["agreement"] is True


class TestRealDataDir:
    def test_load_dataset_dir(self, tmp_path):
        root = tmp_path / "data"
        for label in ("normal", "BPFO"):
            d = root / label
            d.mkdir(parents=True)
            x, _ = simulate("BPFO" if label == "BPFO" else "normal",
                            fs=16000, duration=1.6, rpm=1150, bearing=F7,
                            snr_db=10, resonance_hz=2200, seed=31)
            save_signal_csv(d / "a.csv", x, 16000)
        (root / "manifest.json").write_text(json.dumps(
            {"fs": 16000, "rpm": 1150, "bearing": "FIELD7", "window_s": 0.8}),
            encoding="utf-8")
        ds = load_dataset_dir(root)
        assert ds.n_samples == 4
        assert set(ds.y) == {"normal", "BPFO"}
        assert ds.n_features == 51

    def test_manifest_requires_fs(self, tmp_path):
        root = tmp_path / "data"
        (root / "normal").mkdir(parents=True)
        (root / "manifest.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="fs"):
            load_dataset_dir(root)

    def test_unknown_label_dir(self, tmp_path):
        root = tmp_path / "data"
        (root / "FTF").mkdir(parents=True)  # FTF 不在统一包标签集
        (root / "manifest.json").write_text(json.dumps({"fs": 16000}),
                                            encoding="utf-8")
        with pytest.raises(ValueError, match="未知标签"):
            load_dataset_dir(root)


class TestTwoStage:
    """两级层次分类：检测器门控 + 故障分类 + 阈值策略。"""

    def _spec(self, report, stage):
        return {"model": report[stage]["best_model"],
                "best_params": report[stage]["best_params"]}

    def _save_bundle(self, two_stage_trained, dataset, tmp_path):
        from bearing_unified.ml.features import feature_names

        m1, m2, report = two_stage_trained
        p = save_model(
            tmp_path / "two.joblib",
            feature_names_=feature_names(True),
            classes=list(dataset.label_counts()),
            use_physical=True, config={"bearing": "FIELD7"},
            stage1=m1, stage2=m2,
            stage1_spec=self._spec(report, "stage1"),
            stage2_spec=self._spec(report, "stage2"))
        return load_model(p)

    def test_stage_structure(self, two_stage_trained):
        m1, m2, report = two_stage_trained
        assert set(m1.classes_) == {"normal", "abnormal"}
        assert set(m2.classes_) == {"BPFO", "BPFI", "BSF"}
        assert report["arch"] == "two_stage"

    def test_hierarchical_cv_report(self, dataset, two_stage_trained):
        _, _, report = two_stage_trained
        hier = hierarchical_cv_report(
            dataset, self._spec(report, "stage1"), self._spec(report, "stage2"))
        assert set(hier["stage1"]) == {"0.30", "0.50", "0.70"}
        for m in hier["stage1"].values():
            assert m["miss_rate"] == 0.0 and m["false_alarm_rate"] == 0.0
        assert hier["stage2"]["accuracy"] >= 0.95
        assert hier["end_to_end_accuracy"]["0.50"] >= 0.95

    def test_loso_two_stage(self, dataset, two_stage_trained):
        _, _, report = two_stage_trained
        out = loso_evaluate_two_stage(
            dataset, self._spec(report, "stage1"), self._spec(report, "stage2"),
            verbose=False)
        assert len(out["per_speed"]) == 4
        assert out["summary"]["mean_end_to_end"] >= 0.9

    def test_predict_two_stage_gating(self, two_stage_trained, dataset, tmp_path):
        bundle = self._save_bundle(two_stage_trained, dataset, tmp_path)
        x, info = simulate("BPFO", fs=16000, duration=2.0, rpm=1150,
                           bearing=F7, snr_db=8, resonance_hz=2200, seed=61)
        r = predict_array(x, fs=16000, bundle=bundle,
                          shaft_freq=info["shaft_freq"], bearing_name="FIELD7")
        assert r["label"] == "BPFO"
        mc = r["model_conf"]
        assert mc["p_abnormal"] >= mc["threshold"]
        assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-9
        top_cond = max(mc["stage2_conditional"].values())
        assert r["confidence"] == pytest.approx(mc["p_abnormal"] * top_cond, abs=1e-9)

    def test_threshold_forces_gate(self, two_stage_trained, dataset, tmp_path):
        """阈值策略落地验证：阈值 1.0 时任何样本都判正常。"""
        bundle = self._save_bundle(two_stage_trained, dataset, tmp_path)
        x, info = simulate("BPFO", fs=16000, duration=2.0, rpm=1150,
                           bearing=F7, snr_db=8, resonance_hz=2200, seed=62)
        r = predict_array(x, fs=16000, bundle=bundle,
                          shaft_freq=info["shaft_freq"], bearing_name="FIELD7",
                          threshold=1.0)
        assert r["label"] == "normal"
        assert r["model_conf"]["stage2_conditional"] is None

    def test_old_flat_bundle_compat(self, trained, dataset):
        """无 arch 字段的旧模型包按 flat 兼容。"""
        from bearing_unified.ml.features import feature_names

        model, _ = trained
        legacy = {"model": model, "classes": list(model.classes_),
                  "feature_names": feature_names(True), "use_physical": True,
                  "config": {"bearing": "FIELD7"}}
        x, info = simulate("BSF", fs=16000, duration=2.0, rpm=1150,
                           bearing=F7, snr_db=8, resonance_hz=2200, seed=63)
        r = predict_array(x, fs=16000, bundle=legacy,
                          shaft_freq=info["shaft_freq"], bearing_name="FIELD7")
        assert r["label"] == "BSF" and r["model_conf"] is None

    def test_predict_features_dim_check(self, two_stage_trained, dataset, tmp_path):
        bundle = self._save_bundle(two_stage_trained, dataset, tmp_path)
        with pytest.raises(ValueError, match="维度不匹配"):
            predict_features(bundle, np.zeros(10))
