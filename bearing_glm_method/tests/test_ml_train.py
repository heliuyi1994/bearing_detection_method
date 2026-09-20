"""ML 数据集 / 训练 / 评估 / 推理 / CLI 测试。

数据集与模型用 module 级 fixture 共享，控制总运行时间。
"""

import json

import numpy as np
import pytest

from bearing_diag.fault_freqs import BUILTIN_BEARINGS
from bearing_diag.loaders import save_signal_csv
from bearing_diag.ml.dataset import (
    LABELS,
    load_dataset_dir,
    make_sim_dataset,
)
from bearing_diag.ml.evaluate import (
    hierarchical_cv_report,
    loso_evaluate,
    loso_evaluate_two_stage,
)
from bearing_diag.ml.infer import predict_array, predict_features, predict_file
from bearing_diag.ml.train import (
    compare_classifiers,
    load_model,
    save_model,
    train_pipeline,
)
from bearing_diag.simulate import simulate

B = BUILTIN_BEARINGS["SKF6205"]
N_BASE, N_WIN = 4, 4  # 小规模：5 类 × 16 样本 = 80


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
    from bearing_diag.ml.train import train_pipeline_two_stage

    m1, m2, report = train_pipeline_two_stage(dataset, verbose=False)
    return m1, m2, report


class TestDataset:
    def test_shapes_and_labels(self, dataset):
        assert dataset.n_samples == len(LABELS) * N_BASE * N_WIN
        assert set(dataset.y) == set(LABELS)
        assert dataset.X.shape[1] == dataset.n_features == 48

    def test_group_leakage_structure(self, dataset):
        """同一条基录音的所有窗必须共享 group；不同基录音不同 group。"""
        assert dataset.groups.size == dataset.n_samples
        n_groups = len(set(dataset.groups))
        assert n_groups == len(LABELS) * N_BASE
        # 每组恰好 N_WIN 个样本且标签一致
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
        assert ds.n_features == 28 and ds.use_physical is False


class TestTrain:
    def test_comparison_table(self, dataset):
        rows = compare_classifiers(dataset)
        assert len(rows) == 4
        assert all(0.0 <= r["accuracy"] <= 1.0 for r in rows)
        f1s = [r["macro_f1"] for r in rows]
        assert f1s == sorted(f1s, reverse=True)

    def test_pipeline_accuracy(self, trained):
        model, report = trained
        # 物理特征下防泄漏 CV 应达到高准确率（仿真数据）
        assert report["cv_macro_f1"] >= 0.95
        assert report["best_model"] in {"LinearSVM", "RBF-SVM",
                                        "RandomForest", "HistGB"}

    def test_save_load_roundtrip(self, trained, dataset, tmp_path):
        from bearing_diag.ml.features import extract_features

        model, report = trained
        p = save_model(tmp_path / "m.joblib", model,
                       feature_names_=dataset.feature_names_,
                       classes=list(model.classes_), use_physical=True,
                       config={"fs": 12000.0}, metrics={"cv": report})
        bundle = load_model(p)
        assert bundle["feature_names"] == dataset.feature_names_
        x, info = simulate("BPFI", fs=12000, duration=1.0, snr_db=5, seed=8)
        fv = extract_features(x, fs=12000, shaft_freq=info["shaft_freq"],
                              bearing=B)
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
        # 数据集转速轮转，四档转速必须全部出现
        assert set(out["per_speed"]) == {"1730rpm", "1750rpm",
                                         "1772rpm", "1797rpm"}
        assert out["summary"]["mean_accuracy"] >= 0.9


class TestInfer:
    def _bundle(self, trained):
        model, _ = trained
        from bearing_diag.ml.features import feature_names
        return {"arch": "flat", "model": model, "classes": list(model.classes_),
                "feature_names": feature_names(True), "use_physical": True}

    def test_predict_array_correct(self, trained):
        bundle = self._bundle(trained)
        for fault in ("BPFO", "FTF"):
            x, info = simulate(fault, fs=12000, duration=2.0,
                               snr_db=10, seed=21)
            r = predict_array(x, fs=12000, bundle=bundle,
                              shaft_freq=info["shaft_freq"], bearing_name="SKF6205")
            assert r["label"] == fault, f"{fault} → {r['label']}"
            assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-9

    def test_physical_model_requires_params(self, trained):
        bundle = self._bundle(trained)
        with pytest.raises(ValueError, match="物理特征"):
            predict_array(np.zeros(4096), fs=12000, bundle=bundle)

    def test_predict_file_with_compare(self, trained, tmp_path):
        bundle = self._bundle(trained)
        x, info = simulate("BSF", fs=12000, duration=2.0, snr_db=8, seed=13)
        p = tmp_path / "s.csv"
        save_signal_csv(p, x, 12000)
        r = predict_file(p, bundle, fs=12000, shaft_freq=info["shaft_freq"],
                         bearing_name="SKF6205", compare_traditional=True)
        assert r["label"] == "BSF"
        assert r["traditional"]["verdict"] == "BSF"
        assert r["agreement"] is True


class TestRealDataDir:
    def test_load_dataset_dir(self, tmp_path):
        root = tmp_path / "data"
        for label in ("normal", "BPFO"):
            d = root / label
            d.mkdir(parents=True)
            x, _ = simulate("BPFO" if label == "BPFO" else "normal",
                            fs=8000, duration=1.6, snr_db=10, seed=31)
            save_signal_csv(d / "a.csv", x, 8000)
        (root / "manifest.json").write_text(json.dumps(
            {"fs": 8000, "rpm": 1797, "bearing": "SKF6205", "window_s": 0.8}),
            encoding="utf-8")
        ds = load_dataset_dir(root)
        # 每文件 12800 点 / 窗 6400 点 = 2 窗 × 2 类 × 1 文件
        assert ds.n_samples == 4
        assert set(ds.y) == {"normal", "BPFO"}
        assert ds.n_features == 48

    def test_manifest_requires_fs(self, tmp_path):
        root = tmp_path / "data"
        (root / "normal").mkdir(parents=True)
        (root / "manifest.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="fs"):
            load_dataset_dir(root)


class TestTwoStage:
    """两级层次分类：检测器门控 + 故障分类 + 阈值策略。"""

    def _spec(self, report, stage):
        return {"model": report[stage]["best_model"],
                "best_params": report[stage]["best_params"]}

    def _save_bundle(self, two_stage_trained, dataset, tmp_path):
        from bearing_diag.ml.features import feature_names

        m1, m2, report = two_stage_trained
        p = save_model(
            tmp_path / "two.joblib",
            feature_names_=feature_names(True),
            classes=list(dataset.label_counts()),
            use_physical=True, config={},
            stage1=m1, stage2=m2,
            stage1_spec=self._spec(report, "stage1"),
            stage2_spec=self._spec(report, "stage2"))
        return load_model(p)

    def test_stage_structure(self, two_stage_trained):
        m1, m2, report = two_stage_trained
        assert set(m1.classes_) == {"normal", "abnormal"}
        assert set(m2.classes_) == {"BPFO", "BPFI", "BSF", "FTF"}
        assert report["arch"] == "two_stage"

    def test_hierarchical_cv_report(self, dataset, two_stage_trained):
        _, _, report = two_stage_trained
        hier = hierarchical_cv_report(
            dataset, self._spec(report, "stage1"), self._spec(report, "stage2"))
        assert set(hier["stage1"]) == {"0.30", "0.50", "0.70"}
        # 防泄漏 CV 下漏报/误报应为 0（仿真数据、物理特征）
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
        x, info = simulate("BPFO", fs=12000, duration=2.0, snr_db=8, seed=61)
        r = predict_array(x, fs=12000, bundle=bundle,
                          shaft_freq=info["shaft_freq"], bearing_name="SKF6205")
        assert r["label"] == "BPFO"
        mc = r["model_conf"]
        assert mc["p_abnormal"] >= mc["threshold"]  # 故障样本应通过门控
        assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-9  # 联合分布归一
        # 联合置信度 = p_abnormal × 条件概率
        top_cond = max(mc["stage2_conditional"].values())
        assert r["confidence"] == pytest.approx(mc["p_abnormal"] * top_cond, abs=1e-9)

    def test_threshold_forces_gate(self, two_stage_trained, dataset, tmp_path):
        """阈值策略落地验证：阈值 1.0 时任何样本都判正常。"""
        bundle = self._save_bundle(two_stage_trained, dataset, tmp_path)
        x, info = simulate("BPFO", fs=12000, duration=2.0, snr_db=8, seed=62)
        r = predict_array(x, fs=12000, bundle=bundle,
                          shaft_freq=info["shaft_freq"], bearing_name="SKF6205",
                          threshold=1.0)
        assert r["label"] == "normal"
        assert r["model_conf"]["stage2_conditional"] is None

    def test_old_flat_bundle_compat(self, trained, dataset):
        """无 arch 字段的旧模型包按 flat 兼容。"""
        from bearing_diag.ml.features import feature_names

        model, _ = trained
        legacy = {"model": model, "classes": list(model.classes_),
                  "feature_names": feature_names(True), "use_physical": True}
        x, info = simulate("BSF", fs=12000, duration=2.0, snr_db=8, seed=63)
        r = predict_array(x, fs=12000, bundle=legacy,
                          shaft_freq=info["shaft_freq"], bearing_name="SKF6205")
        assert r["label"] == "BSF" and r["model_conf"] is None

    def test_predict_features_dim_check(self, two_stage_trained, dataset, tmp_path):
        bundle = self._save_bundle(two_stage_trained, dataset, tmp_path)
        with pytest.raises(ValueError, match="维度不匹配"):
            predict_features(bundle, np.zeros(10))


class TestCLI:
    def test_ml_train_and_predict(self, tmp_path, capsys, monkeypatch):
        from bearing_diag.cli import main as cli_main

        monkeypatch.chdir(tmp_path)
        rc = cli_main(["ml-train", "--n-base", "3", "--windows", "3",
                       "--window-s", "0.6", "--seed", "9",
                       "--out", "mdl"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "分类器对比" in out and "LOSO" in out
        assert "两级架构分级评估" in out and "阈值工作点" in out
        assert (tmp_path / "mdl" / "model.joblib").exists()
        report = json.loads((tmp_path / "mdl" / "training_report.json")
                            .read_text(encoding="utf-8"))
        assert report["arch"] == "two_stage"
        assert report["stage1"]["cv_macro_f1"] >= 0.9
        assert report["stage2"]["cv_macro_f1"] >= 0.9
        assert set(report["hierarchical_cv"]["stage1"]) == {"0.30", "0.50", "0.70"}
        assert report["loso"]["summary"]["mean_end_to_end"] >= 0.9

        x, info = simulate("BPFO", fs=12000, duration=2.0, snr_db=8, seed=41)
        f = tmp_path / "q.csv"
        save_signal_csv(f, x, 12000)
        rc = cli_main(["predict", str(f), "--fs", "12000",
                       "--model", "mdl/model.joblib", "--rpm", "1797", "--compare"])
        assert rc == 0
        assert "判定类别 : BPFO" in capsys.readouterr().out

    def test_ml_train_flat_arch(self, tmp_path, capsys, monkeypatch):
        from bearing_diag.cli import main as cli_main

        monkeypatch.chdir(tmp_path)
        rc = cli_main(["ml-train", "--arch", "flat", "--n-base", "3",
                       "--windows", "3", "--window-s", "0.6", "--seed", "9",
                       "--out", "mdl_flat"])
        assert rc == 0
        report = json.loads((tmp_path / "mdl_flat" / "training_report.json")
                            .read_text(encoding="utf-8"))
        assert report["arch"] == "flat"
        assert report["cv_macro_f1"] >= 0.9
