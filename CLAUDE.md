# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Git repository root is this directory, containing two independent, self-contained sibling projects implementing the **same task** (rolling-bearing fault diagnosis from vibration signals, via traditional signal processing + machine learning), developed separately for side-by-side comparison. No shared code — never cross-import between them, and keep fixes/features synchronized only when asked.

| Directory | Package | CLI |
|---|---|---|
| `bearing_glm_method/` | `bearing_diag` (src layout) | `bearing-diag` |
| `bearing_kimi_method/` | `bearing_fault` (src layout) | `bearing-fault` |

All docs, reports, and user-facing output are in Chinese — keep that consistent.

## Environment & commands

One shared conda env serves both projects; version pins live in the root `requirements.txt` (single source of truth; each project's `pyproject.toml` keeps only compatible ranges). The env is defined in root `environment.yml`, which pip-installs both packages editable — both CLIs work side by side:

```bash
conda env create -f environment.yml   # from repo root; creates env bearing-detection
conda env update -f environment.yml   # after dependency changes
```

Run commands from inside the respective project directory with the `bearing-detection` env:

```bash
# Tests — glm (100 tests) / kimi (88 tests)
cd bearing_glm_method && conda run -n bearing-detection python -m pytest
cd bearing_kimi_method && conda run -n bearing-detection python -m pytest -q
# single file / keyword: append tests/test_kurtogram.py or -k kurtogram

# glm: analyze real/simulated data (needs sampling rate + rpm + bearing model)
conda run -n bearing-detection bearing-diag simulate --fault BPFO --snr 5 --seed 42 --out sim_bpfo.csv
conda run -n bearing-detection bearing-diag analyze sim_bpfo.csv --fs 12000 --rpm 1797 --bearing SKF6205 --out report/
conda run -n bearing-detection bearing-diag bearings --rpm 1797     # built-in bearings + fault frequencies
conda run -n bearing-detection bearing-diag ml-train --out models/  # ~5 min, 600 simulated samples
conda run -n bearing-detection python examples/run_demo.py          # 5-state demo

# kimi: simulate + full diagnosis; ML dataset/train/predict
conda run -n bearing-detection bearing-fault diagnose --fault inner --snr -6 --out out_diag
conda run -n bearing-detection bearing-fault ml-dataset --out data/ml_dataset.npz
conda run -n bearing-detection bearing-fault ml-train --dataset data/ml_dataset.npz --out models/
```

`ml-train`, `ml-dataset`, and demo scripts regenerate the artifacts (`models*/`, `data/*.npz`) — they are outputs, not sources, and are gitignored.

## Domain pipeline (shared by both projects)

The core flow is the same; each module below exists in both packages:

1. **Bearing geometry → characteristic fault frequencies**: BPFO (outer race), BPFI (inner race), BSF (ball spin), FTF (cage) — all proportional to shaft frequency (rpm/60).
2. **Simulator**: Randall impulse-response model (impact train + decaying resonance + slip + amplitude modulation + additive noise) with quantitative SNR — used for ground-truth test data and ML training data.
3. **Time-domain indicators**: RMS, kurtosis, crest factor, impulse factor, clearance factor, etc.
4. **Envelope demodulation (resonance demodulation)**: bandpass filter → Hilbert transform → envelope spectrum.
5. **Harmonic matching**: find families of harmonics at each candidate fault frequency in the envelope spectrum → verdict + confidence → plots/text/JSON reports.

### Project-specific differences

**glm (`bearing_diag`)** — design decisions documented in `docs/design.md` (D1–D13), theory in `docs/principles.md`:
- Kurtogram (spectral-kurtosis tree search) for adaptive band selection; cepstrum module.
- Harmonic matching uses a local-neighborhood-median prominence baseline and integer-multiple confusion disambiguation (BPFO ≈ n×FTF ambiguity); envelope-kurtosis baseline is 2.0, not 3.0.
- Low-SNR strategy: prefer missed detection over wrong fault identification.
- ML (`src/bearing_diag/ml/`): 48-dim dual-track features (physical + statistical); **two-stage hierarchical** architecture — Stage1 normal/abnormal detector with tunable threshold (`--threshold`, report gives miss/false-alarm rates at 0.3/0.5/0.7), Stage2 4-class conditional classification; joint confidence = p_abnormal × P(fault|abnormal). Evaluation: GroupKFold (anti-leakage grouping) + LOSO cross-condition + held-out test vs traditional method (`--compare`). Ablations: `--arch flat`, `--no-physical` (blind features, no rpm needed at inference).
- `SKILL.md` at project root is an AI-assistant skill entry point (documents the three inputs that must never be guessed: fs, rpm/shaft-freq, bearing geometry).

**kimi (`bearing_fault`)** — 8-chapter theory docs in `docs/` (one per module), tutorial notebooks in `notebooks/`:
- Order analysis (angle-domain resampling for variable-speed conditions) and STFT + Morlet CWT time-frequency modules — glm has neither.
- Diagnosis adds a Mahalanobis-distance health baseline.
- ML (`src/bearing_fault/ml/`): 26-dim features; two-stage = IsolationForest/EllipticEnvelope detection + SVM/RF/kNN classification.

### Naming mapping between projects

| Concept | glm | kimi |
|---|---|---|
| Fault labels | `BPFO` / `BPFI` / `BSF` / `FTF` / `normal` | `outer` / `inner` / `ball` / `cage` / `healthy` |
| CWRU 6205 bearing | `"SKF6205"` (string, loose match) | `Bearing.sk6205()` |
| Full diagnosis | `bearing-diag analyze` / `diagnose()` | `bearing-fault diagnose` / `DiagnosisPipeline.run()` |

## Testing conventions

Tests encode physics ground truths and are the arbiter for DSP changes: Gaussian kurtosis ≈ 3, sine kurtosis = 1.5, characteristic frequencies checked against CWRU-published values, SNR verified quantitatively, end-to-end fault-type identification at two SNR levels per fault, no false alarms on healthy signals, loaders round-trip, CLI smoke tests. Generated plots go through the `Agg` matplotlib backend (forced in both CLIs) — never require a display.
