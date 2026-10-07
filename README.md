# chb-mit-seizure-bandpower-benchmark

Frequency-domain feature engineering and a multi-model benchmark for EEG seizure detection on the CHB-MIT dataset.

## Overview

This project builds a seizure/non-seizure classifier from single-second EEG windows (23 channels × 256 samples @ 256 Hz) drawn from the CHB-MIT scalp EEG dataset. It compares a naive raw-signal baseline against hand-engineered frequency-domain features, then benchmarks eight classical ML models on top of those features.

**Pipeline:**
1. **EDA** — raw signal overlays, FFT power spectra, per-band power heatmaps, channel discriminability, Hilbert envelopes, cross-channel correlation matrices, and global per-channel statistics, comparing seizure vs. non-seizure windows.
2. **Baseline** — XGBoost on the raw flattened signal (23×256 = 5888 dims, zero feature engineering), used as a floor to beat.
3. **Frequency features** — Welch PSD per channel, band power in 5 standard EEG bands (delta/theta/alpha/beta/gamma) per channel, both absolute and relative to total power → 230-dim feature vector (115 + 115).
4. **Model benchmark** — XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, Gradient Boosting, Logistic Regression, SVM (RBF), and an MLP, all trained on the 230-dim feature set.

## Results

| Model | AUC | Accuracy | F1 | Recall | Precision |
|---|---|---|---|---|---|
| **XGBoost** | **0.9897** | 0.9461 | 0.9465 | 0.9521 | 0.9409 |
| LightGBM | 0.9882 | 0.9453 | 0.9454 | 0.9475 | 0.9433 |
| CatBoost | 0.9868 | 0.9433 | 0.9434 | 0.9453 | 0.9416 |
| Gradient Boosting | 0.9847 | 0.9365 | 0.9369 | 0.9419 | 0.9319 |
| Random Forest | 0.9817 | 0.9239 | 0.9239 | 0.9239 | 0.9239 |
| MLP | 0.9488 | 0.8973 | 0.8965 | 0.8895 | 0.9037 |
| Extra Trees | 0.9491 | 0.8765 | 0.8744 | 0.8596 | 0.8897 |
| SVM (RBF) | 0.8929 | 0.7730 | 0.7362 | 0.6334 | 0.8787 |
| Logistic Regression | 0.8721 | 0.7953 | 0.7851 | 0.7479 | 0.8262 |

Raw-signal XGBoost baseline (no feature engineering): 0.9376 AUC. Engineered frequency features improve AUC by ~0.05 over the raw-signal floor, confirming the band-power features carry most of the useful signal.

Top features by importance were almost entirely theta/alpha/delta power in a small number of channels (ch02, ch09), consistent with known seizure-related spectral shifts.

## Important caveat on evaluation

The CHB-MIT subset used here ships three files: `train`, `val`, and `val_balanced`. There is **no independent held-out test set**. `val_balanced` is reused across every model in the benchmark above for both comparison and model selection, so the reported numbers reflect best-of-N-models performance on a set that was effectively used for selection, not a single untouched evaluation. There is also no check for patient-level (subject-ID) separation between train and validation, which is the standard source of inflated performance on this dataset — a model can learn patient-specific signatures rather than general seizure physiology. Treat the AUC/accuracy figures above as optimistic upper bounds, not a reliable estimate of out-of-subject generalization.

## Limitations / what this isn't

- No temporal/sequence modeling — band power collapses each 1-second window to a flat feature vector, discarding within-window dynamics.
- No raw-signal deep learning (CNN/RNN/transformer over the waveform) was attempted.
- No subject-wise cross-validation.
- No deployment, streaming, or latency considerations — this is an offline benchmark notebook.

## Possible next steps

- Re-split train/val by patient ID to get an honest generalization estimate.
- Carve out a true held-out test set touched exactly once, after model/feature selection on val.
- Try a small 1D-CNN or transformer directly on the raw multichannel signal as a stronger baseline than flattened-raw + XGBoost.

## Setup

```bash
pip install numpy scipy matplotlib seaborn pandas scikit-learn xgboost lightgbm catboost
```

Dataset: [CHB-MIT Seizure subset (Kaggle)](https://www.kaggle.com/datasets/adibadea/chbmitseizuredataset)

## Repo contents

- `eeg-classification.ipynb` — full pipeline: EDA → raw baseline → frequency features → multi-model benchmark.
