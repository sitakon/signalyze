# signalyze

A Python library for signal analysis, feature extraction, machine learning, and explainability — built on NumPy, SciPy, scikit-learn, SHAP, and LIME.

## Features

- **Preprocessing** — zero-phase Butterworth / Chebyshev filtering (lowpass, highpass, bandpass) and fixed-length segmentation with configurable overlap
- **Spectral analysis** — Welch PSD estimation, single-signal and comparison plots, percentile power spectrum across large signal populations
- **Feature extraction** — time-domain statistics (mean, RMS, kurtosis, …) and spectral features (band power, spectral centroid, entropy, …)
- **ML models** — train and cross-validate Random Forest, SVM, Logistic Regression, Gradient Boosting, or k-NN on raw signals, raw PSDs, or extracted features
- **Explainability** — SHAP (TreeExplainer / KernelExplainer), LIME (tabular), and permutation feature importance

## Installation

**From source (recommended for development):**

```bash
git clone https://github.com/<your-username>/signalyze.git
cd signalyze
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

**Dependencies only:**

```bash
pip install -r requirements.txt
```

## Modules

| Module | Key functions |
|---|---|
| `signalyze.preprocessing` | `filter_signal`, `segment_signal`, `preprocess_signal` |
| `signalyze.spectral` | `compute_welch`, `plot_welch`, `plot_percentile_psd`, `plot_welch_comparison` |
| `signalyze.features` | `extract_time_features`, `extract_spectral_features`, `extract_all_features`, `features_to_matrix` |
| `signalyze.models` | `build_feature_matrix`, `build_model`, `train_model`, `cross_validate_model`, `run_pipeline`, `plot_confusion_matrix`, `plot_feature_importance` |
| `signalyze.explainability` | `explain_shap`, `explain_shap_instance`, `explain_lime`, `explain_permutation`, `explain_all` |

## Quick start

### Preprocessing

```python
from signalyze.preprocessing import preprocess_signal

# Bandpass filter (1–40 Hz) then segment into 1-second windows with 50 % overlap
segments = preprocess_signal(
    raw_signal,
    fs=1000.0,
    btype="bandpass",
    cutoff=(1.0, 40.0),
    order=4,
    filter_type="butter",   # or "cheby1" / "cheby2"
    segment_length=1000,
    overlap=0.5,
)
```

### Spectral analysis

```python
from signalyze.spectral import plot_welch, plot_percentile_psd
import matplotlib.pyplot as plt

# Single-signal PSD
plot_welch(signal, fs=1000.0, log_scale=True)
plt.show()

# Percentile PSD across many signals, with one signal overlaid
ax = plot_percentile_psd(
    signals,
    fs=1000.0,
    percentiles=(5, 25, 50, 75, 95),
    overlay_signal=signals[0],
    overlay_label="example signal",
)
plt.show()
```

### Feature extraction

```python
from signalyze.features import extract_all_features, features_to_matrix

# Extract combined time + spectral features for one signal
feats = extract_all_features(signal, fs=1000.0)

# Build a matrix from a list of signals
X, feature_names = features_to_matrix(
    [extract_all_features(s, fs=1000.0) for s in signals]
)
```

### Training a model

```python
from signalyze.models import run_pipeline, plot_confusion_matrix, plot_feature_importance
import matplotlib.pyplot as plt

model, le, X, feature_names, train_res, cv_res = run_pipeline(
    signals=signals,
    labels=labels,
    fs=1000.0,
    feature_type="both",   # "time" | "spectral" | "both" | "psd" | "raw"
    model_type="rf",       # "rf" | "svm" | "lr" | "gb" | "knn"
    cv=5,
)

print(train_res["classification_report"])
print(f"CV accuracy: {cv_res['mean_accuracy']:.3f} ± {cv_res['std_accuracy']:.3f}")

plot_confusion_matrix(model, train_res["X_test"], train_res["y_test"],
                      class_names=le.classes_.astype(str))
plot_feature_importance(model, feature_names, top_n=15)
plt.show()
```

### Explainability

```python
from signalyze.explainability import explain_all
import matplotlib.pyplot as plt

results = explain_all(
    model=model,
    X_train=X_train,
    X_explain=X_test,
    y_explain=y_test,
    feature_names=feature_names,
    class_names=le.classes_.astype(str).tolist(),
    lime_instance_idx=0,
)

results["shap"]["fig_beeswarm"].savefig("shap_beeswarm.png", dpi=150)
results["lime"]["fig"].savefig("lime_instance.png", dpi=150, bbox_inches="tight")
results["permutation"]["fig"].savefig("permutation_importance.png", dpi=150)
plt.show()
```

## Requirements

| Package | Minimum version |
|---|---|
| numpy | 1.20.0 |
| scipy | 1.7.0 |
| scikit-learn | 1.0.0 |
| matplotlib | 3.4.0 |
| pandas | 1.3.0 |
| shap | 0.44.0 |
| lime | 0.2.0 |

## License

MIT — see [LICENSE](LICENSE).
