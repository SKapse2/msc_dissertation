"""Regenerate models/tier1_artefact.json from the current model state.

The file previously in the repo at this path was a Python source script
that was meant to *generate* the JSON, but the script itself was committed
by mistake. This rebuilds it as an actual JSON file with values derived
live from the saved models.
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import preprocessing as pp
from src import model as md
from src import evaluation as ev

import tensorflow as tf
from tensorflow import keras

# --- Data and preprocessing ---
NAB_ROOT = PROJECT_ROOT / "data" / "raw" / "NAB"
CSV_PATH = NAB_ROOT / "data" / "realKnownCause" / "machine_temperature_system_failure.csv"
LABELS_FILE = NAB_ROOT / "labels" / "combined_windows.json"
TARGET_FILE = "realKnownCause/machine_temperature_system_failure.csv"

df = pp.load_nab_stream(CSV_PATH)
anomaly_windows = pp.load_anomaly_windows(LABELS_FILE, TARGET_FILE)
train_df, val_df, test_df = pp.split_by_time(df, pp.DEFAULT_SPLIT)
train_df_clean = pp.remove_anomaly_windows(train_df, anomaly_windows)
scaler = pp.fit_scaler(train_df_clean["value"].values)

X_train = pp.window_dataframe_by_segments(train_df_clean, scaler)
X_test = pp.window_dataframe_by_segments(test_df, scaler)

# --- Float Keras reference ---
keras_path = PROJECT_ROOT / "models" / "autoencoder_v1.keras"
model = keras.models.load_model(keras_path)
err_train_keras = md.reconstruction_error(model, X_train)
err_test_keras = md.reconstruction_error(model, X_test)
threshold_keras = md.compute_threshold(err_train_keras)

# --- Int8 deployed ---
int8_path = PROJECT_ROOT / "models" / "autoencoder_v1_int8_v2.tflite"
interp_int8 = tf.lite.Interpreter(model_path=str(int8_path))
interp_int8.allocate_tensors()

X_train_recon_int8 = md.predict_tflite(interp_int8, X_train)
X_test_recon_int8 = md.predict_tflite(interp_int8, X_test)
err_train_int8 = np.mean(np.square(X_train - X_train_recon_int8), axis=(1, 2))
err_test_int8 = np.mean(np.square(X_test - X_test_recon_int8), axis=(1, 2))
threshold_int8 = md.compute_threshold(err_train_int8)

# --- Event-level metrics (300-min merge gap is now the default) ---
test_anomaly_windows = [
    (s, e)
    for s, e in anomaly_windows
    if s >= pp.DEFAULT_SPLIT.test_start and s < pp.DEFAULT_SPLIT.test_end
]

events_keras = ev.detect_events(
    err_test_keras, threshold_keras, test_df["timestamp"].values, pp.WINDOW_SIZE
)
events_int8 = ev.detect_events(
    err_test_int8, threshold_int8, test_df["timestamp"].values, pp.WINDOW_SIZE
)
scores_keras_nab = ev.score_events(events_keras, test_anomaly_windows)
scores_int8_nab = ev.score_events(events_int8, test_anomaly_windows)

# Also against the augmented label set if it exists
aug_labels_path = PROJECT_ROOT / "data" / "processed" / "augmented_labels.json"
if aug_labels_path.exists():
    with open(aug_labels_path) as f:
        aug = json.load(f)
    import pandas as pd
    augmented_windows = [
        (pd.Timestamp(s), pd.Timestamp(e)) for s, e in aug["augmented_labels"]
    ]
    scores_keras_aug = ev.score_events(events_keras, augmented_windows)
    scores_int8_aug = ev.score_events(events_int8, augmented_windows)
else:
    scores_keras_aug = None
    scores_int8_aug = None

# --- Quantisation parameters ---
in_details = interp_int8.get_input_details()[0]
out_details = interp_int8.get_output_details()[0]
in_scale, in_zp = in_details["quantization"]
out_scale, out_zp = out_details["quantization"]


def metrics_dict(s):
    if s is None:
        return None
    return {
        "tp_events": int(s["tp_events"]),
        "fp_events": int(s["fp_events"]),
        "fn_events": int(s["fn_events"]),
        "precision": round(float(s["precision"]), 4),
        "recall": round(float(s["recall"]), 4),
        "f1": round(float(s["f1"]), 4),
    }


artefact = {
    "schema_version": 2,
    "generated_by": "scripts/build_tier1_artefact.py",
    "models": {
        "reference_keras": "models/autoencoder_v1.keras",
        "reference_tflite": "models/autoencoder_v1_float.tflite",
        "deployed_tflite": "models/autoencoder_v1_int8_v2.tflite",
    },
    "window_size": int(pp.WINDOW_SIZE),
    "sampling_interval_minutes": 5,
    "preprocessing": {
        "scaler_mean": float(scaler.mean),
        "scaler_std": float(scaler.std),
        "training_period": [
            str(pp.DEFAULT_SPLIT.train_start),
            str(pp.DEFAULT_SPLIT.train_end),
        ],
        "test_period": [
            str(pp.DEFAULT_SPLIT.test_start),
            str(pp.DEFAULT_SPLIT.test_end),
        ],
        "anomaly_windows_excluded_from_training": True,
        "target_file": TARGET_FILE,
    },
    "thresholds": {
        "keras": float(threshold_keras),
        "int8": float(threshold_int8),
        "basis": "99th percentile of training reconstruction error",
        "firmware_copy": (
            "firmware/tier1_inference/main/test_reference.h :: kThresholdInt8"
        ),
    },
    "input_quantisation": {
        "scale": float(in_scale),
        "zero_point": int(in_zp),
        "representable_range_z": [
            float((-128 - in_zp) * in_scale),
            float((127 - in_zp) * in_scale),
        ],
    },
    "output_quantisation": {
        "scale": float(out_scale),
        "zero_point": int(out_zp),
    },
    "evaluation": {
        "metric": "event-level F1",
        "merge_gap_minutes": 300,
        "merge_gap_basis": "one window length at 5-minute sampling",
        "keras_vs_nab_labels": metrics_dict(scores_keras_nab),
        "int8_vs_nab_labels": metrics_dict(scores_int8_nab),
        "keras_vs_augmented_labels": metrics_dict(scores_keras_aug),
        "int8_vs_augmented_labels": metrics_dict(scores_int8_aug),
        "augmented_labels_source": (
            "data/processed/augmented_labels.json"
            if aug_labels_path.exists()
            else None
        ),
    },
    "decision_rule": {
        "confidence_formula": "(error - threshold) / threshold",
        "escalation_rule": "confidence > 0 -> escalate to Tier 2",
    },
}

out_path = PROJECT_ROOT / "models" / "tier1_artefact.json"
with open(out_path, "w") as f:
    json.dump(artefact, f, indent=2)

print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")
print()
print(json.dumps(artefact, indent=2))