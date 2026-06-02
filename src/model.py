"""Conv1D autoencoder for time-series anomaly detection.

Designed for 60-reading windows of univariate sensor data, sized to
fit on an ESP32-class microcontroller after 8-bit quantisation.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(
    window_size: int = 60,
    n_channels: int = 1,
    filters: Sequence[int] = (16, 8),
    kernel_size: int = 3,
) -> keras.Model:
    """Build a symmetric Conv1D autoencoder.

    Args:
        window_size: number of timesteps per input window.
        n_channels: number of input channels (1 for univariate).
        filters: filter counts for encoder Conv1D layers. The decoder
            mirrors these in reverse. Each Conv1D is followed by a
            2x pooling layer, so window_size must be divisible by
            2 ** len(filters).
        kernel_size: kernel width for all Conv1D layers.

    Returns:
        A compiled-ready keras.Model.
    """
    if window_size % (2 ** len(filters)) != 0:
        raise ValueError(
            f"window_size={window_size} not divisible by "
            f"2**{len(filters)} = {2 ** len(filters)}"
        )

    inputs = keras.Input(shape=(window_size, n_channels), name="window")

    # --- Encoder ---
    x = inputs
    for f in filters:
        x = layers.Conv1D(f, kernel_size, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(2, padding="same")(x)

    # --- Decoder (mirror) ---
    for f in reversed(filters):
        x = layers.Conv1D(f, kernel_size, padding="same", activation="relu")(x)
        x = layers.UpSampling1D(2)(x)

    # --- Output layer: reconstruct original signal ---
    outputs = layers.Conv1D(
        n_channels, kernel_size, padding="same", activation="linear", name="reconstruction"
    )(x)

    model = keras.Model(inputs, outputs, name="conv1d_autoencoder")
    return model


def compile_autoencoder(
    model: keras.Model,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Compile the autoencoder with MSE loss and Adam optimiser."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model

def reconstruction_error(
    model: keras.Model, X: "np.ndarray", batch_size: int = 128
) -> "np.ndarray":
    """Per-window MSE between input and reconstruction.

    Returns an array of shape (num_windows,) — one error value per window.
    """
    import numpy as np
    reconstructed = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean(np.square(X - reconstructed), axis=(1, 2))


THRESHOLD = 0.00879  # 99th percentile of training reconstruction error on
                    # machine_temperature_system_failure.csv. See findings.md for
                    # derivation. Re-derive per-stream when applied to other data.


def confidence_score(
    error: "np.ndarray", threshold: float = THRESHOLD
) -> "np.ndarray":
    """Threshold-relative anomaly score.

    confidence = (error - threshold) / threshold

    Interpretation:
      < 0  : reconstruction error below threshold; window looks normal.
      = 0  : at the threshold; boundary case.
      > 0  : above threshold; magnitude indicates how far above.
        1.0 = error is 2x threshold (clear anomaly)
        5.0 = error is 6x threshold (extreme anomaly)

    Designed for cheap ESP32 inference: one subtraction, one division.
    """
    return (error - threshold) / threshold


def predict_tflite(interpreter, X: "np.ndarray") -> "np.ndarray":
    """Run inference through a TFLite model on a batch of windows.

    Handles input/output quantisation automatically. TFLite's Python
    interpreter doesn't batch nicely, so we iterate window-by-window.
    """
    import numpy as np
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Quantise input if model expects int8
    if input_details['dtype'] == np.int8:
        scale, zero_point = input_details['quantization']
        X_in = np.clip(np.round(X / scale + zero_point), -128, 127).astype(np.int8)
    else:
        X_in = X.astype(input_details['dtype'])

    outputs = np.empty((len(X), X.shape[1], X.shape[2]), dtype=np.float32)
    for i in range(len(X_in)):
        interpreter.set_tensor(input_details['index'], X_in[i:i+1])
        interpreter.invoke()
        raw = interpreter.get_tensor(output_details['index'])
        if output_details['dtype'] == np.int8:
            scale, zero_point = output_details['quantization']
            raw = (raw.astype(np.float32) - zero_point) * scale
        outputs[i] = raw[0]

    return outputs