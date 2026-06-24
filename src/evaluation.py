"""Evaluation utilities for time-series anomaly detection.

Functions for labelling windows against temporal anomaly windows,
grouping consecutive detections into events, and scoring detected
events against ground-truth labelled windows.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def label_windows_by_end_timestamp(
    timestamps: np.ndarray,
    anomaly_windows: Sequence[tuple],
    window_size: int,
) -> np.ndarray:
    """Binary label per window: True if the window's end timestamp falls
    inside any labelled anomaly window.
    """
    n_windows = len(timestamps) - window_size + 1
    end_timestamps = timestamps[window_size - 1 : window_size - 1 + n_windows]
    labels = np.zeros(n_windows, dtype=bool)
    for start, end in anomaly_windows:
        in_window = (
            (end_timestamps >= np.datetime64(start))
            & (end_timestamps <= np.datetime64(end))
        )
        labels |= in_window
    return labels


def detect_events(
    err: np.ndarray,
    threshold: float,
    timestamps: np.ndarray,
    window_size: int,
    merge_gap_minutes: int = 300,
) -> list:
    """Group consecutive flagged windows into discrete detection events.

    Detections within merge_gap_minutes of each other are merged into
    one event, so a single anomaly producing several adjacent
    above-threshold windows counts as one event.

    Default merge gap is 300 minutes = one window length at 5-minute
    sampling. Below that, brief sub-threshold dips fragment a single
    long anomaly into several events and inflate the false-alarm count;
    above that, genuinely distinct nearby anomalies would be collapsed.
    Override per-stream when sampling rate or window size differs.
    """
    flagged = err > threshold
    end_indices = np.arange(window_size - 1, window_size - 1 + len(err))
    flagged_timestamps = pd.to_datetime(timestamps[end_indices][flagged])

    if len(flagged_timestamps) == 0:
        return []

    events = []
    event_start = flagged_timestamps[0]
    event_end = flagged_timestamps[0]
    merge_gap = pd.Timedelta(minutes=merge_gap_minutes)

    for t in flagged_timestamps[1:]:
        if t - event_end <= merge_gap:
            event_end = t
        else:
            events.append((event_start, event_end))
            event_start = t
            event_end = t
    events.append((event_start, event_end))
    return events


def score_events(detected_events: list, true_windows: list) -> dict:
    """Compare detected events to labelled anomaly windows.

    Scoring convention:
      TP = number of labelled anomalies that overlapped any detected event
      FP = number of detected events that didn't overlap any labelled anomaly
      FN = number of labelled anomalies that no detected event hit

    Note: TP counts labelled anomalies, not detected events. A single
    anomaly hit by three fragmented detections still counts as TP=1.
    """
    detected_overlaps = [False] * len(detected_events)
    true_detected = [False] * len(true_windows)

    for i, (d_start, d_end) in enumerate(detected_events):
        for j, (t_start, t_end) in enumerate(true_windows):
            # Overlap: not (d ends before t starts) and not (d starts after t ends)
            if d_end >= t_start and d_start <= t_end:
                detected_overlaps[i] = True
                true_detected[j] = True

    tp = sum(true_detected)
    fp = sum(1 for x in detected_overlaps if not x)
    fn = sum(1 for x in true_detected if not x)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp_events": tp,
        "fp_events": fp,
        "fn_events": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detected_events": detected_events,
        "missed_events": [w for w, d in zip(true_windows, true_detected) if not d],
    }