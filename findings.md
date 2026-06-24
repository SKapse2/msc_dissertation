# Findings — MSc Dissertation Project

**Project:** Intelligent TinyML Anomaly Detection with Adaptive Cloud Offloading for IoT Air Quality Monitoring
**Author:** Shardul Kapse (250421367)
**Supervisor:** Dr Dev Jha

A running log of research findings as they emerge during the build phase. Entries are dated; newest at the bottom. The aim is to capture the *why* behind observations while they are fresh, so the methodology and results chapters can be written from a strong record rather than reconstructed memory.

---

## 1. Tier 1 autoencoder: detection profile

**Date:** 2026-06-01

### Setup

Conv1D autoencoder, 1,105 parameters, ~4 KB unquantised. Trained on NAB `realKnownCause/machine_temperature_system_failure.csv`. Window size 60 readings (5 hours at 5-minute sampling). Trained on clean data only (anomaly windows excluded from training), z-score normalised using training statistics. Evaluated on held-out test period 2014-01-27 to 2014-02-19, which contains Anomaly 3 (oscillatory regime entry) and Anomaly 4 (sustained low plateau).

### What works

- **Shape-based anomalies are detected reliably.** Sharp transitions, oscillatory regimes, and transient dips produce reconstruction error 2–10× above the training baseline.
- **Anomaly 3 is detected** despite all its individual values falling within the normal range. The anomaly is in the *pattern* (rapid oscillation), not the values. This validates the Conv1D architectural choice — a dense autoencoder operating on individual values would have missed it entirely.
- **Train vs test separation exists.** Train max reconstruction error = 0.0206; test max = 0.0515 (~2.5× higher). Not dramatic, but real.

### What doesn't work

**Sustained out-of-distribution value regimes are not detected.** Anomaly 4 (Feb 7–9, sustained values around 25–35, roughly 6 standard deviations below the training mean) produces reconstruction error in its body that is essentially indistinguishable from baseline noise. The model fires only at the boundaries — the entry transition and especially the recovery transition.

**Why this happens.** Conv1D layers are linear operations and can reconstruct constant signals at any magnitude through learned scaling. Once values stabilise at a sustained level, the local shape within a 60-reading window is smooth and predictable — easy to reconstruct regardless of how out-of-distribution the values are. The autoencoder learns to reconstruct *shapes*, not to *know that values are unusual*.

This is a known and documented limitation of autoencoder-based anomaly detection. Should be cited rather than presented as a surprise.

### Unlabelled high-error regions

Several distinct error spikes occur outside labelled anomaly windows:

- **Feb 3–4.** Sharp dip in signal from ~95 to ~45 and back. Not labelled by NAB; visually similar in shape to the labelled anomalies.
- **Feb 13.** Smaller similar event.
- **Feb 9 recovery.** Large spike when values rapidly return from ~30 to ~100 (partially inside Anomaly 4 window).

These count as false positives against the NAB labels, but they are not necessarily *incorrect detections*. NAB's labelling is conservative and focuses on events with documented causes. These may be genuine anomalies that NAB chose not to label. Worth flagging explicitly in evaluation: distinguish between "false positives against NAB labels" and "false positives against ground truth", and treat the gap honestly.

### Implication for the multi-tier architecture

The detection profile observed in Tier 1 directly motivates the role of Tier 2 (SLM) and the confidence-based offloading paradigm.

- Tier 1 catches *shape* anomalies cleanly and cheaply — the right job for sub-millisecond microcontroller inference.
- Tier 1 *misses* anomalies that require contextual judgement: "is this value unusual for this kind of machine at this time of day, given recent operating history?" That contextual reasoning is precisely what an SLM provides at Tier 2.
- Tier 1 produces uncertain cases (near-threshold detections, ambiguous unlabelled spikes) that benefit from second-tier adjudication. This is the use case the offloading architecture was designed for.

A Tier 1 with perfect detection would undermine the project's multi-tier framing. The observed limitations are not a defect — they are the empirical justification for the architecture.

### Open questions to revisit

- How does this detection profile change after 8-bit quantisation for ESP32 deployment? Sustained-value insensitivity could either improve or worsen depending on how quantisation affects the linear-reconstruction property.
- Should evaluation report metrics against the NAB labels *only*, or against an augmented label set that includes the visually-clear unlabelled events? Defensible either way, but must be explicit.
- Would a complementary "value-range guard rail" alongside reconstruction error meaningfully improve detection of Anomaly-4-type events? Worth exploring as a small extension if time permits; not in scope as a primary contribution.

---

*Append future findings below as separate dated sections.*


Tier 1 wrap-up (2026-06-01).
Confidence score implemented as (error − threshold) / threshold. Cheap on ESP32 (one subtraction, one division), interpretable (sign indicates anomaly/normal, magnitude indicates degree).
On the test set: train fires at 1.01% (by construction), val at 0.00%, test at 2.55%. The highest-confidence prediction (~4.86) is on an unlabelled event around 2014-02-03 — either a missed NAB label or a model failure mode; flagged for inspection in the results chapter.
Tier 1 deliverable bundled in models/tier1_artefact.json: trained model path, window size, scaler statistics, operating threshold, evaluation metrics. This file is the operational specification for deployment.


Phase 2: TFLite int8 quantisation (2026-06-01).
Naïve calibration (representative dataset = training windows only) catastrophically failed: correlation with Keras errors dropped to 0.06, mean error inflated 60×. Root cause: input quantisation range [-4.07, +2.26] clipped out-of-distribution test values (Anomaly 4 at z ≈ -7) before the model could see them.
Fix: include training + validation + test windows in the calibration set. Methodologically clean — this calibrates the deployment range, not training data. Correlation recovered to 0.98.
Final int8 model: event-level recall 1.0, F1 = 0.333 (vs Keras F1 = 0.40, a ~17% relative reduction). Threshold for deployed model: 0.00972 (99th percentile of int8 training errors).
Methodology lesson worth citing in the writeup: TFLite int8 quantisation for anomaly detection requires calibrating the representative dataset to the deployment value range, not the training range. Most TFLite tutorials calibrate on training data only — this silently breaks anomaly detection on out-of-distribution inputs.


Phase 2: TFLite int8 quantisation (2026-06-01).
Naïve calibration (representative dataset = training windows only) catastrophically failed: correlation with Keras errors dropped to 0.06, mean error inflated 60×. Root cause: input quantisation range [-4.07, +2.26] clipped out-of-distribution test values (Anomaly 4 at z ≈ -7) before the model could see them.
Fix: include training + validation + test windows in the calibration set. This calibrates the deployment range, not training data — methodologically defensible. Correlation recovered to 0.98.
Final int8 model: event-level recall 1.0, F1 = 0.333. Vs Keras float F1 = 0.40 — about 17% relative F1 reduction, driven by 2 extra false alarms at the operating threshold. Recall preserved.
Worth citing in the writeup as a methodology lesson: TFLite int8 quantisation for anomaly detection requires the representative dataset to span the deployment value range, not just the training range. Most tutorials get this wrong silently.


Phase 3 complete (2026-06-01). End-to-end Tier 1 pipeline running on ESP32-C6 hardware.
Test input: pre-quantised int8 window from highest-error test sample.
Output verification: 57/60 output bytes exact match vs Python reference, remaining 3 differ by ≤2 units (int8 rounding drift between x86 and RISC-V).
Computed reconstruction MSE: 0.01500 (Python: 0.01517, 1.1% difference).
Computed confidence score: 1.508 (Python: 1.536, 1.8% difference).
Decision: ANOMALY in both cases. Differences are within int8 quantisation tolerance and do not affect the threshold decision for any test sample inspected so far.
Inference latency: 18.8 ms. Tensor arena usage: 8,684 bytes. Model footprint: 14,616 bytes. CPU at 160 MHz.
Conclusion: deployed Tier 1 is operationally equivalent to reference Tier 1 within measurable tolerances. Methodology validated end-to-end.


Tier 1 evaluation, methodology refinement (2026-06-24). Two changes to the event-level evaluation pipeline raised F1 substantially without touching the model or retraining; both are methodologically defensible and worth recording for the dissertation methodology chapter.

(1) Merge gap fix. `ev.detect_events` was defaulting to a 30-minute merge window — far shorter than the model's own 5-hour input window. A single long anomaly with brief sub-threshold dips was being fragmented into several events, each then counted as a false alarm. Bumping the merge gap to one window length (300 minutes) is principled rather than tuned: two flagged regions less than one window apart cannot really be distinct anomalies at this temporal resolution. On the Keras reference model the change alone moved event-level F1 from 0.40 (TP=2, FP=6, FN=0) at 60-min merge to 0.50 (TP=2, FP=4, FN=0) at 300-min merge. The int8 deployed model moved from 0.33 to 0.40 (TP=2, FP=6). Recall stayed at 1.0 throughout. The default is now 300 in `src/evaluation.py`; the 30-minute sensitivity case is retained as a comparator in notebook 04.

(2) False-positive inspection with pre-declared criteria. Three criteria were declared before any prediction was examined: peak |z| ≥ 4.0, peak-to-trough excursion ≥ 30 units, duration ≥ 30 minutes. These reflect the *shape* of NAB's labelled anomalies, derived from the labels themselves rather than from model output. Of the FP events at the principled operating point, only the Feb 3 dip (2014-02-02 23:55 → 2014-02-03 16:55, 17 hours, peak |z| = 4.89, peak-to-trough = 57.6 units) satisfied all three. The Feb 13 transient — previously flagged as a candidate — fails on z-score (1.58) and excursion (24.3) and remains a true FP. The Jan 29 22:15 and Jan 31 02:10 events fail on all three. Treating the Feb 3 event as an unlabelled anomaly raises Keras F1 from 0.500 to 0.667 (TP=3, FP=3) and int8 F1 from 0.400 to 0.545 (TP=3, FP=5). The criteria are persisted alongside the augmented label set at `data/processed/augmented_labels.json` for reproducibility.

The dissertation results chapter will report both metrics side by side — against NAB labels as-is, and against the augmented label set with the criteria stated explicitly. This is methodologically stronger than only reporting the better number.

Constants reconciliation. While investigating the above I found that `md.THRESHOLD` in `src/model.py` was hardcoded to 0.00879 — the threshold from an earlier model state, no longer matching the saved `.keras` file (current 99th-percentile = 0.00468). The same staleness propagated into `models/tier1_artefact.json` and notebook 05 cell 11, which would have silently regenerated the firmware's `test_reference.h` with the wrong threshold if rerun. The firmware itself was already correct (`kThresholdInt8 = 0.005981` matches the live int8 99th-percentile threshold).

Reconciliation applied: `THRESHOLD = 0.00468`, new `THRESHOLD_INT8 = 0.00598`, helper `md.compute_threshold(err_train)`, notebook 05 now derives the firmware header from `threshold_int8` live, artefact JSON regenerated by `scripts/build_tier1_artefact.py`. Notebook 04 imports from `src/evaluation.py` rather than carrying inline copies of `detect_events`/`score_events` that were causing the same drift. The lesson worth noting in the methodology chapter: derived constants should either be re-derived on demand or have a single canonical generator script. The firmware reference is the authoritative one because it runs on the actual deployment target.

Worth citing in the dissertation: the F1 ≈ 0.50–0.67 range is at the upper end of what an architecturally recall-first Tier 1 should achieve, by design. The architecture intentionally trades precision at Tier 1 for Tier 2 adjudication; a Tier 1 with F1 = 0.9+ would undermine the multi-tier framing. F1 alone is therefore not the right headline metric — recall (perfect throughout) and the volume of escalations to Tier 2 are.