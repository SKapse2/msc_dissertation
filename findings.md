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