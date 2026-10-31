# Sensor Anomaly Triage

A reproducible anomaly-detection baseline for three numeric sensor channels. It fits robust median/MAD statistics, calibrates a whole-observation alert threshold on a separate normal set and evaluates against a fixed 3-sigma reference. An independent portfolio demonstration, not a deployed industrial safety system.

## Quick start

Requires Python 3.10+ and standard library only. No downloads, GPU or credentials.

```sh
python -m unittest -v
python detector.py --output artifacts
```

Outputs: `model.json`, `report.json`, `predictions.csv`. The checked-in `examples/` files are actual outputs from `python detector.py --output examples` using default seed 31.

To use your own data, supply all three independent files:

```sh
python detector.py --train-csv train.csv --calibration-csv normal-calibration.csv --test-csv labelled-test.csv --output artifacts/custom
```

Every CSV needs `temperature,vibration,current`; the test file also requires a binary `label`. Training and calibration need at least 20 rows. All readings must be finite, and training channels must have measurable variation. Train/calibration files must represent known-normal conditions; the CLI cannot establish this or detect overlapping input files on your behalf. Validation errors exit nonzero.

## Architecture

1. Fit independent per-channel location and scale from training data only.
2. Score each reading by its maximum absolute standardized deviation across channels.
3. Choose the empirical 95th percentile of calibration scores as the robust alert threshold. Test labels are not used.
4. Compare held-out predictions with the mean/standard-deviation baseline using a fixed threshold of 3.
5. Save confusion counts, precision, recall, F1, false-positive rate and every prediction.

The 1.4826 MAD scale factor approximates Gaussian standard deviation. A zero-variance channel raises an error instead of silently imposing an arbitrary sensor tolerance. Comments document why calibration is row-level and why model fitting stays frozen.

## Data and evidence

Default training (300 rows), calibration (150) and test (180) use independent seeds 31, 32 and 33. Normal readings are correlated Gaussian values around invented sensor units. Every ninth test observation receives a large positive shift in one sensor, with a deterministic label. No real machines, real faults, customer records or pretrained weights are involved.

The synthetic faults are intentionally easy. Results establish that the pipeline runs, not industrial fault-detection accuracy; inspect `examples/report.json` for the actual comparison, including cases where the simpler baseline performs better. Tests cover label leakage, non-mutating calibration, a known large fault, nonfinite/constant inputs and metric edge cases.

## Limits and next steps

This is statistical triage rather than a learned temporal or causal model. It ignores time, cross-channel covariance and operating regimes. Correlation-only faults may be missed; drift and contaminated calibration can cause bad thresholds. A 95th calibration percentile is not a guaranteed future false-positive rate. Real use needs sensor semantics, separate machine/time splits, subtle fault examples, drift monitoring and human investigation. See [MODEL_CARD.md](MODEL_CARD.md). No firmware, hardware control, automatic shutdown or production monitoring is included.
