# Model card: robust maximum standardized sensor deviation

**Owner:** Arwa Ahmed. **Status:** portfolio baseline with synthetic evaluation only.

## Intended use

Demonstrate anomaly-scoring, normal-only calibration and honest evaluation boundaries. Inputs are three finite numeric values named temperature, vibration and current. Outputs are a continuous anomaly score and binary triage flag. Not suitable for safety interlocks, autonomous equipment shutdown or regulated monitoring.

## Method and training

Each channel's median and median absolute deviation are fit on training observations; MAD is multiplied by 1.4826. The score is the maximum channel deviation. The robust threshold is the 95th percentile of a separate known-normal calibration set. The reference uses training means/population standard deviations and a fixed 3-sigma threshold. Comparisons are illustrative: the reference and robust method use different threshold policies.

No labels are read during fitting, scoring or calibration. Only held-out test labels produce evaluation metrics. The user must ensure CSV splits are disjoint and reflect normal training/calibration conditions. Calibration contamination can mask faults.

## Data provenance and actual evidence

Default data are entirely synthetic, from three independent seeded generators. Normal channels share an invented latent load with Gaussian noise; labelled test faults are large one-channel shifts. There are no real industrial assets or customer outcomes. `examples/report.json` and `examples/predictions.csv` preserve an actual default run. High recall on these large artificial shifts does not establish detection of subtle, temporal or real-world faults.

## Limitations

No time-series structure, cross-sensor dependency model, missing-value support, sensor saturation model, regime handling, uncertainty guarantee or automatic retraining. A constant training channel is rejected rather than assigned an invented tolerance. Precision and false-positive rates vary with prevalence and operating distribution. Evaluation is not a deployment certification.

## Before deployment

Establish data rights, units and equipment context; isolate validation by machine/time; compare robust and temporal/covariance baselines on realistic faults; tune thresholds against human-reviewed false-alarm costs; add drift, alert routing, audit retention and recovery procedures. Keep final control decisions with the authorized equipment operator.
