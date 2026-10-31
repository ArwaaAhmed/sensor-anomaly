"""Robust sensor anomaly scoring with separate fitting, calibration and test sets."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, median, pstdev

SENSORS = ("temperature", "vibration", "current")


def generate(count: int, seed: int, anomalies: bool = False) -> list[dict]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        load = rng.gauss(0, 1)
        row = {"temperature": 45 + 1.5 * load + rng.gauss(0, 1.5),
               "vibration": 2 + 0.12 * load + rng.gauss(0, 0.16),
               "current": 8 + 0.35 * load + rng.gauss(0, 0.45)}
        label = int(anomalies and i % 9 == 0)
        if label:
            sensor = SENSORS[(i // 9) % len(SENSORS)]
            row[sensor] += {"temperature": 14, "vibration": 1.8, "current": 4.5}[sensor]
        rows.append({**{name: round(value, 6) for name, value in row.items()}, "label": label})
    return rows


def validate(rows: list[dict], min_count: int = 1) -> list[dict]:
    if len(rows) < min_count:
        raise ValueError(f"need at least {min_count} observations")
    clean = []
    for row in rows:
        values = {name: float(row[name]) for name in SENSORS}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("sensor readings must be finite")
        clean.append(values)
    return clean


def fit(rows: list[dict], robust: bool = True) -> dict:
    rows = validate(rows, min_count=20)
    centers, scales = {}, {}
    for name in SENSORS:
        values = [r[name] for r in rows]
        center = median(values) if robust else mean(values)
        # 1.4826 makes MAD comparable to standard deviation for Gaussian inputs.
        scale = 1.4826 * median(abs(v - center) for v in values) if robust else pstdev(values)
        if scale <= 1e-12:
            raise ValueError(f"{name} has no measurable variation; configure a domain tolerance instead")
        centers[name], scales[name] = center, scale
    return {"model": "median-mad-max-score" if robust else "mean-std-max-score",
            "centers": centers, "scales": scales, "training_rows": len(rows)}


def score(model: dict, rows: list[dict]) -> list[float]:
    rows = validate(rows)
    return [max(abs(r[name] - model["centers"][name]) / model["scales"][name] for name in SENSORS) for r in rows]


def calibrate(model: dict, normal_rows: list[dict], quantile: float = 0.95) -> dict:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between zero and one")
    validate(normal_rows, min_count=20)
    values = sorted(score(model, normal_rows))
    # Calibration labels are not inspected. Callers must supply a known-normal set.
    # The score is the maximum over channels, so calibration targets the whole row.
    threshold = values[math.ceil(len(values) * quantile) - 1]
    return {**model, "threshold": threshold, "calibration_rows": len(values), "calibration_quantile": quantile}


def predictions(model: dict, rows: list[dict]) -> list[int]:
    return [int(value > model["threshold"]) for value in score(model, rows)]


def classification_metrics(labels: list[int], predicted: list[int]) -> dict:
    if not labels or len(labels) != len(predicted) or any(v not in (0, 1) for v in labels + predicted):
        raise ValueError("labels and predictions must be equal nonempty binary lists")
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predicted))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None}


def evaluate(training: list[dict], calibration: list[dict], test: list[dict]) -> tuple[dict, dict, list[dict]]:
    robust = calibrate(fit(training), calibration)
    baseline = {**fit(training, robust=False), "threshold": 3.0}
    labels = []
    for row in test:
        label = int(row["label"])
        if label not in (0, 1) or float(row["label"]) != label:
            raise ValueError("test labels must be binary")
        labels.append(label)
    robust_predictions = predictions(robust, test)
    baseline_predictions = predictions(baseline, test)
    results = [{**{name: row[name] for name in SENSORS}, "label": label,
                "robust_score": value, "robust_prediction": r, "baseline_prediction": b}
               for row, label, value, r, b in zip(test, labels, score(robust, test), robust_predictions, baseline_predictions)]
    return {"counts": {"training": len(training), "calibration": len(calibration), "test": len(test)},
            "robust": classification_metrics(labels, robust_predictions),
            "mean_std_threshold_3": classification_metrics(labels, baseline_predictions),
            "method_note": "Both centers/scales fit training only. Robust threshold uses separate normal calibration rows; the baseline uses fixed 3-sigma. Test labels affect metrics only."}, robust, results


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--calibration-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    paths = (args.train_csv, args.calibration_csv, args.test_csv)
    if any(paths) and not all(paths):
        parser.error("supply all three CSV paths, or none for the synthetic demo")
    if all(paths):
        training, calibration, test = [read_csv(path) for path in paths]
        provenance = {"type": "user_csv", "files": [path.name for path in paths]}
    else:
        training, calibration, test = generate(300, args.seed), generate(150, args.seed + 1), generate(180, args.seed + 2, anomalies=True)
        provenance = {"type": "synthetic_demo", "seeds": [args.seed, args.seed + 1, args.seed + 2], "generator": "correlated Gaussian normal operation; labelled single-sensor shifts only in test"}
    report, model, results = evaluate(training, calibration, test)
    report["data_source"] = provenance
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, value in [("report.json", report), ("model.json", model)]:
        (args.output / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"report": str(args.output / "report.json"), "metrics": report["robust"]}, indent=2))


if __name__ == "__main__":
    main()
