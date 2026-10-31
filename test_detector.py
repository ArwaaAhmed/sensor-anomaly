import unittest

from detector import calibrate, classification_metrics, evaluate, fit, generate, predictions, score


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.training = generate(300, 31)
        self.calibration = generate(150, 32)
        self.test = generate(180, 33, anomalies=True)

    def test_test_labels_change_metrics_not_model_or_scores(self):
        report, model, result = evaluate(self.training, self.calibration, self.test)
        changed = [{**r, "label": 1 - r["label"]} for r in self.test]
        report2, model2, result2 = evaluate(self.training, self.calibration, changed)
        self.assertEqual(model, model2)
        self.assertEqual([r["robust_score"] for r in result], [r["robust_score"] for r in result2])
        self.assertNotEqual(report["robust"], report2["robust"])

    def test_calibration_does_not_mutate_fit(self):
        model = fit(self.training)
        calibrated = calibrate(model, self.calibration)
        self.assertNotIn("threshold", model)
        self.assertEqual(model["centers"], calibrated["centers"])
        self.assertEqual(model["scales"], calibrated["scales"])

    def test_clear_sensor_fault_scores_above_normal_center(self):
        model = calibrate(fit(self.training), self.calibration)
        normal = dict(model["centers"])
        fault = {**normal, "vibration": normal["vibration"] + 100}
        self.assertEqual(predictions(model, [normal, fault]), [0, 1])

    def test_rejects_constant_and_nonfinite_training(self):
        with self.assertRaises(ValueError):
            fit([{"temperature": 1, "vibration": 1, "current": 1}] * 30)
        with self.assertRaises(ValueError):
            score(fit(self.training), [{"temperature": float("inf"), "vibration": 1, "current": 1}])

    def test_confusion_metrics_and_zero_division(self):
        result = classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(result["f1"], 0.5)
        self.assertEqual(result["false_positive_rate"], 0.5)
        self.assertEqual(classification_metrics([0], [0])["precision"], 0)

    def test_training_ignores_labels(self):
        changed = [{**r, "label": 1} for r in self.training]
        self.assertEqual(fit(self.training), fit(changed))


if __name__ == "__main__":
    unittest.main()
