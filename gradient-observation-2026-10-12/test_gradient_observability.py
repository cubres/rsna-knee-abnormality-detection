"""Invented CPU tensors only: no model, optimizer, backward, CUDA or API."""
import math
import random
import unittest
from unittest.mock import patch

import torch

from gradient_observability import capture


class Parameter:
    def __init__(self, gradient, requires_grad=True):
        self.grad = gradient
        self.requires_grad = requires_grad
        self.shape = (2,)


class DeclaredCollection:
    def __init__(self, gradients):
        self.values = [Parameter(g) for g in gradients]

    def named_parameters(self):
        return [(f"invented-{index}", value)
                for index, value in enumerate(self.values)]


class InventedScaler:
    def get_scale(self):
        return 2048.0

    def state_dict(self):
        return {"scale": 2048.0, "growth_factor": 2.0,
                "backoff_factor": 0.5, "growth_interval": 2000,
                "_growth_tracker": 1}

    def step(self, *args):
        raise AssertionError("observer must not step")

    def update(self, *args):
        raise AssertionError("observer must not update")


PHASE = "after_unscale_before_clip"


class GradientObservationControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def observe(self, gradients, **kwargs):
        return capture(DeclaredCollection(gradients), InventedScaler(),
                       kwargs.pop("phase", PHASE), **kwargs)

    def assert_unqualified(self, result):
        self.assertFalse(result["native_training_qualified"])
        self.assertFalse(result["parameter_updates_proven"])
        self.assertFalse(result["model_scaler_optimizer_changed_by_observer"])

    def test_finite_present(self):
        result = self.observe([torch.tensor([0.0, 2.0]), torch.tensor([-3.0, 4.0])])
        self.assertTrue(result["all_observed_required_gradients_finite_and_present"])
        self.assertEqual(result["finite_present_gradient_parameters"], 2)
        self.assertEqual(result["parameter_records"][1]["maximum_absolute_gradient"], 4.0)
        self.assert_unqualified(result)

    def test_missing_is_distinct_from_nonfinite(self):
        result = self.observe([torch.zeros(2), None])
        self.assertEqual(result["missing_gradient_parameters"], 1)
        self.assertEqual(result["nonfinite_gradient_parameters"], 0)
        self.assertEqual(result["parameter_records"][1]["classification"], "MISSING_GRADIENT")
        self.assertFalse(result["all_observed_required_gradients_finite_and_present"])

    def test_nan_and_inf_counts(self):
        result = self.observe([torch.tensor([float("nan"), float("inf")]), torch.ones(2)])
        self.assertEqual(result["missing_gradient_parameters"], 0)
        self.assertEqual(result["nonfinite_gradient_parameters"], 1)
        self.assertEqual(result["NaN_gradient_elements"], 1)
        self.assertEqual(result["Inf_gradient_elements"], 1)

    def test_mixed_missing_and_nonfinite(self):
        result = self.observe([None, torch.tensor([float("-inf"), 1.0])])
        self.assertEqual(result["missing_gradient_parameters"], 1)
        self.assertEqual(result["nonfinite_gradient_parameters"], 1)
        self.assert_unqualified(result)

    def test_observation_does_not_mutate_tensors_or_rng(self):
        gradients = [None, torch.tensor([float("nan"), float("inf")]), torch.ones(2)]
        before = [None if value is None else value.clone() for value in gradients]
        torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
        self.observe(gradients)
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        for old, current in zip(before, gradients):
            if old is None:
                self.assertIsNone(current)
            else:
                self.assertTrue(torch.equal(torch.isnan(old), torch.isnan(current)))
                self.assertTrue(torch.allclose(old, current, equal_nan=True))

    def test_two_declared_phases_use_actual_scaler_state(self):
        # Independent invented snapshots, not a forward/backward or unscale run.
        for phase in ("after_scaled_backward_before_unscale", PHASE):
            result = self.observe([torch.tensor([1.0, 2.0])], phase=phase)
            self.assertEqual(result["phase"], phase)
            self.assertEqual(result["actual_scaler_scale"], 2048.0)
            self.assertEqual(result["actual_scaler_public_state"]["_growth_tracker"], 1)
            self.assert_unqualified(result)

    def test_unknown_phase_rejected(self):
        with self.assertRaises(ValueError):
            self.observe([torch.zeros(2)], phase="after_optimizer")

    def test_unbounded_or_invalid_budget_rejected(self):
        for budget in (31, 0, -1, True, float("nan"), float("inf")):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.observe([torch.zeros(2)], wall_budget_s=budget)

    def test_record_cap_retains_aggregate_counts(self):
        result = self.observe([torch.zeros(2), None, torch.tensor([float("nan"), 0.0])],
                              max_parameter_records=1)
        self.assertEqual(len(result["parameter_records"]), 1)
        self.assertEqual(result["requires_grad_parameters_observed"], 3)
        self.assertEqual(result["parameter_records_omitted"], 2)
        self.assertEqual(result["missing_gradient_parameters"], 1)
        self.assertEqual(result["nonfinite_gradient_parameters"], 1)
        for cap in (True, 0, 1025):
            with self.assertRaises(ValueError):
                self.observe([torch.zeros(2)], max_parameter_records=cap)

    def test_partial_budget_never_qualifies(self):
        with patch("gradient_observability.time.monotonic", side_effect=[0, 1, 21, 22]):
            result = self.observe([torch.zeros(2), torch.ones(2)])
        self.assertEqual(result["status"], "PARTIAL_DIAGNOSTIC_WALL_BUDGET")
        self.assertEqual(result["requires_grad_parameters_observed"], 1)
        self.assertFalse(result["all_observed_required_gradients_finite_and_present"])

    def test_sparse_and_integer_gradients_are_unreviewed(self):
        sparse = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (2,))
        result = self.observe([sparse, torch.tensor([1, 2])])
        self.assertEqual(result["unreviewed_gradient_layout_or_dtype"], 2)
        self.assertFalse(result["all_observed_required_gradients_finite_and_present"])

    def test_empty_collection_never_passes(self):
        self.assertFalse(self.observe([])["all_observed_required_gradients_finite_and_present"])

    def test_invalid_scaler_state_rejected(self):
        class BadScale(InventedScaler):
            def get_scale(self):
                return math.inf
        with self.assertRaises(ValueError):
            capture(DeclaredCollection([torch.zeros(2)]), BadScale(), PHASE)
        class BadState(InventedScaler):
            def state_dict(self):
                return {"unexpected": [1, 2]}
        with self.assertRaises(ValueError):
            capture(DeclaredCollection([torch.zeros(2)]), BadState(), PHASE)


if __name__ == "__main__":
    unittest.main()
