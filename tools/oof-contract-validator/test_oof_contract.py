from pathlib import Path
import tempfile
import unittest

import numpy as np

from oof_contract import Artifact, Bank, Fold, sha256_file, validate_contract


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="knee_oof_contract_fixture_"))
        for kind in ("source", "state", "input", "exposure"):
            (cls.root / kind).write_bytes(kind.encode("ascii"))
        cls.artifacts = [Artifact(k, cls.root / k, k, sha256_file(cls.root / k)) for k in ("source", "state", "input", "exposure")]

    def args(self):
        return dict(
            official={"a": {"ACL": 0, "MCL": 1}, "b": {"ACL": 1, "MCL": 0}},
            label_names=("ACL", "MCL"), expected_held_ids=("a", "b"),
            banks=[Bank("model", ("a", "b"), ("ACL", "MCL"), np.array([[.1, .8], [.9, .2]]), np.array([[0, 1], [1, 0]]), ("0", "1"))],
            folds=[Fold("0", ("b",), ("a",)), Fold("1", ("a",), ("b",))],
            artifacts=self.artifacts,
            components=[{"name": "model", "source_artifacts": ["source"], "state_artifacts": ["state"], "input_artifacts": ["input"]}],
            bank_components={"model": ["model"]},
            group_by_id={"a": "ga", "b": "gb"}, group_kind="report", runtime_input_closure_attested=True,
            exposure={"status": "attested_unexposed", "artifact_names": ["exposure"]}, current_ensemble_equivalent=True,
        )

    def codes(self, **changes):
        args = self.args()
        args.update(changes)
        return {i["code"] for i in validate_contract(**args)["issues"]}

    def test_valid_contract_is_scoped_and_does_not_mutate_inputs(self):
        args = self.args()
        before = args["banks"][0].probabilities.copy()
        result = validate_contract(**args)
        self.assertEqual(result["status"], "PASS_DATA_AND_DECLARED_PROVENANCE_ONLY")
        self.assertFalse(result["quality_promotion_authorized"])
        self.assertIsNone(result["official_score"])
        self.assertFalse(result["patient_independence_proven"])
        np.testing.assert_array_equal(before, args["banks"][0].probabilities)

    def test_row_and_label_permutations_join_by_id_and_name(self):
        bank = Bank("model", ("b", "a"), ("MCL", "ACL"), np.array([[.2, .9], [.8, .1]]), np.array([[0, 1], [1, 0]]), ("1", "0"))
        self.assertEqual(self.codes(banks=[bank]), set())

    def test_fractional_labels_are_not_truncated_to_integers(self):
        bank = self.args()["banks"][0]
        bad = Bank(bank.name, bank.ids, bank.label_names, bank.probabilities, np.array([[.5, 1], [1, 0]]), bank.fold_ids)
        self.assertIn("NONBINARY_SUPPLIED_LABEL", self.codes(banks=[bad]))

    def test_mismatch_counts_are_named_and_inputs_not_repaired(self):
        args = self.args()
        bad_y = np.array([[1, 0], [1, 0]])
        args["banks"] = [Bank("model", ("a", "b"), ("ACL", "MCL"), np.full((2, 2), .5), bad_y, ("0", "1"))]
        result = validate_contract(**args)
        self.assertEqual(result["banks"][0]["label_mismatch_count"], 2)
        self.assertEqual(result["banks"][0]["label_mismatches_by_target"], {"ACL": 1, "MCL": 1})
        np.testing.assert_array_equal(bad_y, [[1, 0], [1, 0]])

    def test_nan_infinity_and_out_of_range_probabilities(self):
        bank = Bank("model", ("a", "b"), ("ACL", "MCL"), np.array([[np.nan, np.inf], [-.1, 1.1]]), np.zeros((2, 2)), ("0", "1"))
        codes = self.codes(banks=[bank])
        self.assertIn("NONFINITE_PROBABILITY", codes)
        self.assertIn("PROBABILITY_RANGE", codes)

    def test_duplicate_rows_do_not_substitute_for_coverage(self):
        bank = Bank("model", ("a", "a"), ("ACL", "MCL"), np.full((2, 2), .5), np.array([[0, 1], [0, 1]]), ("0", "0"))
        codes = self.codes(banks=[bank])
        self.assertIn("DUPLICATE_PREDICTION_ID", codes)
        self.assertIn("PREDICTION_COVERAGE", codes)

    def test_unknown_and_missing_studies_are_reported(self):
        bank = Bank("model", ("a", "unknown"), ("ACL", "MCL"), np.full((2, 2), .5), np.zeros((2, 2)), ("0", "1"))
        codes = self.codes(banks=[bank])
        self.assertIn("UNKNOWN_PREDICTION_ID", codes)
        self.assertIn("PREDICTION_COVERAGE", codes)

    def test_duplicate_and_missing_target_names_hold(self):
        for names in (("ACL", "ACL"), ("ACL", "other")):
            bank = Bank("model", ("a", "b"), names, np.full((2, 2), .5), np.zeros((2, 2)), ("0", "1"))
            self.assertIn("BANK_LABEL_SCHEMA", self.codes(banks=[bank]))

    def test_probability_and_label_shapes_hold(self):
        bank = Bank("model", ("a", "b"), ("ACL", "MCL"), np.full((1, 2), .5), np.zeros((2, 2)), ("0", "1"))
        self.assertIn("PROBABILITY_SHAPE_TYPE", self.codes(banks=[bank]))
        bank = Bank("model", ("a", "b"), ("ACL", "MCL"), np.full((2, 2), .5), np.zeros((2, 1)), ("0", "1"))
        self.assertIn("SUPPLIED_LABEL_SHAPE_TYPE", self.codes(banks=[bank]))

    def test_missing_official_labels_are_never_filled(self):
        official = self.args()["official"]
        official["a"]["ACL"] = None
        self.assertIn("MISSING_OFFICIAL_TARGET", self.codes(official=official))

    def test_missing_attached_labels_do_not_claim_complete_zero_mismatches(self):
        args = self.args()
        bank = args["banks"][0]
        args["banks"] = [Bank(bank.name, bank.ids, bank.label_names, bank.probabilities, None, bank.fold_ids)]
        report = validate_contract(**args)
        self.assertFalse(report["banks"][0]["label_comparison_complete"])
        self.assertEqual(report["banks"][0]["compared_valid_label_cells"], 0)

    def test_train_held_overlap_and_group_overlap_hold(self):
        folds = [Fold("0", ("a", "b"), ("a",)), Fold("1", ("a",), ("b",))]
        self.assertIn("TRAIN_HELD_OVERLAP", self.codes(folds=folds))
        self.assertIn("GROUP_OVERLAP", self.codes(group_by_id={"a": "same", "b": "same"}))

    def test_multiple_folds_and_incomplete_held_union_hold(self):
        folds = [Fold("0", ("b",), ("a",)), Fold("1", ("b",), ("a",))]
        codes = self.codes(folds=folds)
        self.assertIn("MULTIPLE_HELD_FOLDS", codes)
        self.assertIn("HELD_COVERAGE", codes)

    def test_prediction_fold_provenance_must_match(self):
        bank = self.args()["banks"][0]
        for fold_ids, code in ((None, "MISSING_PREDICTION_FOLD"), (("1", "0"), "PREDICTION_FOLD_MISMATCH"), (("0",), "FOLD_VECTOR_SHAPE")):
            b = Bank(bank.name, bank.ids, bank.label_names, bank.probabilities, bank.supplied_labels, fold_ids)
            self.assertIn(code, self.codes(banks=[b]))

    def test_missing_unpinned_and_drifted_artifacts_hold(self):
        for claim, code in ((Artifact("source", self.root / "absent", "source", "0"*64), "MISSING_ARTIFACT"),
                            (Artifact("source", self.root / "source", "source", None), "UNPINNED_ARTIFACT"),
                            (Artifact("source", self.root / "source", "source", "0"*64), "ARTIFACT_HASH_MISMATCH")):
            self.assertIn(code, self.codes(artifacts=[claim] + self.artifacts[1:]))

    def test_wrong_artifact_kind_and_missing_component_closure_hold(self):
        components = [{"name": "model", "source_artifacts": ["state"], "state_artifacts": [], "input_artifacts": ["missing"]}]
        self.assertIn("INCOMPLETE_COMPONENT_PROVENANCE", self.codes(components=components))
        self.assertIn("MISSING_COMPONENT_PROVENANCE", self.codes(components=[]))

    def test_exposed_and_unknown_validation_are_not_fresh(self):
        self.assertIn("VALIDATION_EXPOSED", self.codes(exposure={"status": "exposed", "artifact_names": ["exposure"]}))
        self.assertIn("VALIDATION_EXPOSURE_UNKNOWN", self.codes(exposure=None))
        self.assertIn("EXPOSURE_EVIDENCE_INCOMPLETE", self.codes(exposure={"status": "attested_unexposed", "artifact_names": []}))

    def test_state_hash_is_not_validation_exposure_evidence(self):
        self.assertIn("EXPOSURE_EVIDENCE_INCOMPLETE", self.codes(exposure={"status": "attested_unexposed", "artifact_names": ["state"]}))

    def test_runtime_and_current_ensemble_scope_are_explicit(self):
        self.assertIn("RUNTIME_CLOSURE_UNATTESTED", self.codes(runtime_input_closure_attested=False))
        self.assertIn("CURRENT_ENSEMBLE_NOT_ATTESTED", self.codes(current_ensemble_equivalent=False))

    def test_bank_provenance_cannot_borrow_an_unrelated_component(self):
        self.assertIn("BANK_COMPONENT_BINDING", self.codes(bank_components=None))
        self.assertIn("BANK_COMPONENT_BINDING", self.codes(bank_components={"model": ["other"]}))
        self.assertIn("BANK_COMPONENT_BINDING", self.codes(bank_components={"model": ["model", "model"]}))


if __name__ == "__main__":
    unittest.main()
