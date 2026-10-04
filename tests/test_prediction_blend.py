# MIT License
# Copyright (c) 2026 cubres
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Adversarial identity and numeric contracts; synthetic inputs only.

From the repository root: PYTHONPATH=src python -m unittest discover -s tests -v
"""
import copy
import io
import math
import unittest

from prediction_blend import blend_prediction_tables, read_prediction_csv


class PredictionContracts(unittest.TestCase):
    def setUp(self):
        self.first = [{"study": "a", "label_x": .1, "label_y": .8},
                      {"study": "b", "label_x": .9, "label_y": .2}]
        self.second = [{"study": "b", "label_y": .6, "label_x": .3},
                       {"study": "a", "label_y": .4, "label_x": .7}]

    def blend(self, tables=None, *, ids=("b", "a"), labels=("label_y", "label_x"), weights=(1, 3)):
        return blend_prediction_tables(tables or [self.first, self.second], study_ids=ids,
                                       label_names=labels, weights=weights, id_column="study")

    def test_shuffled_rows_and_columns_align_by_identity(self):
        output = self.blend()
        self.assertEqual([r["study"] for r in output], ["b", "a"])
        self.assertEqual(list(output[0]), ["study", "label_y", "label_x"])
        self.assertAlmostEqual(output[0]["label_x"], .45)
        self.assertAlmostEqual(output[1]["label_x"], .55)
        self.assertAlmostEqual(output[0]["label_y"], .5)
        self.assertAlmostEqual(output[1]["label_y"], .5)

    def test_duplicate_table_id_is_rejected_even_with_identical_values(self):
        self.first.append(copy.deepcopy(self.first[0]))
        with self.assertRaisesRegex(ValueError, "duplicate study"): self.blend()

    def test_missing_id_is_rejected(self):
        self.first.pop()
        with self.assertRaisesRegex(ValueError, "missing=.*b"): self.blend()

    def test_extra_id_is_rejected(self):
        self.first.append({"study": "c", "label_x": .3, "label_y": .4})
        with self.assertRaisesRegex(ValueError, "extra=.*c"): self.blend()

    def test_duplicate_requested_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"): self.blend(ids=("a", "a"))

    def test_numeric_identity_is_not_coerced_to_string(self):
        self.first[0]["study"] = 1
        with self.assertRaisesRegex(ValueError, "invalid study"): self.blend()

    def test_whitespace_in_identity_is_not_stripped(self):
        self.first[0]["study"] = "a "
        with self.assertRaisesRegex(ValueError, "study mismatch"): self.blend()

    def test_similar_numeric_strings_remain_distinct(self):
        rows = [{"study": "01", "p": .2}, {"study": "1", "p": .8}]
        out = blend_prediction_tables([rows], study_ids=["1", "01"], label_names=["p"], weights=[1], id_column="study")
        self.assertEqual(out, [{"study": "1", "p": .8}, {"study": "01", "p": .2}])

    def test_missing_identity_column_is_rejected(self):
        self.first[0].pop("study")
        with self.assertRaisesRegex(ValueError, "missing study"): self.blend()

    def test_missing_label_in_one_row_is_rejected(self):
        self.first[1].pop("label_y")
        with self.assertRaisesRegex(ValueError, "missing labels"): self.blend()

    def test_duplicate_declared_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"): self.blend(labels=("label_x", "label_x"))

    def test_identity_cannot_be_declared_as_a_label(self):
        with self.assertRaisesRegex(ValueError, "cannot also"): self.blend(labels=("study",))

    def test_probabilities_must_be_finite_and_in_range(self):
        for invalid in [float("nan"), float("inf"), -float("inf"), -.01, 1.01, "NaN", "Infinity", "bad", True, None]:
            with self.subTest(invalid=invalid):
                self.first[0]["label_x"] = invalid
                with self.assertRaises(ValueError): self.blend()

    def test_decimal_strings_outside_bounds_are_not_rounded_into_validity(self):
        for invalid in ["1.0000000000000000000001", "-1e-999"]:
            with self.subTest(invalid=invalid):
                self.first[0]["label_x"] = invalid
                with self.assertRaisesRegex(ValueError, "outside"): self.blend()

    def test_explicit_weights_must_be_finite_nonnegative(self):
        for invalid in [-1, float("nan"), float("inf"), "-1e-999", True]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError): self.blend(weights=(1, invalid))

    def test_all_zero_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"): self.blend(weights=(0, 0))

    def test_weight_count_must_match_tables(self):
        for weights in [[], [1], [1, 2, 3]]:
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(ValueError, "one explicit weight"): self.blend(weights=weights)

    def test_zero_weight_table_is_still_validated(self):
        self.second[0]["label_x"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"): self.blend(weights=(1, 0))

    def test_zero_weight_is_allowed_when_another_is_positive(self):
        out = self.blend(weights=(1, 0))
        self.assertEqual(out[0]["label_x"], .9)

    def test_large_finite_weights_do_not_overflow_the_sum(self):
        out = self.blend(weights=(1e308, 1e308))
        self.assertTrue(all(math.isfinite(r[k]) for r in out for k in ("label_x", "label_y")))
        self.assertAlmostEqual(out[0]["label_x"], .6)

    def test_generator_tables_are_consumed_once(self):
        out = self.blend(tables=(iter(t) for t in [self.first, self.second]))
        self.assertAlmostEqual(out[0]["label_x"], .45)

    def test_inputs_are_preserved(self):
        original = copy.deepcopy([self.first, self.second])
        self.blend()
        self.assertEqual([self.first, self.second], original)

    def test_extra_metadata_is_ignored_without_changing_label_order(self):
        self.first[0]["metadata"] = "kept by caller"
        self.assertEqual(list(self.blend()[0]), ["study", "label_y", "label_x"])

    def test_csv_probabilities_are_validated_and_blended(self):
        rows = read_prediction_csv(io.StringIO("study,label_x,label_y\nb,0.9,0.2\na,0.1,0.8\n"))
        out = self.blend(tables=[rows, self.second])
        self.assertAlmostEqual(out[0]["label_x"], .45)

    def test_duplicate_csv_label_headers_are_rejected_before_mapping(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_prediction_csv(io.StringIO("study,p,p\na,0.1,0.2\n"))

    def test_duplicate_csv_identity_headers_are_rejected_before_mapping(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_prediction_csv(io.StringIO("study,study,p\na,b,0.1\n"))

    def test_malformed_csv_width_is_rejected(self):
        for text in ["study,p\na,0.1,extra\n", "study,p\na\n"]:
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "expected"): read_prediction_csv(io.StringIO(text))

    def test_empty_csv_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"): read_prediction_csv(io.StringIO(""))

    def test_unclosed_csv_quote_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid prediction CSV"):
            read_prediction_csv(io.StringIO('study,p\na,"0.1\n'))

    def test_probability_endpoints_are_preserved(self):
        rows = [{"study": "a", "p": 0}, {"study": "b", "p": 1}]
        out = blend_prediction_tables([rows, list(reversed(rows))], study_ids=["b", "a"], label_names=["p"], weights=[1e308, 1e308], id_column="study")
        self.assertEqual(out, [{"study": "b", "p": 1.0}, {"study": "a", "p": 0.0}])

    def test_empty_requested_ids_or_labels_are_rejected(self):
        with self.assertRaises(ValueError): self.blend(ids=[])
        with self.assertRaises(ValueError): self.blend(labels=[])

    def test_empty_prediction_tables_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            blend_prediction_tables([], study_ids=["a"], label_names=["p"], weights=[], id_column="study")


if __name__ == "__main__":
    unittest.main(verbosity=2)
