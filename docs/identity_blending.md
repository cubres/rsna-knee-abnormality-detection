# Blend predictions by study identity

`src/prediction_blend.py` combines probability tables after aligning exact study IDs and declared label names. It uses the requested output order, even when input rows and columns arrive in a different order.

Requires Python 3.12 and the standard library. From the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The 32 synthetic tests cover identity collisions, missing studies, malformed probabilities, zero-weight inputs, and numeric overflow. They validate the table contract; they do not measure MRI model accuracy.

For the import example below, start Python from the repository root with `PYTHONPATH=src python`, or run your own script with `PYTHONPATH=src python your_script.py`.

```python
from prediction_blend import blend_prediction_tables

first = [{"study": "a", "abnormal": 0.1},
         {"study": "b", "abnormal": 0.9}]
second = [{"study": "b", "abnormal": 0.3},
          {"study": "a", "abnormal": 0.7}]

result = blend_prediction_tables(
    [first, second],
    study_ids=["b", "a"],
    label_names=["abnormal"],
    weights=[1, 3],
    id_column="study",
)
# [{'study': 'b', 'abnormal': 0.45},
#  {'study': 'a', 'abnormal': 0.55}]
```

For a real inference table, obtain `study_ids` and `label_names` from the competition's sample submission. Supply every label explicitly. The function accepts finite probabilities in [0, 1] and nonnegative finite weights with a positive total. It scales weights before summing to avoid overflow.

Duplicate IDs are errors, including duplicate rows with identical predictions. Every input must contain exactly the requested studies. IDs remain exact strings: `01`, `1`, and `1 ` are distinct. Extra metadata columns are ignored, but missing probability columns are errors. A zero-weight table still undergoes all validation.

`read_prediction_csv(stream)` accepts an open text stream and checks the CSV header and row shape. Blending accepts its numeric strings directly. The module neither chooses ensemble weights nor writes a submission. Validate model quality independently before selecting weights.

This original utility and its tests carry the MIT license in their source files.
