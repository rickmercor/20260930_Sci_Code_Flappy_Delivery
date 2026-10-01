"""
Training and validation evidence are computed separately across frozen expression resources. Two differential-expression summaries are used for each resource-gene pair: the case-control mean difference and the case-control median difference. Their two-sided empirical p-values use one seeded stream of label permutations per split, reused for every resource, method, and gene, with the finite-permutation correction p = (1 + c) / (B + 1).

Inputs

------

adjusted_expression: Float array of shape (n_samples, n_resources, n_genes).

labels: Binary phenotype array of shape (n_samples,).

split_ids: Integer array of shape (n_samples,), with training coded 0 and validation coded 1.

n_permutations: Positive integer B.

seed: Integer seed for the permutation generator.

Returns

-------

effects: Float array of shape (2, n_resources, 2, n_genes), with mean then median methods.

p_values: Float array with the same shape as effects.

Returns
-------
tuple of two float64 arrays shaped (2, n_resources, 2, n_genes), effects then p-values
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_split_associations(
    adjusted_expression: "np.ndarray",
    labels: "np.ndarray",
    split_ids: "np.ndarray",
    n_permutations: int,
    seed: int,
) -> "tuple[np.ndarray, np.ndarray]":
    """Compute split-specific mean and median permutation associations.

    Parameters
    ----------
    adjusted_expression : np.ndarray
        Adjusted values with shape (n_samples, n_resources, n_genes).
    labels : np.ndarray
        Binary case-control labels with one entry per sample.
    split_ids : np.ndarray
        Split identifiers, zero for training and one for validation.
    n_permutations : int
        Positive number of random label permutations per split.
    seed : int
        Seed for a single NumPy random generator.

    Raises
    ------
    ValueError
        If array dimensions or sample counts disagree, values are non-finite,
        labels are not binary, split identifiers are not exactly zero or one,
        either split lacks a case or control, `n_permutations` is not a
        positive integer, or `seed` is not an integer.

    Returns
    -------
    effects : np.ndarray
        Mean and median case-control differences with shape
        (2, n_resources, 2, n_genes).
    p_values : np.ndarray
        Corrected empirical two-sided p-values with the same shape.
    """
    return association_results  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811

def _association_statistic(values, labels, method_index):
    """Return a case-control location difference."""
    cases = values[labels == 1]
    controls = values[labels == 0]
    if method_index == 0:
        return float(cases.mean() - controls.mean())
    return float(np.median(cases) - np.median(controls))


def _oracle_compute_split_associations(
    adjusted_expression: "np.ndarray",
    labels: "np.ndarray",
    split_ids: "np.ndarray",
    n_permutations: int,
    seed: int,
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference implementation."""
    expression = np.asarray(adjusted_expression, dtype=float)
    labels = np.asarray(labels)
    split_ids = np.asarray(split_ids)
    if expression.ndim != 3:
        raise ValueError("adjusted_expression must be three dimensional")
    if labels.shape != (expression.shape[0],):
        raise ValueError("labels must have one entry per sample")
    if split_ids.shape != (expression.shape[0],):
        raise ValueError("split_ids must have one entry per sample")
    if not np.all(np.isfinite(expression)):
        raise ValueError("adjusted_expression must be finite")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be binary")
    if not np.all((split_ids == 0) | (split_ids == 1)):
        raise ValueError("split_ids must contain only zero and one")
    if set(np.unique(split_ids).tolist()) != {0, 1}:
        raise ValueError("both training and validation splits must be present")
    for split_index in (0, 1):
        split_labels = labels[split_ids == split_index]
        if set(np.unique(split_labels).tolist()) != {0, 1}:
            raise ValueError("each split must contain at least one case and control")
    if not isinstance(n_permutations, (int, np.integer)) or n_permutations <= 0:
        raise ValueError("n_permutations must be a positive integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    n_resources = expression.shape[1]
    n_genes = expression.shape[2]
    effects = np.empty((2, n_resources, 2, n_genes), dtype=float)
    p_values = np.empty_like(effects)
    rng = np.random.default_rng(int(seed))
    for split_index in (0, 1):
        in_split = split_ids == split_index
        split_labels = labels[in_split].astype(int)
        split_expression = expression[in_split]
        permuted_labels = np.array(
            [rng.permutation(split_labels) for _ in range(n_permutations)],
            dtype=int,
        )
        for resource_index in range(n_resources):
            for method_index in range(2):
                for gene_index in range(n_genes):
                    values = split_expression[:, resource_index, gene_index]
                    observed = _association_statistic(
                        values, split_labels, method_index
                    )
                    null_statistics = np.array(
                        [
                            _association_statistic(values, permuted, method_index)
                            for permuted in permuted_labels
                        ]
                    )
                    exceedances = np.count_nonzero(
                        np.abs(null_statistics) + 1e-12 >= abs(observed)
                    )
                    effects[
                        split_index, resource_index, method_index, gene_index
                    ] = observed
                    p_values[
                        split_index, resource_index, method_index, gene_index
                    ] = (1.0 + exceedances) / (n_permutations + 1.0)
    return effects, p_values

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
E = np.array([[[0., 1.]], [[2., 0.]], [[1., 2.]], [[3., 1.]], [[0., 2.]], [[2., 0.]], [[1., 3.]], [[4., 1.]]])
y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
split = np.array([0, 0, 0, 0, 1, 1, 1, 1])
""",
            "call": "[x.tolist() for x in compute_split_associations(E, y, split, 31, 7)]",
            "gold_call": "[x.tolist() for x in _oracle_compute_split_associations(E, y, split, 31, 7)]",
        },
        {
            "setup": """import numpy as np
E = np.array([0., 1., 9., 2., 4., 5., 1., 3., 8., 2., 6., 7.]).reshape(12, 1, 1)
y = np.tile([0, 0, 0, 1, 1, 1], 2)
split = np.repeat([0, 1], 6)
""",
            "call": "[x.tolist() for x in compute_split_associations(E, y, split, 31, 7)]",
            "gold_call": "[x.tolist() for x in _oracle_compute_split_associations(E, y, split, 31, 7)]",
        },
        {
            "setup": """import numpy as np
E = np.array([[[0.]], [[1.]], [[2.]], [[3.]]])
y = np.array([0, 1, 0, 1])
split = np.array([0, 0, 1, 1])
""",
            "call": "[x.tolist() for x in compute_split_associations(E, y, split, 1, 0)]",
            "gold_call": "[x.tolist() for x in _oracle_compute_split_associations(E, y, split, 1, 0)]",
        },
        {
            "setup": """import numpy as np
E = np.ones((4, 1, 1))
y = np.array([0, 0, 0, 1])
split = np.array([0, 0, 1, 1])
def run_model():
    try:
        compute_split_associations(E, y, split, 7, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_compute_split_associations(E, y, split, 7, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
