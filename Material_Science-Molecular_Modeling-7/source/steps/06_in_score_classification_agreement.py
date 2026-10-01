"""
Build a confusion matrix from the last two columns of classified_rows. Use the ordering supplied in class_labels. Add row totals as the last column and column totals as the last row. Store the exact-class agreement fraction in the bottom-right element. The output must therefore have one more row and one more column than the number of class labels.

A confusion matrix compares the DFT and MLIP class labels. DFT classes are used as rows and MLIP classes are used as columns. Each retained pathway adds one count to the corresponding matrix entry. The row totals show the number of pathways in each DFT class, while the column totals show the number predicted in each MLIP class. Exact agreement is the trace divided by the total number of retained pathways.

Returns
-------
np.ndarray of floating-point values with shape (n_classes + 1, n_classes + 1). The upper-left block is the confusion matrix, the last column contains row totals, the last row contains column totals, and the bottom-right value is the exact-class agreement fraction.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def score_classification_agreement(
    classified_rows: np.ndarray,
    class_labels: np.ndarray,
) -> np.ndarray:
    """
    Build the DFT-versus-MLIP confusion matrix and calculate the
    exact-class agreement fraction.

    Parameters
    ----------
    classified_rows : np.ndarray
        Integer array of shape (n_retained, 6).
        The final two columns contain the DFT and MLIP class labels
        for every retained pathway.
    class_labels : np.ndarray
        One-dimensional integer array containing the allowed class
        labels in the required matrix order.

    Returns
    -------
    score_matrix : np.ndarray
        Floating-point array of shape
        (n_classes + 1, n_classes + 1).
        The upper-left block is the DFT-row/MLIP-column confusion
        matrix. The final column contains row totals. The final row
        contains column totals. The bottom-right element contains
        the exact-class agreement fraction.
    """
    return np.empty(
        (
            class_labels.size + 1,
            class_labels.size + 1,
        ),
        dtype=float,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_score_classification_agreement(
    classified_rows: np.ndarray,
    class_labels: np.ndarray,
) -> np.ndarray:
    """Reference implementation for augmented agreement scoring."""

    if not isinstance(classified_rows, np.ndarray):
        raise ValueError("classified_rows must be a NumPy array.")

    if (
        classified_rows.ndim != 2
        or classified_rows.shape[1] != 6
    ):
        raise ValueError(
            "classified_rows must have shape (n_rows, 6)."
        )

    if not np.issubdtype(
        classified_rows.dtype,
        np.integer,
    ):
        raise ValueError(
            "classified_rows must contain integers."
        )

    if not isinstance(class_labels, np.ndarray):
        raise ValueError(
            "class_labels must be a NumPy array."
        )

    if (
        class_labels.ndim != 1
        or class_labels.size < 1
    ):
        raise ValueError(
            "class_labels must be a nonempty one-dimensional array."
        )

    if not np.issubdtype(
        class_labels.dtype,
        np.integer,
    ):
        raise ValueError(
            "class_labels must contain integers."
        )

    labels = [
        int(value)
        for value in class_labels.tolist()
    ]

    if len(set(labels)) != len(labels):
        raise ValueError(
            "class_labels must contain different values."
        )

    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }

    n_classes = len(labels)

    confusion = np.zeros(
        (n_classes, n_classes),
        dtype=float,
    )

    for row in classified_rows:
        dft_class = int(row[-2])
        mlip_class = int(row[-1])

        if dft_class not in label_to_index:
            raise ValueError(
                "A DFT class is not present in class_labels."
            )

        if mlip_class not in label_to_index:
            raise ValueError(
                "An MLIP class is not present in class_labels."
            )

        confusion[
            label_to_index[dft_class],
            label_to_index[mlip_class],
        ] += 1.0

    total_count = float(np.sum(confusion))

    if total_count <= 0.0:
        raise ValueError(
            "At least one classified pathway is required."
        )

    score_matrix = np.zeros(
        (n_classes + 1, n_classes + 1),
        dtype=float,
    )

    score_matrix[
        :n_classes,
        :n_classes,
    ] = confusion

    score_matrix[
        :n_classes,
        n_classes,
    ] = np.sum(confusion, axis=1)

    score_matrix[
        n_classes,
        :n_classes,
    ] = np.sum(confusion, axis=0)

    score_matrix[
        n_classes,
        n_classes,
    ] = float(np.trace(confusion)) / total_count

    return score_matrix

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return unit-test cases for class-agreement scoring."""

    return [
        {
            "setup": """import numpy as np

classified_rows = np.array([
    [48, 2, 0, 1, 2, 2],
    [48, 2, 1, 2, 1, 5],
    [40, 2, 0, 2, 3, 4],
    [40, 2, 2, 1, 4, 2],
    [36, 2, 0, 2, 6, 6],
    [36, 2, 1, 1, 5, 5],
    [56, 3, 0, 2, 3, 3],
    [56, 3, 2, 1, 4, 3],
    [52, 3, 0, 1, 6, 6],
    [52, 3, 1, 1, 1, 1],
    [52, 3, 2, 1, 5, 6],
    [42, 3, 0, 3, 2, 2],
], dtype=int)

class_labels = np.array(
    [1, 2, 3, 4, 5, 6],
    dtype=int,
)""",
            "call": """score_classification_agreement(
    classified_rows,
    class_labels,
)""",
            "gold_call": """_oracle_score_classification_agreement(
    classified_rows,
    class_labels,
)""",
        },
        {
            "setup": """import numpy as np

classified_rows = np.array([
    [1, 1, 0, 1, 1, 1],
    [2, 1, 0, 1, 2, 2],
    [4, 1, 0, 1, 3, 3],
    [8, 1, 0, 1, 4, 4],
    [16, 1, 0, 1, 5, 5],
    [32, 1, 0, 1, 6, 6],
], dtype=int)

class_labels = np.array(
    [1, 2, 3, 4, 5, 6],
    dtype=int,
)""",
            "call": """score_classification_agreement(
    classified_rows,
    class_labels,
)""",
            "gold_call": """_oracle_score_classification_agreement(
    classified_rows,
    class_labels,
)""",
        },
        {
            "setup": """import numpy as np

classified_rows = np.array([
    [3, 2, 0, 1, 1, 2],
    [5, 2, 1, 1, 1, 2],
    [7, 2, 0, 1, 2, 1],
    [9, 2, 1, 1, 2, 1],
], dtype=int)

class_labels = np.array(
    [1, 2],
    dtype=int,
)""",
            "call": """score_classification_agreement(
    classified_rows,
    class_labels,
)""",
            "gold_call": """_oracle_score_classification_agreement(
    classified_rows,
    class_labels,
)""",
        },
        {
            "setup": """import numpy as np

classified_rows = np.array([
    [3, 2, 0, 1, 2, 2],
    [5, 2, 1, 1, 5, 5],
    [7, 2, 0, 1, 5, 9],
    [9, 2, 1, 1, 9, 2],
], dtype=int)

class_labels = np.array(
    [2, 5, 9],
    dtype=int,
)""",
            "call": """score_classification_agreement(
    classified_rows,
    class_labels,
)""",
            "gold_call": """_oracle_score_classification_agreement(
    classified_rows,
    class_labels,
)""",
        },
        {
            "setup": """import numpy as np

classified_rows = np.array([
    [1, 1, 0, 1, 7, 7],
], dtype=int)

class_labels = np.array(
    [7],
    dtype=int,
)""",
            "call": """score_classification_agreement(
    classified_rows,
    class_labels,
)""",
            "gold_call": """_oracle_score_classification_agreement(
    classified_rows,
    class_labels,
)""",
        },
    ]
