"""
Compute the two functional-arrangement curves defined in the provided research source for a community at the supplied distance thresholds. Preserve threshold order, including repeated thresholds, and return both curves together as a two-row numerical array. Derive the source-defined statistics from the research source rather than hard-coding benchmark-specific values.

The source framework uses two complementary multi-scale summaries of species arrangement in functional space. This step converts a community's coordinates into those two observed curves, which are subsequently evaluated against alternative ecological null expectations.

Returns
-------
np.ndarray of shape (2, len(r_values)); row 0 contains PNcp and row 1 contains NNcp.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_observed_arrangement_curves(
    coords: "np.ndarray",
    r_values: "np.ndarray"
) -> "np.ndarray":
    """Compute observed PNcp and NNcp curves.

    Parameters
    ----------
    coords : np.ndarray
        Finite array of shape (n_species, n_dimensions), with at
        least two species and one dimension.
    r_values : np.ndarray
        Non-empty one-dimensional array of finite, non-negative
        distance thresholds. Input order and repeated values must
        be preserved.

    Returns
    -------
    curves : np.ndarray
        Float array of shape (2, len(r_values)).
        Row 0 contains PNcp and row 1 contains NNcp.

    Raises
    ------
    ValueError
        If the inputs are invalid.
    """
    return curves

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_observed_arrangement_curves(
    coords: "np.ndarray",
    r_values: "np.ndarray"
) -> "np.ndarray":
    coords = np.asarray(coords, dtype=float)
    r_values = np.asarray(r_values, dtype=float)

    if (
        coords.ndim != 2
        or coords.shape[0] < 2
        or coords.shape[1] < 1
    ):
        raise ValueError(
            "coords must contain at least two species and one dimension."
        )

    if not np.all(np.isfinite(coords)):
        raise ValueError(
            "coords must contain only finite values."
        )

    if r_values.ndim != 1 or r_values.size < 1:
        raise ValueError(
            "r_values must be a non-empty one-dimensional array."
        )

    if (
        not np.all(np.isfinite(r_values))
        or np.any(r_values < 0.0)
    ):
        raise ValueError(
            "r_values must contain finite non-negative values."
        )

    differences = (
        coords[:, np.newaxis, :]
        - coords[np.newaxis, :, :]
    )

    distances = np.sqrt(
        np.sum(differences ** 2, axis=2)
    )

    n_species = coords.shape[0]

    off_diagonal = distances[
        ~np.eye(n_species, dtype=bool)
    ]

    nearest_matrix = distances.copy()
    np.fill_diagonal(nearest_matrix, np.inf)

    nearest_distances = np.min(
        nearest_matrix,
        axis=1,
    )

    pncp = np.array(
        [
            np.mean(off_diagonal <= r)
            for r in r_values
        ],
        dtype=float,
    )

    nncp = np.array(
        [
            np.mean(nearest_distances <= r)
            for r in r_values
        ],
        dtype=float,
    )

    return np.vstack((pncp, nncp))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np

coords_candidate = np.array([
    [0.0, 0.0],
    [3.0, 4.0],
    [3.0, 0.0]
], dtype=float)
coords_oracle = coords_candidate.copy()

r_candidate = np.array(
    [2.9, 3.0, 4.0, 5.0],
    dtype=float
)
r_oracle = r_candidate.copy()""",
            "call": "compute_observed_arrangement_curves(coords_candidate, r_candidate)",
            "gold_call": "_oracle_compute_observed_arrangement_curves(coords_oracle, r_oracle)",
        },
        {
            "setup": """import numpy as np

coords_candidate = np.array([
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0]
], dtype=float)
coords_oracle = coords_candidate.copy()

r_candidate = np.array(
    [1.0, 0.0, 0.5, 0.0],
    dtype=float
)
r_oracle = r_candidate.copy()""",
            "call": "compute_observed_arrangement_curves(coords_candidate, r_candidate)",
            "gold_call": "_oracle_compute_observed_arrangement_curves(coords_oracle, r_oracle)",
        },
        {
            "setup": """import numpy as np

coords_candidate = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    [1.0, 1.0, 1.0]
], dtype=float)
coords_oracle = coords_candidate.copy()

r_candidate = np.array(
    [0.0, np.sqrt(3.0), 3.5],
    dtype=float
)
r_oracle = r_candidate.copy()""",
            "call": "compute_observed_arrangement_curves(coords_candidate, r_candidate)",
            "gold_call": "_oracle_compute_observed_arrangement_curves(coords_oracle, r_oracle)",
        },
    ]
