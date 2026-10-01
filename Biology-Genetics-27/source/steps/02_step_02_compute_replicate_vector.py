"""
Compute the numerical vector associated with the supplied matrix state and aligned replicate states.

rg_replicates contains aligned replicate correlation states packed in strict-lower-triangle column-major order:

(1,0), (2,0), ..., (n-1,0), (2,1), ..., (n-1,n-2).

Each row reconstructs a symmetric matrix with unit diagonal.

For a full-sample quantity theta and B corresponding replicate values theta_b, define

p_b = B * theta - (B - 1) * theta_b

and

s(theta) = sd(p_b) / sqrt(B),

using sample standard deviation with denominator B - 1.

For each matrix state, remove target_index to form A and b, solve A x = b, and set q = b.T @ x.

For each non-target index j, let theta_j be its correlation with target_index. Return

[s(q), theta_1 / s(theta_1), ..., theta_k / s(theta_k)]

in the existing non-target index order.

Returns
-------
vector : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_replicate_vector(
    rg_matrix: np.ndarray,
    rg_replicates: np.ndarray,
    target_index: int,
) -> np.ndarray:
    """Return the numerical vector for the supplied replicate states.

    Parameters
    ----------
    rg_matrix : np.ndarray
        Square full-sample correlation matrix.

    rg_replicates : np.ndarray
        Aligned packed replicate correlations in the supplied
        strict-lower-triangle column-major order.

    target_index : int
        Zero-based index identifying the target trait.

    Returns
    -------
    np.ndarray
        One-dimensional numerical vector in the specified order.
    """
    return np.empty(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_replicate_vector(
    rg_matrix: np.ndarray,
    rg_replicates: np.ndarray,
    target_index: int,
) -> np.ndarray:
    rg = np.asarray(rg_matrix, dtype=float)
    replicates = np.asarray(
        rg_replicates,
        dtype=float,
    )

    if (
        rg.ndim != 2
        or rg.shape[0] != rg.shape[1]
        or rg.shape[0] < 2
    ):
        raise ValueError(
            "rg_matrix must be square with at least two traits"
        )

    if not np.all(np.isfinite(rg)):
        raise ValueError(
            "rg_matrix must contain only finite values"
        )

    if not isinstance(target_index, (int, np.integer)):
        raise ValueError(
            "target_index must be an integer"
        )

    target = int(target_index)
    n_traits = rg.shape[0]

    if target < 0 or target >= n_traits:
        raise ValueError(
            "target_index is out of range"
        )

    if not np.allclose(
        rg,
        rg.T,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "rg_matrix must be symmetric"
        )

    if not np.allclose(
        np.diag(rg),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "rg_matrix must have a unit diagonal"
        )

    if np.any(np.abs(rg) > 1.0 + 1e-12):
        raise ValueError(
            "correlations must lie in [-1, 1]"
        )

    expected_columns = (
        n_traits * (n_traits - 1) // 2
    )

    if (
        replicates.ndim != 2
        or replicates.shape[0] < 2
        or replicates.shape[1] != expected_columns
    ):
        raise ValueError(
            "rg_replicates has incompatible shape"
        )

    if not np.all(np.isfinite(replicates)):
        raise ValueError(
            "rg_replicates must contain only finite values"
        )

    if np.any(
        np.abs(replicates) > 1.0 + 1e-12
    ):
        raise ValueError(
            "replicate correlations must lie in [-1, 1]"
        )

    n_blocks = replicates.shape[0]

    packed_pairs = [
        (row, column)
        for column in range(n_traits - 1)
        for row in range(column + 1, n_traits)
    ]

    replicate_matrices = np.repeat(
        np.eye(
            n_traits,
            dtype=float,
        )[None, :, :],
        n_blocks,
        axis=0,
    )

    for packed_column, (
        row,
        column,
    ) in enumerate(packed_pairs):
        replicate_matrices[
            :,
            row,
            column,
        ] = replicates[
            :,
            packed_column,
        ]

        replicate_matrices[
            :,
            column,
            row,
        ] = replicates[
            :,
            packed_column,
        ]

    remaining = np.array(
        [
            idx
            for idx in range(n_traits)
            if idx != target
        ],
        dtype=int,
    )

    point_b = rg[
        remaining,
        target,
    ]

    point_A = rg[
        np.ix_(
            remaining,
            remaining,
        )
    ]

    try:
        point_x = np.linalg.solve(
            point_A,
            point_b,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "remaining-trait matrix must be nonsingular"
        ) from exc

    point_scalar = float(
        point_b @ point_x
    )

    replicate_scalars = np.empty(
        n_blocks,
        dtype=float,
    )

    for block_index, matrix in enumerate(
        replicate_matrices
    ):
        replicate_b = matrix[
            remaining,
            target,
        ]

        replicate_A = matrix[
            np.ix_(
                remaining,
                remaining,
            )
        ]

        try:
            replicate_x = np.linalg.solve(
                replicate_A,
                replicate_b,
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "replicate remaining-trait matrix "
                "must be nonsingular"
            ) from exc

        replicate_scalars[
            block_index
        ] = float(
            replicate_b @ replicate_x
        )

    scalar_pseudovalues = (
        n_blocks * point_scalar
        - (n_blocks - 1)
        * replicate_scalars
    )

    scalar_scale = float(
        np.std(
            scalar_pseudovalues,
            ddof=1,
        )
        / np.sqrt(n_blocks)
    )

    standardized_pairs = np.empty(
        remaining.size,
        dtype=float,
    )

    for output_index, trait_index in enumerate(
        remaining
    ):
        point_value = float(
            rg[
                trait_index,
                target,
            ]
        )

        replicate_values = (
            replicate_matrices[
                :,
                trait_index,
                target,
            ]
        )

        pair_pseudovalues = (
            n_blocks * point_value
            - (n_blocks - 1)
            * replicate_values
        )

        pair_scale = float(
            np.std(
                pair_pseudovalues,
                ddof=1,
            )
            / np.sqrt(n_blocks)
        )

        if (
            pair_scale <= 0.0
            or not np.isfinite(pair_scale)
        ):
            raise ValueError(
                "target-pair replicate scale "
                "must be positive"
            )

        standardized_pairs[
            output_index
        ] = (
            point_value / pair_scale
        )

    vector = np.concatenate(
        (
            np.array(
                [scalar_scale],
                dtype=float,
            ),
            standardized_pairs,
        )
    )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            "replicate vector must contain "
            "only finite values"
        )

    return vector

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for the replicate vector."""

    return [
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00,  0.25, -0.15, 0.10],
    [0.25,  1.00,  0.20, 0.05],
    [-0.15, 0.20,  1.00, 0.30],
    [0.10,  0.05,  0.30, 1.00],
], dtype=float)

rg_replicates = np.array([
    [0.24, -0.14, 0.11, 0.21, 0.04, 0.29],
    [0.26, -0.16, 0.09, 0.19, 0.06, 0.31],
    [0.23, -0.15, 0.10, 0.22, 0.05, 0.30],
    [0.27, -0.13, 0.12, 0.18, 0.03, 0.28],
    [0.25, -0.17, 0.08, 0.20, 0.07, 0.32],
    [0.24, -0.14, 0.10, 0.21, 0.04, 0.30],
], dtype=float)

target_index = 0
""",
            "call": (
                "compute_replicate_vector("
                "rg_matrix, rg_replicates, target_index)"
            ),
            "gold_call": (
                "_oracle_compute_replicate_vector("
                "rg_matrix, rg_replicates, target_index)"
            ),
        },
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00, 0.05,  0.25,  0.20],
    [0.05, 1.00,  0.10,  0.30],
    [0.25, 0.10,  1.00, -0.15],
    [0.20, 0.30, -0.15,  1.00],
], dtype=float)

rg_replicates = np.array([
    [0.04, 0.24, 0.21, 0.11, 0.29, -0.14],
    [0.06, 0.26, 0.19, 0.09, 0.31, -0.16],
    [0.05, 0.23, 0.22, 0.10, 0.30, -0.15],
    [0.03, 0.27, 0.18, 0.12, 0.28, -0.13],
    [0.07, 0.25, 0.20, 0.08, 0.32, -0.17],
    [0.04, 0.24, 0.21, 0.10, 0.30, -0.14],
], dtype=float)

target_index = 2
""",
            "call": (
                "compute_replicate_vector("
                "rg_matrix, rg_replicates, target_index)"
            ),
            "gold_call": (
                "_oracle_compute_replicate_vector("
                "rg_matrix, rg_replicates, target_index)"
            ),
        },
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00, 0.32],
    [0.32, 1.00],
], dtype=float)

rg_replicates = np.array([
    [0.30],
    [0.34],
    [0.31],
    [0.33],
    [0.29],
    [0.35],
], dtype=float)

target_index = 0
""",
            "call": (
                "compute_replicate_vector("
                "rg_matrix, rg_replicates, target_index)"
            ),
            "gold_call": (
                "_oracle_compute_replicate_vector("
                "rg_matrix, rg_replicates, target_index)"
            ),
        },
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00, 0.20, 0.10],
    [0.20, 1.00, 0.25],
    [0.10, 0.25, 1.00],
], dtype=float)

rg_replicates = np.array([
    [0.19, 0.11],
    [0.21, 0.09],
    [0.20, 0.10],
    [0.18, 0.12],
], dtype=float)

target_index = 0

def run_model():
    try:
        compute_replicate_vector(
            rg_matrix,
            rg_replicates,
            target_index,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_compute_replicate_vector(
            rg_matrix,
            rg_replicates,
            target_index,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
