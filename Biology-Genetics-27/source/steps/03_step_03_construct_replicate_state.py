"""
Construct the numerical matrix state associated with the aligned replicate coordinates.

Let X denote rg_replicates and let B be its number of rows. Return B times the sample covariance matrix of the columns of X, using denominator B - 1.

Returns
-------
replicate_state : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def construct_replicate_state(
    rg_replicates: np.ndarray,
) -> np.ndarray:
    """Return the matrix state for aligned replicate coordinates.

    Parameters
    ----------
    rg_replicates : np.ndarray
        Two-dimensional array of aligned block-deletion replicates
        in the supplied packed coordinate order.

    Returns
    -------
    np.ndarray
        Square matrix with both axes aligned to the packed
        coordinate order.
    """
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_construct_replicate_state(
    rg_replicates: np.ndarray,
) -> np.ndarray:
    replicates = np.asarray(
        rg_replicates,
        dtype=float,
    )

    if (
        replicates.ndim != 2
        or replicates.shape[0] < 2
        or replicates.shape[1] < 1
    ):
        raise ValueError(
            "rg_replicates must have shape "
            "(n_blocks, n_coordinates)"
        )

    if not np.all(np.isfinite(replicates)):
        raise ValueError(
            "rg_replicates must contain only finite values"
        )

    if np.any(np.abs(replicates) > 1.0 + 1e-12):
        raise ValueError(
            "replicate correlations must lie in [-1, 1]"
        )

    n_blocks, n_coordinates = replicates.shape

    discriminant = 1 + 8 * n_coordinates
    root = int(round(np.sqrt(discriminant)))

    if (
        root * root != discriminant
        or (1 + root) % 2 != 0
    ):
        raise ValueError(
            "packed width must be a "
            "strict-lower-triangle size"
        )

    replicate_state = (
        n_blocks
        * np.atleast_2d(
            np.cov(
                replicates,
                rowvar=False,
                ddof=1,
            )
        )
    )

    replicate_state = 0.5 * (
        replicate_state
        + replicate_state.T
    )

    expected_shape = (
        n_coordinates,
        n_coordinates,
    )

    if replicate_state.shape != expected_shape:
        raise ValueError(
            "replicate state has an invalid shape"
        )

    if not np.all(np.isfinite(replicate_state)):
        raise ValueError(
            "replicate state must contain only finite values"
        )

    return replicate_state

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for construct_replicate_state."""
    return [
        {
            "setup": """import numpy as np

rg_replicates = np.array([
    [0.246, -0.166, 0.121, 0.229, 0.020, 0.267],
    [0.243, -0.148, 0.110, 0.193, 0.049, 0.304],
    [0.241, -0.112, 0.117, 0.193, 0.044, 0.273],
    [0.258, -0.141, 0.107, 0.196, 0.052, 0.287],
    [0.231, -0.149, 0.118, 0.217, 0.044, 0.298],
    [0.283, -0.150, 0.102, 0.194, 0.046, 0.310],
    [0.238, -0.161, 0.098, 0.223, 0.047, 0.300],
    [0.227, -0.134, 0.112, 0.214, 0.049, 0.294],
    [0.254, -0.138, 0.114, 0.223, 0.025, 0.284],
    [0.231, -0.156, 0.106, 0.240, 0.031, 0.307],
], dtype=float)
""",
            "call": (
                "construct_replicate_state(rg_replicates)"
            ),
            "gold_call": (
                "_oracle_construct_replicate_state("
                "rg_replicates)"
            ),
        },
        {
            "setup": """import numpy as np

rg_replicates = np.array([
    [0.31, 0.12, 0.22],
    [0.29, 0.15, 0.20],
    [0.34, 0.10, 0.24],
    [0.28, 0.13, 0.19],
    [0.32, 0.11, 0.23],
    [0.30, 0.14, 0.21],
    [0.33, 0.09, 0.25],
], dtype=float)[::-1]
""",
            "call": (
                "construct_replicate_state(rg_replicates)"
            ),
            "gold_call": (
                "_oracle_construct_replicate_state("
                "rg_replicates)"
            ),
        },
        {
            "setup": """import numpy as np

rg_replicates = np.array([
    [0.38],
    [0.42],
    [0.39],
    [0.41],
], dtype=float)
""",
            "call": (
                "construct_replicate_state(rg_replicates)"
            ),
            "gold_call": (
                "_oracle_construct_replicate_state("
                "rg_replicates)"
            ),
        },
        {
            "setup": """import numpy as np

rg_replicates = np.array([
    [0.20, 0.10],
    [0.22, 0.12],
    [0.18, 0.08],
    [0.21, 0.11],
], dtype=float)

def run_model():
    try:
        construct_replicate_state(rg_replicates)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_construct_replicate_state(
            rg_replicates,
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
