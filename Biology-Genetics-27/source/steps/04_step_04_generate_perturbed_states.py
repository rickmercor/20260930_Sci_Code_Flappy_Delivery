"""
Generate deterministic retained perturbations of the supplied matrix state.

replicate_state is aligned to the packed strict-lower-triangle coordinate order defined in Scientific Background.

For each proposal, use L = np.linalg.cholesky(replicate_state) and draw the packed perturbation as L @ rng.standard_normal(m), where m is the packed width and rng = np.random.default_rng(seed). Place the packed values into a symmetric zero-diagonal matrix in the supplied coordinate order and add it to rg_matrix.

Retain a proposal only when the complete perturbed matrix is positive definite. Continue until sample_count states have been retained. If proposal_limit proposals are exhausted for any required state, raise ValueError.

Return one row per retained state. The packed perturbation occupies all but the final column; the final column contains the number of proposals required for that retained state.

Returns
-------
perturbed_states : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def generate_perturbed_states(
    rg_matrix: np.ndarray,
    replicate_state: np.ndarray,
    sample_count: int,
    proposal_limit: int,
    seed: int,
) -> np.ndarray:
    """Return deterministic retained perturbations of a matrix state.

    Parameters
    ----------
    rg_matrix : np.ndarray
        Square point correlation matrix for the current trait state.
    replicate_state : np.ndarray
        Square matrix aligned to the packed strict-lower-triangle
        coordinates of rg_matrix.
    sample_count : int
        Number of retained states to return.
    proposal_limit : int
        Maximum proposals allowed for one retained state.
    seed : int
        Seed for the deterministic random-number stream.

    Returns
    -------
    np.ndarray
        One row per retained state. The packed perturbation occupies
        all but the final column; the final column contains the
        proposal count for that row.
    """
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_generate_perturbed_states(
    rg_matrix: np.ndarray,
    replicate_state: np.ndarray,
    sample_count: int,
    proposal_limit: int,
    seed: int,
) -> np.ndarray:
    rg = np.asarray(rg_matrix, dtype=float)
    state = np.asarray(
        replicate_state,
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

    try:
        np.linalg.cholesky(rg)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "rg_matrix must be positive definite"
        ) from exc

    n_traits = rg.shape[0]
    n_coordinates = (
        n_traits * (n_traits - 1) // 2
    )

    if state.shape != (
        n_coordinates,
        n_coordinates,
    ):
        raise ValueError(
            "replicate_state has an incompatible shape"
        )

    if not np.all(np.isfinite(state)):
        raise ValueError(
            "replicate_state must contain only finite values"
        )

    if not np.allclose(
        state,
        state.T,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "replicate_state must be symmetric"
        )

    if (
        not isinstance(
            sample_count,
            (int, np.integer),
        )
        or isinstance(
            sample_count,
            (bool, np.bool_),
        )
        or int(sample_count) < 1
    ):
        raise ValueError(
            "sample_count must be a positive integer"
        )

    if (
        not isinstance(
            proposal_limit,
            (int, np.integer),
        )
        or isinstance(
            proposal_limit,
            (bool, np.bool_),
        )
        or int(proposal_limit) < 1
    ):
        raise ValueError(
            "proposal_limit must be a positive integer"
        )

    if (
        not isinstance(
            seed,
            (int, np.integer),
        )
        or isinstance(
            seed,
            (bool, np.bool_),
        )
        or int(seed) < 0
    ):
        raise ValueError(
            "seed must be a non-negative integer"
        )

    try:
        factor = np.linalg.cholesky(state)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "replicate_state must be positive definite"
        ) from exc

    packed_pairs = [
        (row, column)
        for column in range(n_traits - 1)
        for row in range(column + 1, n_traits)
    ]

    rng = np.random.default_rng(int(seed))

    result = np.empty(
        (
            int(sample_count),
            n_coordinates + 1,
        ),
        dtype=float,
    )

    for sample_index in range(
        int(sample_count)
    ):
        retained = False

        for proposal_count in range(
            1,
            int(proposal_limit) + 1,
        ):
            packed_error = (
                factor
                @ rng.standard_normal(
                    n_coordinates
                )
            )

            error_matrix = np.zeros(
                (n_traits, n_traits),
                dtype=float,
            )

            for value, (row, column) in zip(
                packed_error,
                packed_pairs,
            ):
                error_matrix[row, column] = value
                error_matrix[column, row] = value

            perturbed = rg + error_matrix

            try:
                np.linalg.cholesky(perturbed)
            except np.linalg.LinAlgError:
                continue

            result[
                sample_index,
                :-1,
            ] = packed_error

            result[
                sample_index,
                -1,
            ] = float(proposal_count)

            retained = True
            break

        if not retained:
            raise ValueError(
                "proposal limit exhausted before retention"
            )

    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for generate_perturbed_states."""
    return [
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00, 0.75, 0.65],
    [0.75, 1.00, 0.80],
    [0.65, 0.80, 1.00],
], dtype=float)

replicate_state = np.array([
    [0.040, 0.010, 0.008],
    [0.010, 0.030, 0.006],
    [0.008, 0.006, 0.025],
], dtype=float)

sample_count = 6
proposal_limit = 20
seed = 5
""",
            "call": (
                "generate_perturbed_states("
                "rg_matrix, replicate_state, "
                "sample_count, proposal_limit, seed)"
            ),
            "gold_call": (
                "_oracle_generate_perturbed_states("
                "rg_matrix, replicate_state, "
                "sample_count, proposal_limit, seed)"
            ),
        },
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00, 0.90],
    [0.90, 1.00],
], dtype=float)

replicate_state = np.array([
    [0.09],
], dtype=float)

sample_count = 5
proposal_limit = 20
seed = 1
""",
            "call": (
                "generate_perturbed_states("
                "rg_matrix, replicate_state, "
                "sample_count, proposal_limit, seed)"
            ),
            "gold_call": (
                "_oracle_generate_perturbed_states("
                "rg_matrix, replicate_state, "
                "sample_count, proposal_limit, seed)"
            ),
        },
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00,  0.20, -0.10, 0.15],
    [0.20,  1.00,  0.25, 0.05],
    [-0.10, 0.25,  1.00, 0.20],
    [0.15,  0.05,  0.20, 1.00],
], dtype=float)

factor = np.array([
    [ 0.080,  0.000,  0.000,  0.000, 0.000, 0.000],
    [ 0.020,  0.070,  0.000,  0.000, 0.000, 0.000],
    [-0.010,  0.015,  0.060,  0.000, 0.000, 0.000],
    [ 0.005, -0.010,  0.012,  0.050, 0.000, 0.000],
    [ 0.010,  0.006, -0.008,  0.010, 0.045, 0.000],
    [-0.005,  0.009,  0.004, -0.006, 0.008, 0.040],
], dtype=float)

replicate_state = factor @ factor.T
sample_count = 4
proposal_limit = 20
seed = 12
""",
            "call": (
                "generate_perturbed_states("
                "rg_matrix, replicate_state, "
                "sample_count, proposal_limit, seed)"
            ),
            "gold_call": (
                "_oracle_generate_perturbed_states("
                "rg_matrix, replicate_state, "
                "sample_count, proposal_limit, seed)"
            ),
        },
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00, 0.90],
    [0.90, 1.00],
], dtype=float)

replicate_state = np.array([
    [0.09],
], dtype=float)

sample_count = 1
proposal_limit = 1
seed = 1

def run_model():
    try:
        generate_perturbed_states(
            rg_matrix,
            replicate_state,
            sample_count,
            proposal_limit,
            seed,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_generate_perturbed_states(
            rg_matrix,
            replicate_state,
            sample_count,
            proposal_limit,
            seed,
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
