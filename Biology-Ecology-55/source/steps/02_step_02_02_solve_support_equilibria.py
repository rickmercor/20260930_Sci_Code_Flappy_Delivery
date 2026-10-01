"""
For generalized Lotka-Volterra dynamics in vector form, dx/dt = diag(x)(r + A x), a candidate equilibrium on support I solves A_II x_I = -r_I and has zero abundance outside I. A support is admissible only when this restricted system is nonsingular and every resident abundance is strictly greater than the numerical tolerance. The empty support represents extinction and gives the zero vector; inadmissible supports are retained as all-NaN rows so later classifications stay aligned with the support enumeration.

Returns
-------
Array of shape (Q, N), with all-NaN rows for inadmissible supports.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_support_equilibria(
    interaction: np.ndarray,
    growth: np.ndarray,
    support_masks: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    '''Solve the GLV equilibrium equations on supplied supports.

    Parameters
    ----------
    interaction : np.ndarray
        Finite square interaction matrix A of shape (N, N).
    growth : np.ndarray
        Finite intrinsic growth vector r of shape (N,).
    support_masks : np.ndarray
        Binary array of candidate supports with shape (Q, N).
    tol : float
        Finite positive threshold below which a resident is not admissible.

    Raises
    ------
    ValueError
        If the matrix, vector, or support shapes are inconsistent, if an input
        contains non-finite values, if a support mask is not binary, or if tol
        is not finite and positive.

    Returns
    -------
    equilibria : np.ndarray
        Array of shape (Q, N); inadmissible supports are represented by all-NaN
        rows.
    '''
    return equilibria  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_solve_support_equilibria(
    interaction: np.ndarray,
    growth: np.ndarray,
    support_masks: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    """Reference implementation."""
    interaction = np.asarray(interaction, dtype=float)
    growth = np.asarray(growth, dtype=float)
    support_masks = np.asarray(support_masks)
    if interaction.ndim != 2 or interaction.shape[0] != interaction.shape[1]:
        raise ValueError("interaction must be a square matrix")
    n_species = interaction.shape[0]
    if n_species == 0 or growth.shape != (n_species,):
        raise ValueError("growth must have shape (N,)")
    if support_masks.ndim != 2 or support_masks.shape[1] != n_species:
        raise ValueError("support_masks must have shape (Q, N)")
    if not np.all(np.isfinite(interaction)) or not np.all(np.isfinite(growth)):
        raise ValueError("interaction and growth must be finite")
    if not np.all((support_masks == 0) | (support_masks == 1)):
        raise ValueError("support_masks must be binary")
    if not isinstance(tol, (int, float, np.integer, np.floating)) or not np.isfinite(tol):
        raise ValueError("tol must be finite and positive")
    tol = float(tol)
    if tol <= 0.0:
        raise ValueError("tol must be finite and positive")

    equilibria = np.full((support_masks.shape[0], n_species), np.nan, dtype=float)
    for row, mask in enumerate(support_masks.astype(bool)):
        resident = np.flatnonzero(mask)
        if resident.size == 0:
            equilibria[row] = 0.0
            continue
        restricted = interaction[np.ix_(resident, resident)]
        try:
            resident_abundance = np.linalg.solve(restricted, -growth[resident])
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(resident_abundance)) and np.all(resident_abundance > tol):
            equilibria[row] = 0.0
            equilibria[row, resident] = resident_abundance
    return equilibria

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": '''interaction = np.array([[-1.0, -0.2], [-0.3, -1.0]])
growth = np.ones(2)
support_masks = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
''',
            "call": "np.nan_to_num(solve_support_equilibria(interaction, growth, support_masks), nan=999.0).tolist()",
            "gold_call": "np.nan_to_num(_oracle_solve_support_equilibria(interaction, growth, support_masks), nan=999.0).tolist()",
        },
        {
            "setup": '''interaction = np.array([[-1.0]])
growth = np.array([1.0])
support_masks = np.array([[0]])
''',
            "call": "solve_support_equilibria(interaction, growth, support_masks).tolist()",
            "gold_call": "_oracle_solve_support_equilibria(interaction, growth, support_masks).tolist()",
        },
        {
            "setup": '''interaction = np.zeros((2, 3))
growth = np.ones(2)
support_masks = np.array([[0, 0], [1, 0]])
def run_model():
    try:
        solve_support_equilibria(interaction, growth, support_masks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_solve_support_equilibria(interaction, growth, support_masks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
''',
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
