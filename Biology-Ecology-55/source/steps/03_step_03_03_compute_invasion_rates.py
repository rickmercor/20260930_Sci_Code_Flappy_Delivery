"""
At a generalized Lotka-Volterra state x, the per-capita growth vector is g(x) = r + A x. Entries on the support vanish at equilibrium, while entries outside the support are invasion growth rates. Non-positive invasion growth for every absent species is the saturation, or non-invasibility, condition. Rows that do not represent admissible candidate equilibria remain all-NaN so they cannot enter later classification.

Returns
-------
Per-capita growth array of shape (Q, N).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_invasion_rates(
    interaction: np.ndarray,
    growth: np.ndarray,
    equilibria: np.ndarray,
) -> np.ndarray:
    '''Evaluate resident and absent-species per-capita growth rates.

    Parameters
    ----------
    interaction : np.ndarray
        Finite square GLV interaction matrix A.
    growth : np.ndarray
        Finite intrinsic growth vector r with one entry per species.
    equilibria : np.ndarray
        Candidate equilibria; each row must be entirely finite or entirely NaN.

    Raises
    ------
    ValueError
        If the arrays have incompatible shapes, interaction or growth is
        non-finite, or an equilibrium row mixes finite and non-finite entries.

    Returns
    -------
    invasion_rates : np.ndarray
        Per-capita growth vectors, with all-NaN rows retained for rejected
        candidates.
    '''
    return invasion_rates  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_compute_invasion_rates(
    interaction: np.ndarray,
    growth: np.ndarray,
    equilibria: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    interaction = np.asarray(interaction, dtype=float)
    growth = np.asarray(growth, dtype=float)
    equilibria = np.asarray(equilibria, dtype=float)
    if interaction.ndim != 2 or interaction.shape[0] != interaction.shape[1]:
        raise ValueError("interaction must be square")
    n_species = interaction.shape[0]
    if growth.shape != (n_species,):
        raise ValueError("growth must have one entry per species")
    if equilibria.ndim != 2 or equilibria.shape[1] != n_species:
        raise ValueError("equilibria must have shape (n_supports, n_species)")
    if not np.all(np.isfinite(interaction)) or not np.all(np.isfinite(growth)):
        raise ValueError("interaction and growth must be finite")
    finite_rows = np.all(np.isfinite(equilibria), axis=1)
    nan_rows = np.all(np.isnan(equilibria), axis=1)
    if not np.all(finite_rows | nan_rows):
        raise ValueError("each equilibrium row must be finite or all-NaN")

    invasion_rates = np.full_like(equilibria, np.nan, dtype=float)
    invasion_rates[finite_rows] = (
        growth[None, :] + equilibria[finite_rows] @ interaction.T
    )
    return invasion_rates

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": '''interaction = np.array([[-1.0, -0.2], [-0.3, -1.0]])
growth = np.ones(2)
equilibria = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.851063829787234, 0.74468085106383],
    [np.nan, np.nan],
])
''',
            "call": "np.nan_to_num(compute_invasion_rates(interaction, growth, equilibria), nan=999.0).tolist()",
            "gold_call": "np.nan_to_num(_oracle_compute_invasion_rates(interaction, growth, equilibria), nan=999.0).tolist()",
        },
        {
            "setup": '''interaction = np.array([[-1.0]])
growth = np.array([1.0])
equilibria = np.array([[0.0]])
''',
            "call": "compute_invasion_rates(interaction, growth, equilibria).tolist()",
            "gold_call": "_oracle_compute_invasion_rates(interaction, growth, equilibria).tolist()",
        },
        {
            "setup": '''interaction = -np.eye(2)
growth = np.ones(2)
equilibria = np.array([[1.0, np.nan]])
def run_model():
    try:
        compute_invasion_rates(interaction, growth, equilibria)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_compute_invasion_rates(interaction, growth, equilibria)
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
