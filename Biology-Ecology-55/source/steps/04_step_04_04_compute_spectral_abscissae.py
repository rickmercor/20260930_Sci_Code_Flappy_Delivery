"""
Local asymptotic stability of a generalized Lotka-Volterra equilibrium is determined by the Jacobian J(x) = diag(r + Ax) + diag(x)A. Its spectral abscissa is the largest real part among the eigenvalues. A candidate is asymptotically stable only when this scalar is strictly negative, while rejected supports retain NaN values for alignment with the exhaustive support list.

Returns
-------
Array of length L with one maximum real eigenvalue per candidate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_spectral_abscissae(
    interaction: np.ndarray,
    growth: np.ndarray,
    equilibria: np.ndarray,
) -> np.ndarray:
    '''Compute the GLV Jacobian spectral abscissa at each candidate equilibrium.

    Parameters
    ----------
    interaction : np.ndarray
        Finite square interaction matrix A of shape (N, N).
    growth : np.ndarray
        Finite intrinsic growth vector r of shape (N,).
    equilibria : np.ndarray
        Array of shape (L, N), where each row is finite or entirely NaN.

    Raises
    ------
    ValueError
        If array shapes are incompatible, model parameters are non-finite, or
        an equilibrium row mixes finite and non-finite entries.

    Returns
    -------
    spectral_abscissae : np.ndarray
        Float array of length L. Rejected candidate rows produce NaN entries.
    '''
    return spectral_abscissae  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_compute_spectral_abscissae(
    interaction: np.ndarray,
    growth: np.ndarray,
    equilibria: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    interaction = np.asarray(interaction, dtype=float)
    growth = np.asarray(growth, dtype=float)
    equilibria = np.asarray(equilibria, dtype=float)
    if interaction.ndim != 2 or interaction.shape[0] != interaction.shape[1]:
        raise ValueError("interaction must be a square matrix")
    n_species = interaction.shape[0]
    if growth.shape != (n_species,):
        raise ValueError("growth must have shape (N,)")
    if equilibria.ndim != 2 or equilibria.shape[1] != n_species:
        raise ValueError("equilibria must have shape (L, N)")
    if not np.all(np.isfinite(interaction)) or not np.all(np.isfinite(growth)):
        raise ValueError("interaction and growth must be finite")
    finite_rows = np.all(np.isfinite(equilibria), axis=1)
    invalid_rows = np.all(np.isnan(equilibria), axis=1)
    if not np.all(finite_rows | invalid_rows):
        raise ValueError("each equilibrium row must be finite or entirely NaN")

    spectral_abscissae = np.full(equilibria.shape[0], np.nan, dtype=float)
    for row in np.flatnonzero(finite_rows):
        equilibrium = equilibria[row]
        per_capita_growth = growth + interaction @ equilibrium
        jacobian = np.diag(per_capita_growth) + equilibrium[:, None] * interaction
        eigenvalues = np.linalg.eigvals(jacobian)
        spectral_abscissae[row] = float(np.max(np.real(eigenvalues)))
    return spectral_abscissae

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": '''interaction = np.array([[-1.0, -1.4], [-1.3, -1.0]])
growth = np.ones(2)
equilibria = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [np.nan, np.nan]])
''',
            "call": "np.nan_to_num(compute_spectral_abscissae(interaction, growth, equilibria), nan=999.0).tolist()",
            "gold_call": "np.nan_to_num(_oracle_compute_spectral_abscissae(interaction, growth, equilibria), nan=999.0).tolist()",
        },
        {
            "setup": '''interaction = np.array([[-1.0]])
growth = np.array([1.0])
equilibria = np.array([[1.0]])
''',
            "call": "compute_spectral_abscissae(interaction, growth, equilibria).tolist()",
            "gold_call": "_oracle_compute_spectral_abscissae(interaction, growth, equilibria).tolist()",
        },
        {
            "setup": '''interaction = np.eye(2)
growth = np.ones(2)
equilibria = np.array([[0.0, np.nan]])
def run_model():
    try:
        compute_spectral_abscissae(interaction, growth, equilibria)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_compute_spectral_abscissae(interaction, growth, equilibria)
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
