"""
The full pipeline enumerates every support once, solves and classifies all equilibrium candidates for each interaction matrix, summarizes stable-state multiplicity by richness, constructs one normalized probability vector per system, and compares system-centered with pooled-state weighting. The returned scalar is the signed excess p_state^k - p_system^k at target_richness k. This file is the orchestrator and is the step marked final in Studio.

Returns
-------
Native float equal to p_state^k - p_system^k.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_full_pipeline(
    interaction_matrices: np.ndarray,
    growth: np.ndarray,
    target_richness: int = 2,
    tol: float = 1e-9,
) -> float:
    '''Run the stable-state ensemble analysis and return the pooling bias.

    Parameters
    ----------
    interaction_matrices : np.ndarray
        Finite array of shape (n_systems, n_species, n_species).
    growth : np.ndarray
        Shared finite intrinsic growth vector with one entry per species.
    target_richness : int
        Richness k between zero and n_species, inclusive.
    tol : float
        Finite positive tolerance for feasibility and stability decisions.

    Raises
    ------
    ValueError
        If the ensemble is empty or has an invalid shape, growth is
        incompatible, target_richness is invalid, or tol is not positive.

    Returns
    -------
    pooling_bias : float
        Signed excess of pooled-state over system-centered probability at k.
    '''
    return pooling_bias  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_run_full_pipeline(
    interaction_matrices: np.ndarray,
    growth: np.ndarray,
    target_richness: int = 2,
    tol: float = 1e-9,
) -> float:
    """Reference implementation chaining every earlier step."""
    interaction_matrices = np.asarray(interaction_matrices, dtype=float)
    growth = np.asarray(growth, dtype=float)
    if interaction_matrices.ndim != 3 or interaction_matrices.shape[0] == 0:
        raise ValueError("interaction_matrices must be a nonempty three-dimensional array")
    if interaction_matrices.shape[1] == 0 or interaction_matrices.shape[1] != interaction_matrices.shape[2]:
        raise ValueError("each interaction matrix must be nonempty and square")
    n_species = interaction_matrices.shape[1]
    if growth.shape != (n_species,):
        raise ValueError("growth must have one entry per species")
    if not np.all(np.isfinite(interaction_matrices)) or not np.all(np.isfinite(growth)):
        raise ValueError("interaction_matrices and growth must be finite")
    if isinstance(target_richness, (bool, np.bool_)) or not isinstance(
        target_richness, (int, np.integer)
    ):
        raise ValueError("target_richness must be an integer from zero to n_species")
    target_richness = int(target_richness)
    if target_richness < 0 or target_richness > n_species:
        raise ValueError("target_richness must be an integer from zero to n_species")
    if not isinstance(tol, (int, float, np.integer, np.floating)) or not np.isfinite(tol):
        raise ValueError("tol must be finite and positive")
    tol = float(tol)
    if tol <= 0.0:
        raise ValueError("tol must be finite and positive")

    support_masks = _oracle_enumerate_supports(n_species)  # noqa: F821
    summaries = []
    for interaction in interaction_matrices:
        equilibria = _oracle_solve_support_equilibria(  # noqa: F821
            interaction, growth, support_masks, tol
        )
        invasion_rates = _oracle_compute_invasion_rates(  # noqa: F821
            interaction, growth, equilibria
        )
        spectral_abscissae = _oracle_compute_spectral_abscissae(  # noqa: F821
            interaction, growth, equilibria
        )
        stable_flags = _oracle_classify_stable_states(  # noqa: F821
            support_masks, equilibria, invasion_rates, spectral_abscissae, tol
        )
        summaries.append(
            _oracle_summarize_system_richness(support_masks, stable_flags)  # noqa: F821
        )
    system_summaries = np.asarray(summaries, dtype=int)
    probability_vectors = _oracle_build_system_probability_vectors(  # noqa: F821
        system_summaries
    )
    comparison = _oracle_compare_richness_weightings(  # noqa: F821
        system_summaries, probability_vectors, target_richness
    )
    return float(comparison[2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": '''interaction_matrices = np.array(
    [
        [
            [-1.000, -0.190, -1.970, -2.048],
            [-2.143, -1.000, -2.115, -0.598],
            [-2.132, -0.301, -1.000, -2.039],
            [-1.176, -0.312, -1.717, -1.000],
        ],
        [
            [-1.000, 0.000, 0.000, 0.000],
            [0.000, -1.000, 0.000, 0.000],
            [0.000, 0.000, -1.000, 0.000],
            [0.000, 0.000, 0.000, -1.000],
        ],
        [
            [-1.000, -1.400, 0.000, 0.000],
            [-1.300, -1.000, 0.000, 0.000],
            [0.000, 0.000, -1.000, -1.500],
            [0.000, 0.000, -1.200, -1.000],
        ],
        [
            [-1.000, -0.705, -1.304, -1.376],
            [-2.253, -1.000, -0.702, -0.570],
            [-1.216, -2.339, -1.000, -1.598],
            [-0.909, -1.757, -1.510, -1.000],
        ],
        [
            [-1.000, -0.914, -2.085, -0.513],
            [-0.613, -1.000, -1.965, -0.092],
            [-1.578, -1.251, -1.000, -1.194],
            [-0.374, -0.853, -1.392, -1.000],
        ],
        [
            [-1.000, -0.487, -0.474, -0.207],
            [-0.796, -1.000, -1.951, -0.719],
            [-1.105, -0.352, -1.000, -1.139],
            [-0.098, -1.496, -2.160, -1.000],
        ],
    ],
    dtype=float,
)
growth = np.ones(4, dtype=float)
target_richness = 2
tol = 1e-9
''',
            "call": "float(run_full_pipeline(interaction_matrices, growth, target_richness, tol))",
            "gold_call": "float(_oracle_run_full_pipeline(interaction_matrices, growth, target_richness, tol))",
        },
        {
            "setup": '''interaction_matrices = np.array([[[-1.0]]])
growth = np.array([1.0])
target_richness = 1
tol = 1e-9
''',
            "call": "float(run_full_pipeline(interaction_matrices, growth, target_richness, tol))",
            "gold_call": "float(_oracle_run_full_pipeline(interaction_matrices, growth, target_richness, tol))",
        },
        {
            "setup": '''interaction_matrices = np.zeros((2, 3))
growth = np.ones(3)
def run_model():
    try:
        run_full_pipeline(interaction_matrices, growth)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_run_full_pipeline(interaction_matrices, growth)
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
