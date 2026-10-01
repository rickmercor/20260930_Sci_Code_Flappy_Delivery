"""
Accumulate the Euclidean path length traced by ordering-mismatch vectors over an ordered configuration sequence.

The final benchmark follows the full three-component mismatch across increasingly convection-dominated, high-wavenumber configurations, so all three spectral effects contribute to the cumulative path.

Returns
-------
float, the cumulative Euclidean path length
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cumulative_spectral_ordering_path(configurations: "np.ndarray", tau_star: float) -> float:
    """Return the cumulative Euclidean path length of ordering mismatches.

    Parameters
    ----------
    configurations : np.ndarray
        Float array of shape (n, 4), n >= 2, with rows
        ``[order, K_star, a_star, kappa_star]`` in path order. Supported orders
        are 2, 3, and 4.
    tau_star : float
        Time-first dimensionless fine-scale parameter shared by all rows.

    Returns
    -------
    path_length : float
        Sum of Euclidean distances between consecutive three-component
        spectral ordering mismatch vectors.
    """
    return path_length

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_cumulative_spectral_ordering_path(configurations: "np.ndarray", tau_star: float) -> float:
    configurations = np.asarray(configurations, dtype=float)
    mismatch_vectors = []
    for row in configurations:
        mismatch_vectors.append(
            _oracle_spectral_ordering_mismatch(
                int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(tau_star)
            )
        )
    mismatch_vectors = np.asarray(mismatch_vectors, dtype=float)
    increments = np.linalg.norm(np.diff(mismatch_vectors, axis=0), axis=1)
    return float(np.sum(increments))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the benchmark path, a shorter prefix, and an alternate path."""
    return [
        {
            "setup": """import numpy as np
configurations=np.array([[2,.75,.12,.060],[3,1.10,.22,.050],[4,1.45,.32,.040],[2,1.80,.40,.030],[3,2.15,.48,.025],[4,2.50,.56,.020]],dtype=float)
tau_star=.37""",
            "call": "cumulative_spectral_ordering_path(configurations.copy(),tau_star)",
            "gold_call": "_oracle_cumulative_spectral_ordering_path(configurations.copy(),tau_star)",
            "tol": 5e-12,
        },
        {
            "setup": """import numpy as np
configurations=np.array([[2,.75,.12,.060],[3,1.10,.22,.050],[4,1.45,.32,.040],[2,1.80,.40,.030]],dtype=float)
tau_star=.37""",
            "call": "cumulative_spectral_ordering_path(configurations.copy(),tau_star)",
            "gold_call": "_oracle_cumulative_spectral_ordering_path(configurations.copy(),tau_star)",
            "tol": 5e-12,
        },
        {
            "setup": """import numpy as np
configurations=np.array([[2,.62,.11,.075],[3,1.02,.19,.058],[4,1.52,.31,.041],[3,1.98,.45,.029],[4,2.38,.54,.021]],dtype=float)
tau_star=.43""",
            "call": "cumulative_spectral_ordering_path(configurations.copy(),tau_star)",
            "gold_call": "_oracle_cumulative_spectral_ordering_path(configurations.copy(),tau_star)",
            "tol": 5e-12,
        },
    ]
