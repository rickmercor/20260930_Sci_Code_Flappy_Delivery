"""
Orchestrator: approximate total-network sensitivity over Estrada edge sensitivity.

Call prior public APIs for edge (i, j) at depth k and return the prompt ratio.

Returns
-------
float, S_TN_approx / S_EE
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import importlib.util
from pathlib import Path

import numpy as np


def sensitivity_ratio_pipeline(A: np.ndarray, i: int, j: int, k: int) -> float:
    """Ratio of approximate total-network sensitivity to Estrada edge sensitivity.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix.
    i : int
        Row index (0-based).
    j : int
        Column index (0-based).
    k : int
        Krylov depth.

    Returns
    -------
    float
        Approximate total-network sensitivity divided by Estrada edge sensitivity.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_sensitivity_ratio_pipeline(
    A: np.ndarray, i: int, j: int, k: int
) -> float:
    import numpy as np

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    i = int(i)
    j = int(j)
    k = int(k)
    if A.shape != (n, n) or not (0 <= i < n and 0 <= j < n) or k < 1 or k > n:
        raise ValueError("invalid inputs")

    AT = A.T
    E = np.ones((n, n), dtype=float)
    ej = np.zeros(n, dtype=float)
    ej[j] = 1.0

    # Chain prior _oracle_* only (shared Studio namespace). Never call public
    # names: those can be rebound to the candidate on delivery.
    r_lead = _oracle_sep_orth_R_lead_entry(AT, E, ej, k)
    offdiag_energy = _oracle_compressed_offdiag_energy(AT, E, ej, k)
    v_approx = _oracle_frechet_action_start_overlap(AT, E, ej, k)
    v_exact = _oracle_exact_frechet_action_start_overlap(AT, E, ej)
    s_tn = _oracle_approximate_total_network_sensitivity(A, i, j, k)
    s_tn_exact = _oracle_exact_total_network_sensitivity(A, i, j)
    s_ee = _oracle_estrada_edge_sensitivity(A, i, j)

    for name, val in (
        ("_oracle_sep_orth_R_lead_entry", r_lead),
        ("_oracle_compressed_offdiag_energy", offdiag_energy),
        ("_oracle_frechet_action_start_overlap", v_approx),
        ("_oracle_exact_frechet_action_start_overlap", v_exact),
        ("_oracle_approximate_total_network_sensitivity", s_tn),
        ("_oracle_exact_total_network_sensitivity", s_tn_exact),
        ("_oracle_estrada_edge_sensitivity", s_ee),
    ):
        if not np.isfinite(val):
            raise ValueError(f"{name} returned a non-finite value: {val!r}")

    if offdiag_energy < 0:
        raise ValueError(
            "compressed_offdiag_energy must be non-negative (it is a squared Frobenius norm)"
        )

    def _same_ballpark(approx, exact, factor=5.0, atol=1e-8):
        if abs(exact) < atol:
            return abs(approx) < atol * factor
        ratio = approx / exact
        return (1.0 / factor) <= ratio <= factor

    if not _same_ballpark(v_approx, v_exact):
        raise ValueError(
            "frechet_action_start_overlap is not within a plausible range of "
            "exact_frechet_action_start_overlap"
        )
    if not _same_ballpark(s_tn, s_tn_exact):
        raise ValueError(
            "approximate_total_network_sensitivity is not within a plausible "
            "range of exact_total_network_sensitivity"
        )

    if abs(s_ee) < 1e-15:
        raise ValueError("Estrada edge sensitivity is zero")
    return float(s_tn / s_ee)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nA = np.array([[0.0, 1.3, 0.0, 0.4, 0.0, 0.2], [0.0, 0.0, 0.9, 0.0, 0.5, 0.0], [0.6, 0.0, 0.0, 1.1, 0.0, 0.3], [0.0, 0.7, 0.0, 0.0, 0.8, 0.0], [0.2, 0.0, 0.5, 0.0, 0.0, 1.4], [0.0, 0.3, 0.0, 0.6, 0.0, 0.0]])\ni, j, k = 2, 5, 2",
            "call": "sensitivity_ratio_pipeline(A, i, j, k)",
            "gold_call": "_oracle_sensitivity_ratio_pipeline(A, i, j, k)",
        },
        {
            "setup": "import numpy as np\nA = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.4], [0.2, 0.0, 1.2]])\ni, j, k = 1, 0, 2",
            "call": "sensitivity_ratio_pipeline(A, i, j, k)",
            "gold_call": "_oracle_sensitivity_ratio_pipeline(A, i, j, k)",
        },
        {
            "setup": "import numpy as np\nA = np.array([[0.0, 1.0], [0.5, 0.0]])\ni, j, k = 0, 1, 1",
            "call": "sensitivity_ratio_pipeline(A, i, j, k)",
            "gold_call": "_oracle_sensitivity_ratio_pipeline(A, i, j, k)",
        },
    ]
