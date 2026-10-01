"""
End-to-end orchestrator for the dissipative-preconditioning synthesized scalar.

Chain every prior public step for the given configuration and return the single synthesized scalar defined by the problem statement, which compares the realized energy-norm error of the fixed-length iteration against its own a-priori estimate and then offsets that comparison by the reduced control operator's leading diagonal entry. Propagate the structural certifications rather than recomputing them, and refuse configurations whose certifications are inconsistent with the presence of transport. A degenerate estimate is admissible only when the realized error is itself numerically zero.

Returns
-------
float, synthesized pipeline scalar from Widlund error, bound, and Hessian entry
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def orchestrate_dissipative_preconditioning_pipeline(
    n: int,
    nu: float,
    b_adv: float,
    c: float,
    b: np.ndarray,
    k: int,
    mu: float,
) -> float:
    """Run the full pipeline and return the synthesized scalar.

    
    Returns
    -------
    float
        Synthesized scalar.

    Raises
    ------
    ValueError
        If k is not a positive even integer, or upstream certifications are inconsistent.

    Assembled by calling the earlier sub-problem functions.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_orchestrate_dissipative_preconditioning_pipeline(
    n: int,
    nu: float,
    b_adv: float,
    c: float,
    b: np.ndarray,
    k: int,
    mu: float,
) -> float:
    b = np.asarray(b, dtype=float).reshape(n)
    if k < 2 or k % 2 != 0:
        raise ValueError("orchestrator requires even k for Eq. (2.3)")

    s_mag = _oracle_skew_symmetry_residual(n, nu, b_adv, c)
    k_mag = _oracle_h_skew_symmetry_residual(n, nu, b_adv, c)
    if not np.isfinite(s_mag) or s_mag < -1e-15:
        raise ValueError("invalid skew-block magnitude")
    if not np.isfinite(k_mag) or k_mag < -1e-15:
        raise ValueError("invalid H-congruence magnitude")
    if abs(b_adv) > 1e-14 and (s_mag <= 0.0 or k_mag <= 0.0):
        raise ValueError("nonzero advection must yield positive structural magnitudes")

    lam = _oracle_spectral_width_lambda(n, nu, b_adv, c)
    rho_r = _oracle_rapoport_residual_reduction_factor(n, nu, b_adv, c)
    if not np.isfinite(lam) or not np.isfinite(rho_r):
        raise ValueError("non-finite spectral diagnostics")
    if rho_r < 0.0 or rho_r > 1.0 + 1e-12:
        raise ValueError("Rapoport factor out of range")

    bound = _oracle_widlund_even_iterate_bound(n, nu, b_adv, c, k)
    err = _oracle_widlund_relative_h_error(n, nu, b_adv, c, b, k)
    k00 = _oracle_condensed_ocp_hessian_entry(n, nu, b_adv, c, mu)

    if bound == 0.0:
        if err > 1e-12:
            raise ValueError("zero Widlund bound with nonzero error")
        ratio = 0.0
    else:
        ratio = err / bound

    tau = float(ratio + k00)
    if not np.isfinite(tau):
        raise ValueError("non-finite tau")
    return tau

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nb = np.array([0.7, -1.1, 0.4, 1.2, -0.85, 0.3, 0.55, -0.2])",
            "call": "orchestrate_dissipative_preconditioning_pipeline(8, 0.37, 1.85, 0.18, b, 6, 0.07)",
            "gold_call": "_oracle_orchestrate_dissipative_preconditioning_pipeline(8, 0.37, 1.85, 0.18, b, 6, 0.07)",
        },
        {
            "setup": "import numpy as np\nb = np.ones(4)",
            "call": "orchestrate_dissipative_preconditioning_pipeline(4, 1.0, 0.0, 0.5, b, 2, 0.25)",
            "gold_call": "_oracle_orchestrate_dissipative_preconditioning_pipeline(4, 1.0, 0.0, 0.5, b, 2, 0.25)",
        },
        {
            "setup": "import numpy as np\nb = np.array([0.5, -0.25, 0.75, 0.0, 1.0, -0.5, 0.25])",
            "call": "orchestrate_dissipative_preconditioning_pipeline(7, 0.08, 1.6, 0.12, b, 4, 0.05)",
            "gold_call": "_oracle_orchestrate_dissipative_preconditioning_pipeline(7, 0.08, 1.6, 0.12, b, 4, 0.05)",
        },
        {
            "setup": "import numpy as np\nb = np.array([1.0, -1.0, 1.0, -1.0, 1.0])",
            "call": "orchestrate_dissipative_preconditioning_pipeline(5, 0.02, 2.5, 0.0, b, 4, 1e-4)",
            "gold_call": "_oracle_orchestrate_dissipative_preconditioning_pipeline(5, 0.02, 2.5, 0.0, b, 4, 1e-4)",
        },
        {
            "setup": "import numpy as np\nb = np.linspace(2.0, -1.0, 9)",
            "call": "orchestrate_dissipative_preconditioning_pipeline(9, 0.03, -3.0, 0.05, b, 6, 0.01)",
            "gold_call": "_oracle_orchestrate_dissipative_preconditioning_pipeline(9, 0.03, -3.0, 0.05, b, 6, 0.01)",
        },
        {
            "setup": "import numpy as np\nb = np.array([1.0, 0.5, -0.5, 0.25, -0.25, 0.125])",
            "call": "orchestrate_dissipative_preconditioning_pipeline(6, 0.15, 1.25, 0.05, b, 4, 0.2)",
            "gold_call": "_oracle_orchestrate_dissipative_preconditioning_pipeline(6, 0.15, 1.25, 0.05, b, 4, 0.2)",
        },
    ]
