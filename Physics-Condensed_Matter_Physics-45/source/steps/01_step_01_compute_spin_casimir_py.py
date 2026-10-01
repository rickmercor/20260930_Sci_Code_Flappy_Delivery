"""
Compute the finite-spin Casimir retained by the semiclassical spin-correlation mapping.

The semiclassical correlation framework retains the quantum single-spin Casimir rather than replacing it with the classical spin-length value. For spin quantum number $S$ and reduced Planck constant $\hbar$, the invariant is

$$

Q=S(S+1)\hbar^2.

$$

This quantity enters the nonlinear correlation dynamics of the isolated Heisenberg dimer.

Returns
-------
float, the finite-spin Casimir $Q=S(S+1)\hbar^2$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_spin_casimir(S: float, hbar: float) -> float:
    """Compute the quantum single-spin Casimir.

    Parameters
    ----------
    S : float
        Spin quantum number.
    hbar : float
        Reduced Planck constant in the units used by the calculation.

    Returns
    -------
    Q : float
        The finite-spin Casimir $S(S+1)\hbar^2$.
    """
    return Q

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_spin_casimir(S: float, hbar: float) -> float:
    """Reference implementation."""
    return float(S * (S + 1.0) * hbar**2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "S = 1.5\nhbar = 1.0",
            "call": "compute_spin_casimir(S, hbar)",
            "gold_call": "_oracle_compute_spin_casimir(S, hbar)",
        },
        {
            "setup": "S = 0.5\nhbar = 1.0",
            "call": "compute_spin_casimir(S, hbar)",
            "gold_call": "_oracle_compute_spin_casimir(S, hbar)",
        },
        {
            "setup": "S = 2.0\nhbar = 0.5",
            "call": "compute_spin_casimir(S, hbar)",
            "gold_call": "_oracle_compute_spin_casimir(S, hbar)",
        },
    ]
