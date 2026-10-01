"""
Evaluates the MSE' error proxy (Step 4, alpha=2) for the naive linear interpolation of the intermediate Hamiltonians between the boundary energies, H(2,tau) = Ei + (Ef-Ei)*tau/(N+1) (the source method's Eq. 18). This is the reference baseline against which the source method's optimal intermediates are compared (Fig. 3a): it reports that optimal intermediates reduce the MSE by up to an order of magnitude relative to this baseline for large changes in the energy landscape.

Before evaluating whether unrestricted optimization actually improves on established practice, the source method benchmarks its optimal intermediates against the most widely used interpolation scheme for both equilibrium and non-equilibrium free energy calculations: linear interpolation of the Hamiltonian itself between its boundary values (Eq. 18). Computing MSE′ for this specific, parameter-free protocol gives a concrete reference point — for the source method's large-ΔE two-state example, this baseline is nearly an order of magnitude worse than the optimized result, which is the central quantitative claim the optimization (Steps 6–7) is measured against.

Returns
-------
float, the MSE′ value obtained under linear interpolation of the intermediate Hamiltonians, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def linear_interpolation_mse(Ei: float, Ef: float, N: int, alpha: float) -> float:
    '''Evaluate MSE' for linearly interpolated intermediate Hamiltonians.

    Parameters
    ----------
    Ei : float
        Initial energy of state 2, H(2, tau=0).
    Ef : float
        Final energy of state 2, H(2, tau=N+1).
    N : int
        Number of intermediate time steps. Must be a positive integer.
    alpha : float
        Tilting exponent of the objective <exp(-alpha*W)> (Step 4).
        alpha = 2 gives the MSE' error proxy used by the source method.

    Returns
    -------
    result : float
        Native Python float, the value of <exp(-alpha*W)> (Step 4)
        evaluated on the linearly interpolated trajectory
        H(2, tau) = Ei + (Ef - Ei) * tau / (N + 1) for tau = 0, ..., N+1.

    Raises
    ------
    ValueError
        If Ei, Ef, or alpha is not convertible to a finite real scalar, or
        if N is not a positive integer.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_linear_interpolation_mse(Ei: float, Ef: float, N: int, alpha: float) -> float:
    """Reference implementation."""

    import numpy as np
    
    try:
        Ei_v = float(Ei)
        Ef_v = float(Ef)
        alpha_v = float(alpha)
    except (TypeError, ValueError):
        raise ValueError("Ei, Ef, and alpha must be convertible to real scalars.")
    if not (np.isfinite(Ei_v) and np.isfinite(Ef_v) and np.isfinite(alpha_v)):
        raise ValueError("Ei, Ef, and alpha must be finite real numbers.")
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or N < 1:
        raise ValueError("N must be a positive integer.")

    lam_linear = Ei_v + (Ef_v - Ei_v) * np.arange(1, N + 1) / (N + 1)
    H2 = np.concatenate(([Ei_v], lam_linear, [Ef_v]))
    H = np.zeros((2, N + 2))
    H[1, :] = H2
    return _oracle_exponential_work_average(H, alpha_v)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal case: reproduces the source method's own baseline comparison point
            "setup": "Ei = -8.0\nEf = 8.0\nN = 10\nalpha = 2.0",
            "call": "linear_interpolation_mse(Ei, Ef, N, alpha)",
            "gold_call": "_oracle_linear_interpolation_mse(Ei, Ef, N, alpha)",
        },
        {
            # Boundary case: N=1, a single intermediate step
            "setup": "Ei = -3.0\nEf = 3.0\nN = 1\nalpha = 2.0",
            "call": "linear_interpolation_mse(Ei, Ef, N, alpha)",
            "gold_call": "_oracle_linear_interpolation_mse(Ei, Ef, N, alpha)",
        },
        {
            # Edge case: Ei == Ef -> H(2,tau) is constant, so W=0 and MSE' == 1.0 exactly
            "setup": "Ei = 3.0\nEf = 3.0\nN = 6\nalpha = 2.0",
            "call": "linear_interpolation_mse(Ei, Ef, N, alpha)",
            "gold_call": "_oracle_linear_interpolation_mse(Ei, Ef, N, alpha)",
        },
    ]
