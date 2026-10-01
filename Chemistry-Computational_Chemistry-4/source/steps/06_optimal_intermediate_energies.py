"""
Runs a deterministic multi-start gradient-based search (following the source method's Sec. II C, minimized ten times from different random initial guesses to guard against spurious local minima) over the free intermediate energies H(2,1), ..., H(2,N) of a two-state Glauber chain, and returns the best (argmin) vector found. This exposes the recovered optimal protocol itself, whose boundary behavior -- a finite jump at the initial step and a much larger jump at the final step -- is the qualitative signature reported for large changes in the energy landscape (Fig. 2).

Beyond the minimized *value* of MSE′, the central qualitative finding is about the *shape* of the optimal protocol itself: for large ΔE, the optimal intermediate energies jump discontinuously away from the initial boundary, vary smoothly through the interior, and then jump again — more sharply — into the final boundary. Recovering the actual optimal λ vector (not just the objective value it achieves) is what makes this shape verifiable: the first entry's offset from Eᵢ and the last entry's offset from E_f are exactly the two jump magnitudes reported.

Returns
-------
numpy.ndarray of shape (N,) and dtype float64, the recovered MSE′-optimal intermediate energies H(2,1),…,H(2,N)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def optimal_intermediate_energies(
    Ei: float, Ef: float, N: int, alpha: float, n_restarts: int, seed: int
) -> np.ndarray:
    '''Find the MSE-minimizing intermediate energies for a two-state
    Glauber chain via deterministic multi-start optimization.

    Parameters
    ----------
    Ei : float
        Initial energy of state 2, H(2, tau=0).
    Ef : float
        Final energy of state 2, H(2, tau=N+1).
    N : int
        Number of free intermediate energies to optimize. Must be a
        positive integer.
    alpha : float
        Tilting exponent of the objective <exp(-alpha*W)> (Step 4) being
        minimized. alpha = 2 gives the MSE' error proxy used by the source method.
    n_restarts : int
        Number of independent optimization runs from different starting
        points. Must be a positive integer.
    seed : int
        Seed passed to np.random.default_rng to deterministically generate
        the random restart starting points. Must be a non-negative integer.

    Returns
    -------
    lam_star : np.ndarray
        Array of shape (N,) and dtype float64: the intermediate energies
        H(2,1), ..., H(2,N) achieving the smallest value of the Step 4
        objective found across all restarts.

    Raises
    ------
    ValueError
        If Ei, Ef, or alpha is not convertible to a finite real scalar, if
        N is not a positive integer, if n_restarts is not a positive
        integer, or if seed is not a non-negative integer.
    '''
    return lam_star  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_optimal_intermediate_energies(
    Ei: float, Ef: float, N: int, alpha: float, n_restarts: int, seed: int
) -> np.ndarray:
    """Reference implementation."""
    from scipy.optimize import minimize

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
    if not isinstance(n_restarts, (int, np.integer)) or isinstance(n_restarts, bool) or n_restarts < 1:
        raise ValueError("n_restarts must be a positive integer.")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    def _build_H(lam):
        H2 = np.concatenate(([Ei_v], lam, [Ef_v]))
        H = np.zeros((2, N + 2))
        H[1, :] = H2
        return H

    def _objective(lam):
        return _oracle_exponential_work_average(_build_H(lam), alpha_v)

    lam_linear = Ei_v + (Ef_v - Ei_v) * np.arange(1, N + 1) / (N + 1)
    rng = np.random.default_rng(seed)
    starts = [lam_linear.copy()]
    span = max(abs(Ei_v), abs(Ef_v), 1.0) * 1.5
    for i in range(n_restarts - 1):
        if i % 2 == 0:
            starts.append(lam_linear + rng.normal(scale=3.0, size=N))
        else:
            starts.append(rng.uniform(-span, span, size=N))

    best_val = np.inf
    best_lam = lam_linear
    for x0 in starts:
        res = minimize(
            _objective, x0, method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-12},
        )
        if np.isfinite(res.fun) and res.fun < best_val:
            best_val = res.fun
            best_lam = res.x

    return best_lam

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal case: the source method's own large-DeltaE example
            "setup": "Ei = -8.0\nEf = 8.0\nN = 10\nalpha = 2.0\nn_restarts = 30\nseed = 0",
            "call": "optimal_intermediate_energies(Ei, Ef, N, alpha, n_restarts, seed)",
            "gold_call": "_oracle_optimal_intermediate_energies(Ei, Ef, N, alpha, n_restarts, seed)",
        },
        {
            # Boundary case: N=1
            "setup": "Ei = -2.0\nEf = 2.0\nN = 1\nalpha = 2.0\nn_restarts = 5\nseed = 1",
            "call": "optimal_intermediate_energies(Ei, Ef, N, alpha, n_restarts, seed)",
            "gold_call": "_oracle_optimal_intermediate_energies(Ei, Ef, N, alpha, n_restarts, seed)",
        },
        {
            # Edge case: Ei == Ef -> optimal protocol should stay essentially constant
            "setup": "Ei = 0.0\nEf = 0.0\nN = 5\nalpha = 2.0\nn_restarts = 5\nseed = 2",
            "call": "optimal_intermediate_energies(Ei, Ef, N, alpha, n_restarts, seed)",
            "gold_call": "_oracle_optimal_intermediate_energies(Ei, Ef, N, alpha, n_restarts, seed)",
        },
    ]
