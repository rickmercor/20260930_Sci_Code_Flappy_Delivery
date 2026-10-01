"""
Refine the positive-real kinetic root at each supplied wave number.

The Landau pole nearest the real axis controls the long-time collisionless response.

Returns
-------
np.ndarray: float rows [kappa, Re(zeta), Im(zeta), residual].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from scipy.special import wofz
import numpy as np
def solve_least_damped_roots(
    seed_table: np.ndarray, residual_tol: float = 1e-12
) -> np.ndarray:
    """Return rows ``[kappa, Re(zeta), Im(zeta), residual]``.

    Parameters
    ----------
    seed_table : np.ndarray
        Seed rows with columns ``kappa, Re(zeta), Im(zeta)``.
    residual_tol : float
        Positive convergence threshold for the dispersion residual.

    Returns
    -------
    np.ndarray
        Refined real-valued root table.

    Raises
    ------
    ValueError
        If the inputs are invalid or the requested root does not converge.
    """
    return np.empty((np.asarray(seed_table).shape[0], 4), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from scipy.special import wofz
import numpy as np
def _oracle_solve_least_damped_roots(
    seed_table: np.ndarray, residual_tol: float = 1e-12
) -> np.ndarray:
    """Reference implementation."""
    seeds = np.asarray(seed_table, dtype=float)
    tol = float(residual_tol)
    if seeds.ndim != 2 or seeds.shape[1] != 3 or seeds.shape[0] == 0:
        raise ValueError("seed_table must have shape (m, 3)")
    if not np.all(np.isfinite(seeds)) or not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("seeds must be finite and residual_tol positive")
    output = np.empty((seeds.shape[0], 4), dtype=float)
    for row, (kappa, real_seed, imag_seed) in enumerate(seeds):
        zeta = complex(real_seed, imag_seed)
        for _ in range(64):
            plasma_z = 1j * np.sqrt(np.pi) * wofz(zeta)
            response = 1.0 + zeta * plasma_z
            residual = response + kappa**2
            if abs(residual) <= tol:
                break
            derivative = plasma_z - 2.0 * zeta * response
            if abs(derivative) <= np.finfo(float).eps:
                raise ValueError("singular Newton derivative")
            step = residual / derivative
            if abs(step) > 0.75:
                step *= 0.75 / abs(step)
            zeta -= step
        final_residual = abs(1.0 + zeta * 1j * np.sqrt(np.pi) * wofz(zeta) + kappa**2)
        if final_residual > tol or zeta.real <= 0.0 or zeta.imag >= 0.0:
            raise ValueError("requested least-damped root did not converge")
        output[row] = (kappa, zeta.real, zeta.imag, final_residual)
    return output

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    return [
        {
            "setup": "seeds = np.array([[0.27,2.89114347,-0.01953971],[0.40,2.15058132,-0.16969264],[0.58,1.72809890,-0.19763240]])",
            "call": "float(np.sum(solve_least_damped_roots(seeds) * np.arange(1, 13).reshape(3, 4)))",
            "gold_call": "float(np.sum(_oracle_solve_least_damped_roots(seeds) * np.arange(1, 13).reshape(3, 4)))",
        },
        {
            "setup": "seeds = np.array([[0.18,4.11486315,-1.87028531e-5]])",
            "call": "float(solve_least_damped_roots(seeds)[0, 1])",
            "gold_call": "float(_oracle_solve_least_damped_roots(seeds)[0, 1])",
        },
        {
            "setup": "seeds = np.zeros((2, 2));\ndef status(fn):\n    try: fn(); return 0\n    except ValueError: return 1\n    except Exception: return 2",
            "call": "status(lambda: solve_least_damped_roots(seeds))",
            "gold_call": "status(lambda: _oracle_solve_least_damped_roots(seeds))",
        },
    ]
