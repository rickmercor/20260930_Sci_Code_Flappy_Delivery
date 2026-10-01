"""
Calculate the steady propagation stress using the Maxwell equal-area rule.

Necking propagation is treated as a phase separation between un-necked and necked tissue. The Maxwell construction ensures that the work done by the steady propagation stress equals the change in strain energy density.

Returns
-------
s_star : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def compute_maxwell_stress(
    lambdas: np.ndarray,
    stresses: np.ndarray,
    lambda_U: float,
    lambda_N: float
) -> float:
    '''
    Notes
    -----
    Computes the steady propagation stress s_*.
 
    Parameters
    ----------
    lambdas : np.ndarray
        Array of stretch ratios.
    stresses : np.ndarray
        Array of nominal stresses corresponding to lambdas.
    lambda_U : float
        Stretch ratio of the un-necked region.
    lambda_N : float
        Stretch ratio of the necked region.
 
    Returns
    -------
    float
        Steady propagation stress s_*.
 
    Raises
    ------
    ValueError
        If `lambdas` or `stresses` is not a one-dimensional finite array, if
        they differ in length, or if `lambda_U` or `lambda_N` is not a finite
        real number.
    '''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_maxwell_stress(lambdas: np.ndarray, stresses: np.ndarray, lambda_U: float, lambda_N: float) -> float:
        import numpy as np
        def _num(name, value):
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a real number, got {value!r}") from exc
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            return value
        lambdas = np.asarray(lambdas, dtype=float)
        stresses = np.asarray(stresses, dtype=float)
        if lambdas.ndim != 1 or stresses.ndim != 1:
            raise ValueError("lambdas and stresses must be one-dimensional")
        if lambdas.shape != stresses.shape:
            raise ValueError(f"lambdas and stresses must be the same length, got {lambdas.shape} and "
                             f"{stresses.shape}")
        if not (np.all(np.isfinite(lambdas)) and np.all(np.isfinite(stresses))):
            raise ValueError("lambdas and stresses must contain only finite values")
        lambda_U = _num("lambda_U", lambda_U)
        lambda_N = _num("lambda_N", lambda_N)
        mask = (lambdas >= lambda_U) & (lambdas <= lambda_N)
        if not np.any(mask):
            return 0.0
        
        l_sub = lambdas[mask]
        s_sub = stresses[mask]
        
        if len(l_sub) < 2 or lambda_N == lambda_U:
            return 0.0
            
        integral = np.trapezoid(s_sub, l_sub)
        s_star = integral / (lambda_N - lambda_U)
        return float(s_star)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
        return [
            # --- Normal scenario ---
            {
                "setup": """import numpy as np
lambdas = np.linspace(1.0, 2.0, 11)
stresses = np.sin(lambdas)
lambda_U = 1.2
lambda_N = 1.8""",
                "call": "compute_maxwell_stress(lambdas, stresses, lambda_U, lambda_N)",
                "gold_call": "_oracle_compute_maxwell_stress(lambdas, stresses, lambda_U, lambda_N)"
            },
            # --- Boundary case ---
            {
                "setup": """import numpy as np
lambdas = np.array([1.0, 1.5, 2.0])
stresses = np.array([2.0, 2.0, 2.0])
lambda_U = 1.5
lambda_N = 1.5""",
                "call": "compute_maxwell_stress(lambdas, stresses, lambda_U, lambda_N)",
                "gold_call": "_oracle_compute_maxwell_stress(lambdas, stresses, lambda_U, lambda_N)"
            },
            # --- Edge case ---
            {
                "setup": """import numpy as np
lambdas = np.array([1.0, 1.1, 1.2])
stresses = np.array([0.0, 0.0, 0.0])
lambda_U = 1.5
lambda_N = 2.0""",
                "call": "compute_maxwell_stress(lambdas, stresses, lambda_U, lambda_N)",
                "gold_call": "_oracle_compute_maxwell_stress(lambdas, stresses, lambda_U, lambda_N)"
            }
        ]
