"""
Fit nonnegative rate constants in normalized derivative space.

SISR compares candidate mechanisms after fitting one nonnegative rate vector across all species.

Returns
-------
Fitted rates followed by the normalized derivative loss.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_rate_constants(
    design: np.ndarray,
    derivatives: np.ndarray,
    derivative_scales: np.ndarray,
) -> np.ndarray:
    """Fit nonnegative rate constants in normalized derivative space.

    Parameters
    ----------
    design : np.ndarray
        Species-scaled design matrix in time-major row order.
    derivatives : np.ndarray
        Measured derivatives with shape (n_times, n_species).
    derivative_scales : np.ndarray
        Positive scale for each species.

    Returns
    -------
    np.ndarray
        Fitted rates followed by the normalized derivative loss.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_fit_rate_constants(
    design: np.ndarray,
    derivatives: np.ndarray,
    derivative_scales: np.ndarray,
) -> np.ndarray:
    """Reference simultaneous nonnegative least-squares fit."""
    import numpy as np
    from scipy.optimize import lsq_linear

    x = np.asarray(design, dtype=float)
    d = np.asarray(derivatives, dtype=float)
    scales = np.asarray(derivative_scales, dtype=float)

    if d.ndim != 2 or d.shape[0] < 1 or d.shape[1] < 1:
        raise ValueError("derivatives must be a nonempty 2D array")
    if scales.shape != (d.shape[1],) or np.any(scales <= 0.0):
        raise ValueError("derivative_scales must be positive and match species")
    if x.ndim != 2 or x.shape[0] != d.size or x.shape[1] < 1:
        raise ValueError("design must have n_times*n_species rows")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(d)) or not np.all(np.isfinite(scales)):
        raise ValueError("inputs must be finite")

    target = (d / scales[None, :]).reshape(-1)
    fit = lsq_linear(
        x,
        target,
        bounds=(0.0, np.inf),
        tol=1e-12,
        lsmr_tol="auto",
        max_iter=500,
    )
    if not fit.success:
        raise RuntimeError("nonnegative rate fit did not converge")

    residual = (target - x @ fit.x).reshape(d.shape)
    loss = float(np.mean(np.sum(residual * residual, axis=1)))
    return np.concatenate((fit.x.astype(float), np.array([loss], dtype=float)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
derivatives = np.array([[2.0, -1.0], [4.0, -2.0]])
derivative_scales = np.array([4.0, 2.0])
design = np.array([[0.25], [-0.25], [0.50], [-0.50]])""",
            "call": "fit_rate_constants(design, derivatives, derivative_scales)",
            "gold_call": "_oracle_fit_rate_constants(design, derivatives, derivative_scales)",
        },
        {
            "setup": """import numpy as np
derivatives = np.array([[1.0], [-1.0], [2.0]])
derivative_scales = np.array([2.0])
design = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])""",
            "call": "fit_rate_constants(design, derivatives, derivative_scales)",
            "gold_call": "_oracle_fit_rate_constants(design, derivatives, derivative_scales)",
        },
        {
            "setup": """import numpy as np
derivatives = np.zeros((3, 2))
derivative_scales = np.ones(2)
design = np.array([[1.0], [-1.0], [2.0], [-2.0], [0.5], [-0.5]])""",
            "call": "fit_rate_constants(design, derivatives, derivative_scales)",
            "gold_call": "_oracle_fit_rate_constants(design, derivatives, derivative_scales)",
        },
    ]
