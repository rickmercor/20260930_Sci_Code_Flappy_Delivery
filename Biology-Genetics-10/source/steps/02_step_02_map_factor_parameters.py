"""
Map working-scale probit coefficients to a unit-variance factor model.

For working coefficient a_j and fixed threshold tau_j, the one-factor model

uses psi_j = 1 / (1 + a_j^2), b_j = a_j sqrt(psi_j), and

beta_j = -tau_j / sqrt(psi_j). This absorbs b_j^2 + psi_j = 1 exactly.

The factor reflection is fixed by orienting the lead loading b_0 to be positive,

and every uniqueness must remain at or above the prescribed floor psi_min.

Inputs

------

working_loadings : one-dimensional working-scale coefficients

thresholds : fixed latent Gaussian thresholds

psi_min : minimum admissible uniqueness

Returns

-------

loadings : oriented latent factor loadings

uniqueness : residual variances

intercepts : working-scale probit intercepts

Returns
-------
tuple of three float64 arrays of shape (p,): loadings, uniqueness, and intercepts
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def map_factor_parameters(
    working_loadings: np.ndarray, thresholds: np.ndarray, psi_min: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Map working coefficients to oriented unit-variance factor parameters.

    Parameters
    ----------
    working_loadings : np.ndarray
        One-dimensional fitted working-scale coefficients, lead first.
    thresholds : np.ndarray
        Fixed Gaussian thresholds in the same order.
    psi_min : float
        Strictly positive lower bound on every uniqueness.

    Returns
    -------
    loadings : np.ndarray
        Oriented one-factor loadings.
    uniqueness : np.ndarray
        Residual variances satisfying loadings squared plus uniqueness equals one.
    intercepts : np.ndarray
        Working-scale probit intercepts.

    Raises
    ------
    ValueError
        If `working_loadings` is not a finite one-dimensional vector with at
        least two entries; if `thresholds` is not finite with the same shape;
        if `psi_min` is not numeric or does not lie in `(0, 1)`; if the lead
        working loading is zero; or if any implied uniqueness is more than
        `1e-12` below `psi_min`.
    '''
    return result  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_map_factor_parameters(
    working_loadings: np.ndarray, thresholds: np.ndarray, psi_min: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation of the constrained parameter map."""
    coefficients = np.asarray(working_loadings, dtype=float)
    tau = np.asarray(thresholds, dtype=float)
    if coefficients.ndim != 1 or coefficients.size < 2 or not np.isfinite(coefficients).all():
        raise ValueError("working_loadings must be a finite vector with at least two entries")
    if tau.shape != coefficients.shape or not np.isfinite(tau).all():
        raise ValueError("thresholds must be finite and match working_loadings")
    if not isinstance(psi_min, (int, float, np.integer, np.floating)):
        raise ValueError("psi_min must be numeric")
    if not 0.0 < float(psi_min) < 1.0:
        raise ValueError("psi_min must lie in (0, 1)")
    if coefficients[0] == 0.0:
        raise ValueError("the lead working loading must be nonzero")

    uniqueness = 1.0 / (1.0 + coefficients * coefficients)
    if np.any(uniqueness < float(psi_min) - 1e-12):
        raise ValueError("working_loadings violate the uniqueness floor")
    loadings = coefficients * np.sqrt(uniqueness)
    if loadings[0] < 0.0:
        loadings = -loadings
        coefficients = -coefficients
    intercepts = -tau / np.sqrt(uniqueness)
    return loadings, uniqueness, intercepts

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811

_THRESHOLDS = np.array(
    [1.000490545619381, 0.7332357810248108, -0.5813930077385457,
     0.5104216363344174, -0.7332357810248108, 1.2278262639421003,
     -0.18445243845167293, 0.31060942561200097]
)


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """working_loadings = np.array([2.2, 1.7, -1.9, 1.25, -1.55, 0.85, -1.15, 1.45])
thresholds = _THRESHOLDS.copy()
psi_min = 0.15
def pack(result):
    return np.stack(result).astype(float, copy=False)
""",
            "call": "pack(map_factor_parameters(working_loadings, thresholds, psi_min))",
            "gold_call": "pack(_oracle_map_factor_parameters(working_loadings, thresholds, psi_min))",
        },
        {
            "setup": """psi_min = 0.2
working_loadings = np.array([-2.0, 0.0])
thresholds = np.array([0.0, 0.5])
def pack(result):
    return np.stack(result).astype(float, copy=False)
""",
            "call": "pack(map_factor_parameters(working_loadings, thresholds, psi_min))",
            "gold_call": "pack(_oracle_map_factor_parameters(working_loadings, thresholds, psi_min))",
        },
        {
            "setup": """working_loadings = np.array([3.0, 0.5])
thresholds = np.array([0.0, 0.0])
psi_min = 0.2
def run_model():
    try:
        map_factor_parameters(working_loadings, thresholds, psi_min)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_map_factor_parameters(working_loadings, thresholds, psi_min)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
