"""
Evaluate four archived guiding-center response conventions.

Each row uses the same order and terminal residual observables.

Returns
-------
np.ndarray of shape (4,), float: terminal RSS, component maximum, order RMS, and full RSS.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def guiding_center_response_candidates(order_residuals: np.ndarray, power_error: float, width_error: float) -> np.ndarray:
    """Return the four candidate dimensionless responses.

Parameters
----------
order_residuals : numpy.ndarray
    Nonempty one-dimensional residuals `N_i-N_target`.
power_error, width_error : float
    Finite relative final-versus-launch errors.

Returns
-------
responses : numpy.ndarray
    Float array of shape `(4,)` ordered as `[terminal_RSS,
    componentwise_maximum,order_RMS,full_RSS]`.

Conventions
-----------
All residuals must be finite. Evaluate the order RMS without avoidable overflow from squaring. Zero residuals give zero response. Every returned component must be finite in float64.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return np.zeros(4, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_guiding_center_response_candidates(order_residuals, power_error, width_error):
    residuals = np.asarray(order_residuals, dtype=float)
    power, width = float(power_error), float(width_error)
    if residuals.ndim != 1 or residuals.size == 0 or not np.isfinite(residuals).all() or not np.isfinite([power, width]).all():
        raise ValueError("Require finite nonempty residuals and finite terminal errors")
    scale = float(np.max(np.abs(residuals)))
    order_rms = 0.0 if scale == 0 else scale * np.sqrt(np.mean((residuals / scale)**2))
    with np.errstate(over="ignore", invalid="ignore"):
        terminal = np.hypot(power, width)
        output = np.array([terminal, max(scale, abs(power), abs(width)),
                           order_rms, np.hypot(order_rms, terminal)])
    if not np.isfinite(output).all():
        raise ValueError("Response is not representable as float64")
    return output

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Eight regimes isolate exact, order, terminal, sign, scale, chain, sparse, and mixed behavior."""
    return [{'setup': 'r=np.zeros(4)', 'call': 'guiding_center_response_candidates(r,0.0,0.0)', 'gold_call': '_oracle_guiding_center_response_candidates(r,0.0,0.0)'},
        {'setup': 'r=np.array([1.,-1.,1.,-1.])', 'call': 'guiding_center_response_candidates(r,0.0,0.0)', 'gold_call': '_oracle_guiding_center_response_candidates(r,0.0,0.0)'},
        {'setup': 'r=np.zeros(3)', 'call': 'guiding_center_response_candidates(r,0.3,-0.4)', 'gold_call': '_oracle_guiding_center_response_candidates(r,0.3,-0.4)'},
        {'setup': 'r=np.array([-0.2,0.1,0.4])', 'call': 'guiding_center_response_candidates(r,-0.25,0.15)', 'gold_call': '_oracle_guiding_center_response_candidates(r,-0.25,0.15)'},
        {'setup': 'r=np.array([1e-9,-2e-9])', 'call': 'guiding_center_response_candidates(r,3e-9,-4e-9)', 'gold_call': '_oracle_guiding_center_response_candidates(r,3e-9,-4e-9)'},
        {'setup': 'r=np.linspace(-0.5,0.5,11)', 'call': 'guiding_center_response_candidates(r,0.125,0.25)', 'gold_call': '_oracle_guiding_center_response_candidates(r,0.125,0.25)'},
        {'setup': 'r=np.array([0.03,-0.025,0.02,-0.015])', 'call': 'guiding_center_response_candidates(r,0.012,-0.009)', 'gold_call': '_oracle_guiding_center_response_candidates(r,0.012,-0.009)'},
        {'setup': 'r=np.array([0.,0.,.75,0.,0.,-.25])', 'call': 'guiding_center_response_candidates(r,-.125,.375)', 'gold_call': '_oracle_guiding_center_response_candidates(r,-.125,.375)'},
        {'setup': '', 'call': 'guiding_center_response_candidates(np.array([1e200]),0.,0.)/1e200', 'gold_call': '_oracle_guiding_center_response_candidates(np.array([1e200]),0.,0.)/1e200'},
        {'setup': 'def _raises_value_error(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n    except ValueError:\n        return 1.0\n    return 0.0', 'call': '_raises_value_error(guiding_center_response_candidates,np.array([1e308]),1.7e308,1.7e308)', 'gold_call': '_raises_value_error(_oracle_guiding_center_response_candidates,np.array([1e308]),1.7e308,1.7e308)'}]
