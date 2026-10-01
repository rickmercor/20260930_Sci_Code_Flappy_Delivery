"""
Evaluate first-order intensity-sech-squared peak-power normalization.

The normalization fixes the physical power scale of the guiding-center archive.

Returns
-------
float: first-order-soliton peak power in mW.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def sech_soliton_peak_power_mw(beta2_fs2_per_m: float, gamma_w_inv_km: float, fwhm_ps: float) -> float:
    """Return the first-order `sech` soliton peak power in mW.

Parameters
----------
beta2_fs2_per_m : float
    Group-velocity dispersion in `fs^2/m`; its magnitude is used.
gamma_w_inv_km : float
    Positive nonlinear coefficient in `W^-1 km^-1`.
fwhm_ps : float
    Positive intensity FWHM in ps.

Returns
-------
power_mw : float
    First-order-soliton peak power in mW.

Conventions
-----------
All inputs must be finite; gamma and FWHM must be positive. Zero dispersion returns exactly zero. Otherwise the returned power must be finite and strictly positive in float64.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_sech_soliton_peak_power_mw(beta2_fs2_per_m, gamma_w_inv_km, fwhm_ps):
    dispersion, gamma, width = map(float, (beta2_fs2_per_m, gamma_w_inv_km, fwhm_ps))
    if not np.isfinite([dispersion, gamma, width]).all() or gamma <= 0 or width <= 0:
        raise ValueError("Require finite dispersion and positive gamma and FWHM")
    wide = np.longdouble
    factor = 2 * np.arccosh(np.sqrt(wide(2)))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        power = float(abs(wide(dispersion)) * factor**2 / wide(gamma) / wide(width)**2)
    if not np.isfinite(power) or (dispersion != 0 and power <= 0):
        raise ValueError("Positive peak power is not representable as float64")
    return power

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Eight regimes cover units, sign invariance, and independent physical scalings."""
    return [{'setup': '', 'call': 'sech_soliton_peak_power_mw(-6800.0, 1.45, 42.0)', 'gold_call': '_oracle_sech_soliton_peak_power_mw(-6800.0, 1.45, 42.0)'},
        {'setup': '', 'call': 'sech_soliton_peak_power_mw(13600.0, 1.45, 42.0)', 'gold_call': '_oracle_sech_soliton_peak_power_mw(13600.0, 1.45, 42.0)'},
        {'setup': '', 'call': 'sech_soliton_peak_power_mw(-6800.0, 1.45, 21.0)', 'gold_call': '_oracle_sech_soliton_peak_power_mw(-6800.0, 1.45, 21.0)'},
        {'setup': '', 'call': 'sech_soliton_peak_power_mw(6800.0, 2.9, 42.0)', 'gold_call': '_oracle_sech_soliton_peak_power_mw(6800.0, 2.9, 42.0)'},
        {'setup': '', 'call': 'sech_soliton_peak_power_mw(-5000.0, 2.0, 1.0)', 'gold_call': '_oracle_sech_soliton_peak_power_mw(-5000.0, 2.0, 1.0)'},
        {'setup': '', 'call': 'sech_soliton_peak_power_mw(-1.0, 1e-3, 1000.0)', 'gold_call': '_oracle_sech_soliton_peak_power_mw(-1.0, 1e-3, 1000.0)'},
        {'setup': '', 'call': 'sech_soliton_peak_power_mw(-9000.0, 0.9, 37.5)', 'gold_call': '_oracle_sech_soliton_peak_power_mw(-9000.0, 0.9, 37.5)'},
        {'setup': '', 'call': 'sech_soliton_peak_power_mw(4200.0, 3.7, 12.25)', 'gold_call': '_oracle_sech_soliton_peak_power_mw(4200.0, 3.7, 12.25)'},
        {'setup': 'def _raises_value_error(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n    except ValueError:\n        return 1.0\n    return 0.0', 'call': '_raises_value_error(sech_soliton_peak_power_mw,-7650.,1.3,1e-300)', 'gold_call': '_raises_value_error(_oracle_sech_soliton_peak_power_mw,-7650.,1.3,1e-300)'},
        {'setup': '', 'call': 'sech_soliton_peak_power_mw(0.,1.3,1e-300)', 'gold_call': '_oracle_sech_soliton_peak_power_mw(0.,1.3,1e-300)'}]
