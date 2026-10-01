"""
Evaluate four archived guiding-center launch-state conventions.

Rows share one loss, span, and soliton power scale; the integrated loss exponent is dimensionless.

Returns
-------
np.ndarray of shape (4,4), float: integrated loss exponent alpha_db_per_km*span_km*ln(10)/10 (dimensionless), gain, launch power (mW), and target order.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def guiding_center_theory_candidates(alpha_db_per_km: float, span_km: float, p0_mw: float) -> np.ndarray:
    """Return four candidate guiding-center launch states.

Parameters
----------
alpha_db_per_km : float
    Nonnegative power attenuation in dB/km.
span_km : float
    Nonnegative amplifier spacing in km.
p0_mw : float
    Positive first-order-soliton peak power in mW.

Returns
-------
states : numpy.ndarray
    Float array of shape `(4,4)`. Rows follow the declared candidate order;
    columns are `[integrated_loss_exponent,gain,launch_power_mW,target_order]`, where
    `integrated_loss_exponent = alpha_db_per_km*span_km*ln(10)/10` is dimensionless.

Conventions
-----------
All inputs must be finite. Loss and span are nonnegative; p0_mw is positive. At zero integrated loss, every multiplier, gain and order equals one. All gains, launch powers and target orders must be finite and strictly positive in float64.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return np.zeros((4, 4), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_guiding_center_theory_candidates(alpha_db_per_km, span_km, p0_mw):
    loss, span, power = map(float, (alpha_db_per_km, span_km, p0_mw))
    if not np.isfinite([loss, span, power]).all() or loss < 0 or span < 0 or power <= 0:
        raise ValueError("Require finite nonnegative loss/span and positive power")
    wide = np.longdouble
    exponent = wide(loss) * wide(span) * np.log(wide(10)) / 10
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gain = np.exp(exponent)
        if exponent == 0:
            multipliers = np.ones(4, dtype=wide)
        else:
            decrement = -np.expm1(-exponent)
            increment = np.expm1(exponent)
            multipliers = np.array([decrement / exponent, gain * increment / exponent,
                                    exponent / decrement, exponent / increment], dtype=wide)
        output = np.asarray(np.column_stack((np.full(4, exponent, dtype=wide),
                            np.full(4, gain, dtype=wide), wide(power) * multipliers,
                            np.sqrt(multipliers))), dtype=float)
    if not np.isfinite(output).all() or np.any(output[:, 1:] <= 0):
        raise ValueError("Positive launch states are not representable as float64")
    return output

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Eight regimes cover zero loss, removable limits, scaling, and candidate separation."""
    return [{'setup': '', 'call': 'guiding_center_theory_candidates(0.19,17.0,4.2)', 'gold_call': '_oracle_guiding_center_theory_candidates(0.19,17.0,4.2)'},
        {'setup': '', 'call': 'guiding_center_theory_candidates(0.0,23.0,5.0)', 'gold_call': '_oracle_guiding_center_theory_candidates(0.0,23.0,5.0)'},
        {'setup': '', 'call': 'guiding_center_theory_candidates(1e-12,1e-3,2.0)', 'gold_call': '_oracle_guiding_center_theory_candidates(1e-12,1e-3,2.0)'},
        {'setup': '', 'call': 'guiding_center_theory_candidates(0.4,10.0,25.0)', 'gold_call': '_oracle_guiding_center_theory_candidates(0.4,10.0,25.0)'},
        {'setup': '', 'call': 'guiding_center_theory_candidates(0.31,0.0,6.4)', 'gold_call': '_oracle_guiding_center_theory_candidates(0.31,0.0,6.4)'},
        {'setup': '', 'call': 'guiding_center_theory_candidates(1.2,50.0,0.5)', 'gold_call': '_oracle_guiding_center_theory_candidates(1.2,50.0,0.5)'},
        {'setup': '', 'call': 'guiding_center_theory_candidates(0.05,250.0,100.0)', 'gold_call': '_oracle_guiding_center_theory_candidates(0.05,250.0,100.0)'},
        {'setup': '', 'call': 'guiding_center_theory_candidates(0.175,31.75,3.2)', 'gold_call': '_oracle_guiding_center_theory_candidates(0.175,31.75,3.2)'},
        {'setup': 'def _raises_value_error(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n    except ValueError:\n        return 1.0\n    return 0.0', 'call': '_raises_value_error(guiding_center_theory_candidates,.2611,20.,1e308)', 'gold_call': '_raises_value_error(_oracle_guiding_center_theory_candidates,.2611,20.,1e308)'}]
