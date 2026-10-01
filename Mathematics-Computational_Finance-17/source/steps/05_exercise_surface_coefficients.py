"""
Return the intercept and the slope, in the variance, of the supplied exercise-surface iterate at one date in the delivery window. The pipeline only ever needs the surface at one date at a time, and because the surface is affine in the variance those two numbers describe it completely.

The iterate is anchored at the terminal value of the exercise boundary. That level is fixed by the contract and the carry rather than by the numerics, and with rates that move in calendar time it is worth being careful about which rates determine it. The square-root factor takes both coefficients to that terminal value, so at the end of the window the variance drops out of the exercise rule.

Returns
-------
np.ndarray of shape (2,) packed as [intercept, slope], so that the surface at variance v is intercept + slope*v. At u = T2 the slope vanishes and the intercept is the terminal boundary.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def exercise_surface_coefficients(u, T2, A0, B0, model):
    """Intercept and slope of the affine exercise-surface iterate at time u.

    The iterate is

        xstar(u, v) = xstar_terminal - (A0 + B0*v) * sqrt(T2 - u),

    in log-moneyness, where xstar_terminal is the terminal value of the
    exercise boundary of the contract in the problem statement.  That level
    is a property of the payoff and the rate curves, not a configuration
    constant, and it is not zero.

    Args:
        u (float): time in the delivery window, not after T2.
        T2 (float): far end of the delivery window, strictly positive.
        A0 (float): intercept shape constant.
        B0 (float): slope shape constant.
        model (np.ndarray): shape (8,), packed as [kappa, v0, rd_level,
            rd_amplitude, rd_decay, rf_level, rf_amplitude, rf_decay], each
            rate curve being r(s) = level + amplitude*exp(-decay*s).

    Expected return:
        np.ndarray of shape (2,) packed as [intercept, slope], so that the
        surface at variance v is intercept + slope*v.  At u = T2 the slope
        vanishes and the intercept is the terminal boundary.

    Raises:
    ValueError: if model does not have eight entries, if T2 <= 0, if u > T2, or if either rate at T2 is not strictly positive.
    """
    return np.zeros(2)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_exercise_surface_coefficients(u, T2, A0, B0, model):
    model = np.asarray(model, dtype=float).reshape(-1)
    if model.size != 8:
        raise ValueError("model must have eight entries")
    u, T2, A0, B0 = float(u), float(T2), float(A0), float(B0)
    if T2 <= 0.0:
        raise ValueError("T2 must be positive")
    if u > T2:
        raise ValueError("u must not exceed T2")
    rd_T2 = float(model[2] + model[3] * np.exp(-model[4] * T2))
    rf_T2 = float(model[5] + model[6] * np.exp(-model[7] * T2))
    if rd_T2 <= 0.0 or rf_T2 <= 0.0:
        raise ValueError("both rates at T2 must be positive")
    root = np.sqrt(T2 - u)
    return np.array([np.log(rd_T2 / rf_T2) - A0 * root, -B0 * root])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    cfg = ('import numpy as np\n'
           'MODEL = np.array([2.0, 0.09, 0.028, 0.004, 1.5, 0.034, 0.003, 0.8])\n')

    def _wrap(expr_model, expr_gold):
        return (cfg
                + 'def run_model():\n'
                + '    try:\n'
                + '        ' + expr_model + '\n'
                + '        return 0\n'
                + '    except ValueError:\n'
                + '        return 1\n'
                + '    except Exception:\n'
                + '        return 2\n'
                + 'def run_gold():\n'
                + '    try:\n'
                + '        ' + expr_gold + '\n'
                + '        return 0\n'
                + '    except ValueError:\n'
                + '        return 1\n'
                + '    except Exception:\n'
                + '        return 2\n')

    return [
        {'setup': cfg,
         'call': 'list(np.round(exercise_surface_coefficients(0.75, 1.5, 0.28, 1.60, MODEL), 15))',
         'gold_call': 'list(np.round(_oracle_exercise_surface_coefficients(0.75, 1.5, 0.28, 1.60, MODEL), 15))',
         'tol': 1e-12},
        {'setup': cfg,
         'call': 'list(np.round(exercise_surface_coefficients(1.0, 1.5, 0.28, 1.60, MODEL), 15))',
         'gold_call': 'list(np.round(_oracle_exercise_surface_coefficients(1.0, 1.5, 0.28, 1.60, MODEL), 15))',
         'tol': 1e-12},
        {'setup': cfg,
         'call': 'list(np.round(exercise_surface_coefficients(1.5, 1.5, 0.28, 1.60, MODEL), 15))',
         'gold_call': 'list(np.round(_oracle_exercise_surface_coefficients(1.5, 1.5, 0.28, 1.60, MODEL), 15))',
         'tol': 1e-12},
        # the domestic rate above the foreign one, with a growing foreign curve
        {'setup': cfg + 'M2 = np.array([2.0, 0.09, 0.050, -0.010, 0.7, 0.020, -0.005, -0.4])\n',
         'call': 'list(np.round(exercise_surface_coefficients(0.2, 2.0, 0.15, 0.90, M2), 15))',
         'gold_call': 'list(np.round(_oracle_exercise_surface_coefficients(0.2, 2.0, 0.15, 0.90, M2), 15))',
         'tol': 1e-12},
        # constant curves through a zero decay
        {'setup': cfg + 'M3 = np.array([2.0, 0.09, 0.025, 0.005, 0.0, 0.030, 0.005, 0.0])\n',
         'call': 'list(np.round(exercise_surface_coefficients(0.5, 1.5, 0.28, 1.60, M3), 15))',
         'gold_call': 'list(np.round(_oracle_exercise_surface_coefficients(0.5, 1.5, 0.28, 1.60, M3), 15))',
         'tol': 1e-12},
        {'setup': (cfg
                   + 'def _case(F):\n'
                   + '    us = [0.75, 0.9, 1.0, 1.25, 1.4, 1.5]\n'
                   + '    cs = [F(u, 1.5, 0.28, 1.60, MODEL) for u in us]\n'
                   + '    ints = [float(c[0]) for c in cs]\n'
                   + '    slopes = [float(c[1]) for c in cs]\n'
                   + '    return bool(all(ints[i] < ints[i + 1] for i in range(5))\n'
                   + '                and all(slopes[i] < slopes[i + 1] for i in range(5)))\n'),
         'call': '_case(exercise_surface_coefficients)',
         'gold_call': '_case(_oracle_exercise_surface_coefficients)'},
        {'setup': _wrap('exercise_surface_coefficients(1.6, 1.5, 0.28, 1.60, MODEL)',
                        '_oracle_exercise_surface_coefficients(1.6, 1.5, 0.28, 1.60, MODEL)'),
         'call': 'run_model()', 'gold_call': 'run_gold()'},
        {'setup': _wrap('exercise_surface_coefficients(1.0, 1.5, 0.28, 1.60, np.array([2.0, 0.09, 0.028, 0.004, 1.5, -0.034, 0.0, 0.8]))',
                        '_oracle_exercise_surface_coefficients(1.0, 1.5, 0.28, 1.60, np.array([2.0, 0.09, 0.028, 0.004, 1.5, -0.034, 0.0, 0.8]))'),
         'call': 'run_model()', 'gold_call': 'run_gold()'},
    ]
