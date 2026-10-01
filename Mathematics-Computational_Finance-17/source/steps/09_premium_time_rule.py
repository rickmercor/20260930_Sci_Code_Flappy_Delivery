"""
Build the time rule for the early-exercise premium: the dates at which the exercise-region expectation is evaluated and the weights that combine them, with the domestic discounting from the fixing date already folded into the weights.

Delivery cannot be taken before the window opens, so the nodes span the window and nothing outside it. The window is split into equal sub-intervals and the composite trapezoidal rule is applied, so the two end weights are halved. Folding the discounting into the weights makes the premium a single dot product against the vector of expectations.

The rule is pinned rather than converged: halving or doubling the number of sub-intervals moves the final value by far more than the grading tolerance, so the count in the problem statement has to be honoured exactly.

Returns
-------
np.ndarray of shape (2*(nstep+1),) packed as [nodes, weights], nodes increasing from T1 to T2 and weights being the composite trapezoidal weights multiplied by the domestic discount factor from t0 to each node.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def premium_time_rule(T1, T2, nstep, t0, model):
    """Discounted composite trapezoidal rule over the delivery window.

    The nodes are nstep + 1 equally spaced dates from T1 to T2.  Each weight
    is the composite trapezoidal weight of its node multiplied by the
    domestic discount factor from the fixing date t0 to that node, built from
    the domestic rate curve r_d(s) = level + amplitude*exp(-decay*s) in model.

    Args:
        T1 (float): near end of the delivery window, not before t0.
        T2 (float): far end of the delivery window, strictly after T1.
        nstep (int): number of equal sub-intervals, at least one.
        t0 (float): fixing date, non-negative.
        model (np.ndarray): shape (8,), packed as [kappa, v0, rd_level,
            rd_amplitude, rd_decay, rf_level, rf_amplitude, rf_decay].

    Expected return:
        np.ndarray of shape (2*(nstep+1),) packed as [nodes, weights], nodes
        increasing from T1 to T2.

    Raises:
    ValueError: if model does not have eight entries, if T2 <= T1, if nstep < 1, if t0 < 0, or if T1 < t0.
    """
    return np.zeros(2 * (int(nstep) + 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_premium_time_rule(T1, T2, nstep, t0, model):
    model = np.asarray(model, dtype=float).reshape(-1)
    if model.size != 8:
        raise ValueError("model must have eight entries")
    T1, T2, t0 = float(T1), float(T2), float(t0)
    nstep = int(nstep)
    if T2 <= T1:
        raise ValueError("T2 must exceed T1")
    if nstep < 1:
        raise ValueError("nstep must be at least one")
    if t0 < 0.0:
        raise ValueError("t0 must be non-negative")
    if T1 < t0:
        raise ValueError("the window must not open before t0")
    level, amp, decay = float(model[2]), float(model[3]), float(model[4])
    h = (T2 - T1) / nstep
    u = T1 + h * np.arange(nstep + 1)
    if abs(decay) < 1.0e-12:
        integ = (level + amp) * (u - t0)
    else:
        integ = level * (u - t0) + (amp / decay) * (np.exp(-decay * t0) - np.exp(-decay * u))
    w = np.full(nstep + 1, h)
    w[0] *= 0.5
    w[-1] *= 0.5
    return np.concatenate((u, w * np.exp(-integ)))

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
        {'setup': cfg + 'def _case(F):\n    r = F(0.75, 1.5, 16, 0.625, MODEL)\n    h = r.size // 2\n    return list(np.round(np.array([float(r.size), float(r[0]), float(r[1]), float(r[h]), float(r[h + 1]), float(r[-1]), float(r[h:].sum())]), 12))\n',
         'call': '_case(premium_time_rule)',
         'gold_call': '_case(_oracle_premium_time_rule)',
         'tol': 1e-10},
        {'setup': cfg,
         'call': 'list(np.round(premium_time_rule(0.5, 1.0, 2, 0.1, MODEL), 13))',
         'gold_call': 'list(np.round(_oracle_premium_time_rule(0.5, 1.0, 2, 0.1, MODEL), 13))',
         'tol': 1e-11},
        # a growing domestic curve and a fixing date at the window's opening
        {'setup': cfg + 'M2 = np.array([2.0, 0.09, 0.020, 0.010, -1.0, 0.034, 0.003, 0.8])\n',
         'call': 'list(np.round(premium_time_rule(0.4, 1.2, 4, 0.4, M2), 13))',
         'gold_call': 'list(np.round(_oracle_premium_time_rule(0.4, 1.2, 4, 0.4, M2), 13))',
         'tol': 1e-11},
        # a constant domestic rate through a zero decay
        {'setup': cfg + 'M3 = np.array([2.0, 0.09, 0.025, 0.005, 0.0, 0.034, 0.003, 0.8])\n',
         'call': 'list(np.round(premium_time_rule(0.75, 1.5, 3, 0.25, M3), 13))',
         'gold_call': 'list(np.round(_oracle_premium_time_rule(0.75, 1.5, 3, 0.25, M3), 13))',
         'tol': 1e-11},
        {'setup': cfg + 'def _case(F):\n    r = F(0.75, 1.5, 40, 0.625, np.array([2.0, 0.09, 0.0, 0.0, 0.0, 0.034, 0.003, 0.8]))\n    h = r.size // 2\n    return bool(abs(float(r[h:].sum()) - 0.75) < 1e-14 and abs(float(r[h]) - 0.5 * float(r[h + 1])) < 1e-15)\n',
         'call': '_case(premium_time_rule)',
         'gold_call': '_case(_oracle_premium_time_rule)'},
        {'setup': _wrap('premium_time_rule(1.5, 0.75, 16, 0.625, MODEL)',
                        '_oracle_premium_time_rule(1.5, 0.75, 16, 0.625, MODEL)'),
         'call': 'run_model()', 'gold_call': 'run_gold()'},
        {'setup': _wrap('premium_time_rule(0.5, 1.5, 16, 0.625, MODEL)',
                        '_oracle_premium_time_rule(0.5, 1.5, 16, 0.625, MODEL)'),
         'call': 'run_model()', 'gold_call': 'run_gold()'},
        {'setup': _wrap('premium_time_rule(0.75, 1.5, 0, 0.625, MODEL)',
                        '_oracle_premium_time_rule(0.75, 1.5, 0, 0.625, MODEL)'),
         'call': 'run_model()', 'gold_call': 'run_gold()'},
    ]
