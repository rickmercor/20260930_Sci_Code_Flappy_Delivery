"""
Chain the earlier steps into one evaluation of the forward-starting flexible forward on the supplied exercise-surface iterate, and return its value at time 0 in domestic pips.

The strike is not known today. It is set at the fixing date from the forward then prevailing, so every cash flow after that date is proportional to a spot that is itself random, and today's value has to account for that before anything is discounted. As in any early-exercise decomposition, the value splits into what delivery at the far end of the window is worth and a premium collected over the window.

The frequency rule is converged at the stated settings and the time rule is not, so the value is pinned by the stated number of time sub-intervals. The answer is graded to a millionth of a pip.

Returns
-------
float: the contract value at time 0 in domestic pips, ten thousand times the value per unit of foreign notional.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def forward_start_flexible_forward_value(model, term, contract, numerics):
    """Value in domestic pips of the forward-starting flexible forward.

    The contract is on one unit of foreign currency with S_0 = 1.  At the
    fixing date t0 its delivery rate K is set equal to the outright forward
    rate then prevailing for delivery at T2.  The holder then takes delivery
    once, at a single date in [T1, T2] of their choosing, and receives
    K - S at that date in domestic currency.  The exercise surface is the
    supplied iterate in log-moneyness log(S/K) with shape constants A0 and
    B0; the frequency integral uses the composite Simpson rule on
    [-omega_max, omega_max] with n_omega sub-intervals and damping alpha;
    the premium integral over the window uses the composite trapezoidal rule
    with n_step sub-intervals.

    Args:
        model (np.ndarray): shape (8,), packed as [kappa, v0, rd_level,
            rd_amplitude, rd_decay, rf_level, rf_amplitude, rf_decay], each
            rate curve being r(s) = level + amplitude*exp(-decay*s).
        term (np.ndarray): shape (13,), packed as the four interval edges
            followed by the three volatility-of-variance values, the three
            correlations and the three mean-reversion levels.
        contract (np.ndarray): shape (5,), packed as [t0, T1, T2, A0, B0].
        numerics (np.ndarray): shape (4,), packed as
            [alpha, omega_max, n_omega, n_step]; the last two are read as
            integers.

    Expected return:
        float: the contract value at time 0 in domestic pips, ten thousand
        times the value per unit of foreign notional.

    Raises:
    ValueError: if model does not have eight entries, if term does not have thirteen, if contract does not have five, if numerics does not have four, if t0 < 0, if T1 < t0, or if T2 <= T1.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_forward_start_flexible_forward_value(model, term, contract, numerics):
    model = np.asarray(model, dtype=float).reshape(-1)
    term = np.asarray(term, dtype=float).reshape(-1)
    contract = np.asarray(contract, dtype=float).reshape(-1)
    numerics = np.asarray(numerics, dtype=float).reshape(-1)
    if model.size != 8 or term.size != 13:
        raise ValueError("model must have eight entries and term thirteen")
    if contract.size != 5:
        raise ValueError("contract must have five entries")
    if numerics.size != 4:
        raise ValueError("numerics must have four entries")
    t0, T1, T2, A0, B0 = [float(c) for c in contract]
    if t0 < 0.0:
        raise ValueError("t0 must be non-negative")
    if T1 < t0:
        raise ValueError("the window must not open before t0")
    if T2 <= T1:
        raise ValueError("T2 must exceed T1")
    alpha, omega_max = float(numerics[0]), float(numerics[1])
    n_omega, nstep = int(numerics[2]), int(numerics[3])

    def rate_integral(level, amp, decay, lo, hi):
        if abs(decay) < 1.0e-12:
            return (level + amp) * (hi - lo)
        return level * (hi - lo) + (amp / decay) * (np.exp(-decay * lo) - np.exp(-decay * hi))

    rd_int = lambda lo, hi: rate_integral(model[2], model[3], model[4], lo, hi)
    rf_int = lambda lo, hi: rate_integral(model[5], model[6], model[7], lo, hi)
    carry = rd_int(t0, T2) - rf_int(t0, T2)
    ratio = np.exp(carry)
    x_fix = -carry
    rule = _oracle_simpson_frequency_rule(omega_max, n_omega)
    om = rule[:rule.size // 2]
    tr = _oracle_premium_time_rule(T1, T2, nstep, t0, model)
    half = tr.size // 2
    nodes, weights = tr[:half], tr[half:]
    prem = 0.0
    for i in range(half):
        ab = _oracle_exercise_surface_coefficients(float(nodes[i]), T2, A0, B0, model)
        ker = _oracle_forward_start_kernel(om, alpha, t0, float(nodes[i]),
                                           float(ab[0]), float(ab[1]), model, term)
        prem += weights[i] * _oracle_exercise_region_expectation(ker, rule, alpha, x_fix)
    fixing_scale = ratio * np.exp(-rf_int(0.0, t0))
    terminal = fixing_scale * np.exp(-rd_int(t0, T2)) - np.exp(-rf_int(0.0, T2))
    return float(1.0e4 * (terminal + fixing_scale * prem))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    cfg = ('import numpy as np\n'
           'MODEL = np.array([2.0, 0.09, 0.028, 0.004, 1.5, 0.034, 0.003, 0.8])\n'
           'TERM = np.array([0.0, 0.5, 1.0, 1.5, 0.40, 0.50, 0.45,\n'
           '                 -0.60, -0.40, -0.50, 0.09, 0.10, 0.08])\n'
           'CONTRACT = np.array([0.625, 0.75, 1.5, 0.28, 1.60])\n'
           'NUMERICS = np.array([1.0, 300.0, 8192.0, 16.0])\n')

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
         'call': 'round(float(forward_start_flexible_forward_value(MODEL, TERM, CONTRACT, NUMERICS)), 9)',
         'gold_call': 'round(float(_oracle_forward_start_flexible_forward_value(MODEL, TERM, CONTRACT, NUMERICS)), 9)',
         'tol': 1e-7},
        # a coarser pinned time rule and a different damping
        {'setup': cfg + 'N2 = np.array([1.7, 300.0, 8192.0, 8.0])\n',
         'call': 'round(float(forward_start_flexible_forward_value(MODEL, TERM, CONTRACT, N2)), 9)',
         'gold_call': 'round(float(_oracle_forward_start_flexible_forward_value(MODEL, TERM, CONTRACT, N2)), 9)',
         'tol': 1e-7},
        # the domestic rate above the foreign one, a flat surface and an early fixing
        {'setup': cfg + ('M2 = np.array([2.0, 0.09, 0.045, -0.006, 0.9, 0.021, 0.004, 1.2])\n'
                         'C2 = np.array([0.2, 0.5, 1.25, 0.10, 0.0])\n'
                         'N3 = np.array([1.0, 300.0, 8192.0, 6.0])\n'),
         'call': 'round(float(forward_start_flexible_forward_value(M2, TERM, C2, N3)), 9)',
         'gold_call': 'round(float(_oracle_forward_start_flexible_forward_value(M2, TERM, C2, N3)), 9)',
         'tol': 1e-7},
        # a different term structure, fixing inside the first interval
        {'setup': cfg + ('TERM2 = np.array([0.0, 0.4, 0.9, 1.5, 0.30, 0.60, 0.35,\n'
                         '                  -0.30, -0.70, -0.20, 0.12, 0.07, 0.10])\n'
                         'C3 = np.array([0.3, 0.6, 1.5, 0.20, 1.10])\n'
                         'N4 = np.array([1.0, 300.0, 8192.0, 10.0])\n'),
         'call': 'round(float(forward_start_flexible_forward_value(MODEL, TERM2, C3, N4)), 9)',
         'gold_call': 'round(float(_oracle_forward_start_flexible_forward_value(MODEL, TERM2, C3, N4)), 9)',
         'tol': 1e-7},
        # fixing today: the contract reduces to a spot-start one struck at today's forward
        {'setup': cfg + 'C4 = np.array([0.0, 0.75, 1.5, 0.28, 1.60])\nN5 = np.array([1.0, 300.0, 8192.0, 8.0])\n',
         'call': 'round(float(forward_start_flexible_forward_value(MODEL, TERM, C4, N5)), 9)',
         'gold_call': 'round(float(_oracle_forward_start_flexible_forward_value(MODEL, TERM, C4, N5)), 9)',
         'tol': 1e-7},
        {'setup': _wrap('forward_start_flexible_forward_value(MODEL, TERM, np.zeros(3), NUMERICS)',
                        '_oracle_forward_start_flexible_forward_value(MODEL, TERM, np.zeros(3), NUMERICS)'),
         'call': 'run_model()', 'gold_call': 'run_gold()'},
        {'setup': _wrap('forward_start_flexible_forward_value(MODEL, TERM, np.array([0.8, 0.75, 1.5, 0.28, 1.60]), NUMERICS)',
                        '_oracle_forward_start_flexible_forward_value(MODEL, TERM, np.array([0.8, 0.75, 1.5, 0.28, 1.60]), NUMERICS)'),
         'call': 'run_model()', 'gold_call': 'run_gold()'},
        {'setup': _wrap('forward_start_flexible_forward_value(MODEL[:4], TERM, CONTRACT, NUMERICS)',
                        '_oracle_forward_start_flexible_forward_value(MODEL[:4], TERM, CONTRACT, NUMERICS)'),
         'call': 'run_model()', 'gold_call': 'run_gold()'},
    ]
