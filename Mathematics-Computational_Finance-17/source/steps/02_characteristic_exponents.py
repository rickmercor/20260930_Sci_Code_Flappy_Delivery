"""
Return the two exponents of the joint characteristic function of log-spot and variance for one step of the time-inhomogeneous model, starting from a general complex coefficient on the terminal variance. The step may span any of the parameter breakpoints, and the domestic and foreign rates move continuously in calendar time rather than being constant.

The term structure and the rate curves are both indexed by calendar time, while the exponents are most naturally built in time to delivery, so it pays to be deliberate about how the interval pieces are put together. Reproducing the forward price is necessary but not sufficient: implementations that assemble the intervals differently can agree on the forward and still disagree everywhere else.

Returns
-------
np.ndarray of shape (4,) packed as [Re A, Im A, Re B, Im B]. For T == t the result is [0, 0, b0_re, b0_im]. If u1_re, u1_im, b0_re and b0_im are 1-D arrays of length m, the result has shape (4, m), one column per element.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def characteristic_exponents(u1_re, u1_im, b0_re, b0_im, t, T, model, term):
    """Exponents A and B of the joint characteristic function for the step t -> T.

    With x the log-spot and v the variance,

        E[ exp(1j*u1*x_T + b0*v_T) | x_t = x, v_t = v ]
            = exp(1j*u1*x + A + B*v),

    under the dynamics of the problem statement, with the piecewise-constant
    term structure in term and the domestic and foreign rate curves in model.
    Each rate curve has the form r(s) = level + amplitude*exp(-decay*s) in
    calendar time s; a decay of zero means the rate is the constant
    level + amplitude.

    Args:
        u1_re (float): real part of the frequency argument u1.
        u1_im (float): imaginary part of the frequency argument u1.
        b0_re (float): real part of the variance coefficient b0.
        b0_im (float): imaginary part of the variance coefficient b0.
        t (float): calendar start of the step.
        T (float): calendar end of the step, not before t.
        model (np.ndarray): shape (8,), packed as [kappa, v0, rd_level,
            rd_amplitude, rd_decay, rf_level, rf_amplitude, rf_decay].
        term (np.ndarray): shape (13,), packed as the four interval edges
            followed by the three volatility-of-variance values, the three
            correlations and the three mean-reversion levels.

    Expected return:
        np.ndarray of shape (4,) packed as [Re A, Im A, Re B, Im B].  For
        T == t the result is [0, 0, b0_re, b0_im].

    Array inputs:
        u1_re, u1_im, b0_re and b0_im may also be 1-D arrays of one common
        length m (or broadcast against scalars).  A and B are then evaluated
        elementwise and the result has shape (4, m).

    Raises:
    ValueError: if model does not have eight entries, if term does not have thirteen entries, if T < t, if kappa <= 0, if the interval edges are not strictly increasing, or if any volatility of variance is not positive.
    """
    return np.zeros(4)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_characteristic_exponents(u1_re, u1_im, b0_re, b0_im, t, T, model, term):
    model = np.asarray(model, dtype=float).reshape(-1)
    term = np.asarray(term, dtype=float).reshape(-1)
    if model.size != 8:
        raise ValueError("model must have eight entries")
    if term.size != 13:
        raise ValueError("term must have thirteen entries")
    t, T = float(t), float(T)
    if T < t:
        raise ValueError("T must not precede t")
    kappa = float(model[0])
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    edge = term[0:4]
    xis, rhos, thetas = term[4:7], term[7:10], term[10:13]
    if np.any(np.diff(edge) <= 0.0):
        raise ValueError("interval edges must be strictly increasing")
    if np.any(xis <= 0.0):
        raise ValueError("every volatility of variance must be positive")

    def rate_integral(level, amp, decay, lo, hi):
        if abs(decay) < 1.0e-12:
            return (level + amp) * (hi - lo)
        return level * (hi - lo) + (amp / decay) * (np.exp(-decay * lo) - np.exp(-decay * hi))

    u1 = np.asarray(u1_re, dtype=float) + 1j * np.asarray(u1_im, dtype=float)
    B = np.asarray(b0_re, dtype=float) + 1j * np.asarray(b0_im, dtype=float)
    u1, B = np.broadcast_arrays(u1, B)
    B = np.array(B, dtype=complex)
    A = np.zeros(B.shape, dtype=complex)
    segs = []
    for n in range(3):
        lo, hi = max(float(edge[n]), t), min(float(edge[n + 1]), T)
        if hi > lo + 1.0e-14:
            segs.append((lo, hi, float(xis[n]), float(rhos[n]), float(thetas[n])))
    for (lo, hi, xi, rho, theta) in segs[::-1]:
        r = _oracle_riccati_interval_advance(B.real, B.imag, u1.real, u1.imag,
                                             xi, rho, kappa, hi - lo)
        carry = (rate_integral(model[2], model[3], model[4], lo, hi)
                 - rate_integral(model[5], model[6], model[7], lo, hi))
        A = A + 1j * u1 * carry + kappa * theta * (r[2] + 1j * r[3])
        B = r[0] + 1j * r[1]
    return np.array([A.real, A.imag, B.real, B.imag])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    _CFG = (
        'import numpy as np\n'
        'MODEL = np.array([2.0, 0.09, 0.028, 0.004, 1.5, 0.034, 0.003, 0.8])\n'
        'TERM = np.array([0.0, 0.5, 1.0, 1.5, 0.40, 0.50, 0.45,\n'
        '                 -0.60, -0.40, -0.50, 0.09, 0.10, 0.08])\n'
    )

    def _wrap(expr_model, expr_gold):
        return (_CFG
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
        # three-interval chain on the shifted contour with a complex initial condition
        {'setup': _CFG,
         'call': 'list(np.round(characteristic_exponents(-1.5, 1.0, -0.25, 0.4, 0.25, 1.25, MODEL, TERM), 12))',
         'gold_call': 'list(np.round(_oracle_characteristic_exponents(-1.5, 1.0, -0.25, 0.4, 0.25, 1.25, MODEL, TERM), 12))',
         'tol': 1e-10},
        # frequency minus i over the whole horizon: the forward
        {'setup': _CFG,
         'call': 'list(np.round(characteristic_exponents(0.0, -1.0, 0.0, 0.0, 0.0, 1.5, MODEL, TERM), 12))',
         'gold_call': 'list(np.round(_oracle_characteristic_exponents(0.0, -1.0, 0.0, 0.0, 0.0, 1.5, MODEL, TERM), 12))',
         'tol': 1e-12},
        # a step that starts on a breakpoint
        {'setup': _CFG,
         'call': 'list(np.round(characteristic_exponents(-2.0, 1.0, 0.0, 0.0, 0.5, 1.5, MODEL, TERM), 12))',
         'gold_call': 'list(np.round(_oracle_characteristic_exponents(-2.0, 1.0, 0.0, 0.0, 0.5, 1.5, MODEL, TERM), 12))',
         'tol': 1e-10},
        # deep on the contour across all three intervals: sensitive to how the intervals are combined
        {'setup': _CFG,
         'call': 'list(np.round(characteristic_exponents(-120.0, 1.0, -1.0, -135.0, 0.0, 1.5, MODEL, TERM), 9))',
         'gold_call': 'list(np.round(_oracle_characteristic_exponents(-120.0, 1.0, -1.0, -135.0, 0.0, 1.5, MODEL, TERM), 9))',
         'tol': 1e-7},
        # the zero-frequency variance transform with a shifted drift, crossing one breakpoint
        {'setup': _CFG,
         'call': 'list(np.round(characteristic_exponents(0.0, -1.0, -3.0, 5.0, 0.0, 0.625, MODEL, TERM), 12))',
         'gold_call': 'list(np.round(_oracle_characteristic_exponents(0.0, -1.0, -3.0, 5.0, 0.0, 0.625, MODEL, TERM), 12))',
         'tol': 1e-10},
        # constant rates expressed through a zero decay
        {'setup': _CFG + 'M2 = np.array([2.0, 0.09, 0.025, 0.005, 0.0, 0.030, 0.005, 0.0])\n',
         'call': 'list(np.round(characteristic_exponents(-1.5, 1.0, 0.0, 0.0, 0.1, 1.4, M2, TERM), 12))',
         'gold_call': 'list(np.round(_oracle_characteristic_exponents(-1.5, 1.0, 0.0, 0.0, 0.1, 1.4, M2, TERM), 12))',
         'tol': 1e-10},
        {'setup': (_CFG
                   + 'def _case(F):\n'
                   + '    whole = F(-1.5, 1.0, -0.25, 0.4, 0.25, 1.25, MODEL, TERM)\n'
                   + '    late = F(-1.5, 1.0, -0.25, 0.4, 0.75, 1.25, MODEL, TERM)\n'
                   + '    early = F(-1.5, 1.0, float(late[2]), float(late[3]), 0.25, 0.75, MODEL, TERM)\n'
                   + '    ok = (abs(float(early[2]) - float(whole[2])) < 1e-12\n'
                   + '          and abs(float(early[3]) - float(whole[3])) < 1e-12\n'
                   + '          and abs(float(late[0]) + float(early[0]) - float(whole[0])) < 1e-12\n'
                   + '          and abs(float(late[1]) + float(early[1]) - float(whole[1])) < 1e-12)\n'
                   + '    return bool(ok)\n'),
         'call': '_case(characteristic_exponents)',
         'gold_call': '_case(_oracle_characteristic_exponents)'},
        {'setup': (_CFG
                   + 'def _case(F):\n'
                   + '    e = F(0.3, 0.2, -0.7, 0.1, 0.4, 0.4, MODEL, TERM)\n'
                   + '    return list(np.round(np.asarray(e, dtype=float), 12))\n'),
         'call': '_case(characteristic_exponents)',
         'gold_call': '_case(_oracle_characteristic_exponents)',
         'tol': 1e-12},
        {'setup': _CFG + 'OM = np.array([1.5, 10.0, 300.0])\n',
         'call': 'np.round(characteristic_exponents(-OM, np.ones(3), -0.25 * OM, 0.4 * OM, 0.25, 1.25, MODEL, TERM), 12).tolist()',
         'gold_call': 'np.round(_oracle_characteristic_exponents(-OM, np.ones(3), -0.25 * OM, 0.4 * OM, 0.25, 1.25, MODEL, TERM), 12).tolist()',
         'tol': 1e-10},
        {'setup': _wrap('characteristic_exponents(-1.5, 1.0, 0.0, 0.0, 1.25, 0.25, MODEL, TERM)',
                        '_oracle_characteristic_exponents(-1.5, 1.0, 0.0, 0.0, 1.25, 0.25, MODEL, TERM)'),
         'call': 'run_model()',
         'gold_call': 'run_gold()'},
        {'setup': _wrap('characteristic_exponents(-1.5, 1.0, 0.0, 0.0, 0.25, 1.25, MODEL[:4], TERM)',
                        '_oracle_characteristic_exponents(-1.5, 1.0, 0.0, 0.0, 0.25, 1.25, MODEL[:4], TERM)'),
         'call': 'run_model()',
         'gold_call': 'run_gold()'},
    ]
