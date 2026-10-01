"""
Return the exponents of the transform of the variance at a fixing date, taken in the measure that weights every path by the spot at that date, discounted on the foreign curve and normalised so the weights average to one. A contract whose strike is set at the fixing date in proportion to the spot values everything after that date per unit of the fixing-date spot, and this step supplies the average over the fixing-date variance that such a valuation needs.

The weight is correlated with the variance path, so the average is not the one the pricing measure gives, and the difference is not the same on every interval of the term structure. The spot itself does not survive into the result: once the weight has been applied, what is left describes the variance at the fixing date alone. For real q, Ahat + Bhat*v0 is the log-moment generating function of the fixing-date variance under the weighting, which is a convenient check.

Returns
-------
np.ndarray of shape (4,) packed as [Re Ahat, Im Ahat, Re Bhat, Im Bhat]. For t0 = 0 the result is [0, 0, q_re, q_im]. If q_re and q_im are 1-D arrays of length m, the result has shape (4, m), one column per element.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def spot_weighted_variance_exponents(q_re, q_im, t0, model, term):
    """Exponents of the spot-weighted transform of the variance at t0.

    With S the spot, D_d(0, t) and D_f(0, t) the domestic and foreign
    discount factors built from the rate curves in model, and

        Z_t = S_t * D_d(0, t) / (S_0 * D_f(0, t)),

    this step returns Ahat and Bhat such that

        E[ Z_t0 * exp(q * v_t0) ] = exp(Ahat + Bhat * v0),

    the expectation being taken under the pricing measure of the problem
    statement from the initial state at time 0.  Each rate curve has the form
    r(s) = level + amplitude*exp(-decay*s).

    Args:
        q_re (float): real part of the variance coefficient q.
        q_im (float): imaginary part of the variance coefficient q.
        t0 (float): the date at which the variance is observed, non-negative.
        model (np.ndarray): shape (8,), packed as [kappa, v0, rd_level,
            rd_amplitude, rd_decay, rf_level, rf_amplitude, rf_decay].
        term (np.ndarray): shape (13,), packed as the four interval edges
            followed by the three volatility-of-variance values, the three
            correlations and the three mean-reversion levels.

    Expected return:
        np.ndarray of shape (4,) packed as [Re Ahat, Im Ahat, Re Bhat,
        Im Bhat].  For t0 = 0 the result is [0, 0, q_re, q_im].

    Array inputs:
        q_re and q_im may also be 1-D arrays of one common length m (or
        broadcast against a scalar).  The exponents are then evaluated
        elementwise and the result has shape (4, m).

    Raises:
    ValueError: if model does not have eight entries, if term does not have thirteen entries, if t0 < 0, if kappa <= 0, if the interval edges are not strictly increasing, or if any volatility of variance is not positive.
    """
    return np.zeros(4)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_spot_weighted_variance_exponents(q_re, q_im, t0, model, term):
    model = np.asarray(model, dtype=float).reshape(-1)
    term = np.asarray(term, dtype=float).reshape(-1)
    if model.size != 8:
        raise ValueError("model must have eight entries")
    if term.size != 13:
        raise ValueError("term must have thirteen entries")
    t0 = float(t0)
    if t0 < 0.0:
        raise ValueError("t0 must be non-negative")

    def rate_integral(level, amp, decay, lo, hi):
        if abs(decay) < 1.0e-12:
            return (level + amp) * (hi - lo)
        return level * (hi - lo) + (amp / decay) * (np.exp(-decay * lo) - np.exp(-decay * hi))

    q = np.asarray(q_re, dtype=float) + 1j * np.asarray(q_im, dtype=float)
    q = np.atleast_1d(q) if np.ndim(q) else q
    shape = np.shape(q)
    e = _oracle_characteristic_exponents(np.zeros(shape), -np.ones(shape), np.real(q), np.imag(q),
                                         0.0, t0, model, term)
    carry = (rate_integral(model[2], model[3], model[4], 0.0, t0)
             - rate_integral(model[5], model[6], model[7], 0.0, t0))
    return np.array([e[0] - carry, e[1], e[2], e[3]])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    cfg = ('import numpy as np\n'
           'MODEL = np.array([2.0, 0.09, 0.028, 0.004, 1.5, 0.034, 0.003, 0.8])\n'
           'TERM = np.array([0.0, 0.5, 1.0, 1.5, 0.40, 0.50, 0.45,\n'
           '                 -0.60, -0.40, -0.50, 0.09, 0.10, 0.08])\n')

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
        # a fixing date across the first breakpoint, complex coefficient
        {'setup': cfg,
         'call': 'list(np.round(spot_weighted_variance_exponents(-1.2, 0.9, 0.625, MODEL, TERM), 12))',
         'gold_call': 'list(np.round(_oracle_spot_weighted_variance_exponents(-1.2, 0.9, 0.625, MODEL, TERM), 12))',
         'tol': 1e-10},
        # a fixing date inside the first interval, real coefficient
        {'setup': cfg,
         'call': 'list(np.round(spot_weighted_variance_exponents(0.8, 0.0, 0.3, MODEL, TERM), 12))',
         'gold_call': 'list(np.round(_oracle_spot_weighted_variance_exponents(0.8, 0.0, 0.3, MODEL, TERM), 12))',
         'tol': 1e-10},
        # large coefficient of the kind the pipeline produces far out on the contour, over two breakpoints
        {'setup': cfg,
         'call': 'list(np.round(spot_weighted_variance_exponents(-60.0, 250.0, 1.2, MODEL, TERM), 10))',
         'gold_call': 'list(np.round(_oracle_spot_weighted_variance_exponents(-60.0, 250.0, 1.2, MODEL, TERM), 10))',
         'tol': 1e-8},
        # positive correlation: the weighting slows mean reversion instead of speeding it up
        {'setup': cfg + 'T2 = np.array([0.0, 0.3, 0.9, 1.5, 0.60, 0.70, 0.55, 0.50, 0.65, 0.30, 0.05, 0.12, 0.07])\n',
         'call': 'list(np.round(spot_weighted_variance_exponents(np.array([-0.5, 0.4]), np.array([2.0, -0.3]), 1.1, MODEL, T2), 12).ravel())',
         'gold_call': 'list(np.round(_oracle_spot_weighted_variance_exponents(np.array([-0.5, 0.4]), np.array([2.0, -0.3]), 1.1, MODEL, T2), 12).ravel())',
         'tol': 1e-10},
        {'setup': cfg,
         'call': 'list(np.round(spot_weighted_variance_exponents(0.4, -0.2, 0.0, MODEL, TERM), 12))',
         'gold_call': 'list(np.round(_oracle_spot_weighted_variance_exponents(0.4, -0.2, 0.0, MODEL, TERM), 12))',
         'tol': 1e-12},
        # the zero coefficient: the weights average to one
        {'setup': cfg,
         'call': 'list(np.round(spot_weighted_variance_exponents(0.0, 0.0, 0.9, MODEL, TERM), 12))',
         'gold_call': 'list(np.round(_oracle_spot_weighted_variance_exponents(0.0, 0.0, 0.9, MODEL, TERM), 12))',
         'tol': 1e-12},
        # the spot-weighted mean of the fixing-date variance, by a central difference in q
        {'setup': (cfg
                   + 'def _case(F):\n'
                   + '    h = 1e-6\n'
                   + '    up = F(h, 0.0, 0.625, MODEL, TERM)\n'
                   + '    dn = F(-h, 0.0, 0.625, MODEL, TERM)\n'
                   + '    mean = (float(up[0]) + float(up[2]) * 0.09 - float(dn[0]) - float(dn[2]) * 0.09) / (2 * h)\n'
                   + '    return round(mean, 7)\n'),
         'call': '_case(spot_weighted_variance_exponents)',
         'gold_call': '_case(_oracle_spot_weighted_variance_exponents)',
         'tol': 1e-6},
        {'setup': _wrap('spot_weighted_variance_exponents(0.1, 0.0, -0.1, MODEL, TERM)',
                        '_oracle_spot_weighted_variance_exponents(0.1, 0.0, -0.1, MODEL, TERM)'),
         'call': 'run_model()', 'gold_call': 'run_gold()'},
        {'setup': _wrap('spot_weighted_variance_exponents(0.1, 0.0, 0.5, MODEL, TERM[:10])',
                        '_oracle_spot_weighted_variance_exponents(0.1, 0.0, 0.5, MODEL, TERM[:10])'),
         'call': 'run_model()', 'gold_call': 'run_gold()'},
    ]
