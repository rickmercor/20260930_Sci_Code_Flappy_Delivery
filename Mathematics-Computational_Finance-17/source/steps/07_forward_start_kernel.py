"""
Return the damped Fourier transform, in the log-moneyness at the fixing date, of the exercise-region expectation at one date in the delivery window, seen from time 0 for a contract whose strike is set at the fixing date. The inversion step turns this object back into an expectation, and this is where the transform of the truncated amount, the characteristic function after fixing and the average over the fixing-date variance come together.

The exercise level moves with the variance at date u, and the variance at the fixing date is itself random, so at first sight two further integrals over variance are needed on top of the one in log-spot. Neither has to be done numerically. Freezing the exercise level at a representative variance is the obvious shortcut, and it moves the answer by a large fraction of its size.

Returns
-------
np.ndarray of shape (2*m,) packed as [Re W, Im W], each block of length m.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def forward_start_kernel(omega, alpha, t0, u, a, b, model, term):
    """Damped transform, in the fixing-date log-moneyness, of the forward-start
    exercise-region expectation at date u.

    Write X = log S and Z_t0 = S_t0 * D_d(0, t0) / (S_0 * D_f(0, t0)), the
    same spot weight as in the spot-weighted variance step, and let g_u be
    the amount transformed in the truncated-transform step, evaluated with the
    instantaneous rates at date u.  For a fixing-date log-moneyness x define

        Jhat(x) = E[ Z_t0 * g_u(x + X_u - X_t0)
                     * 1{ x + X_u - X_t0 < a + b*v_u } ],

    the expectation being taken under the pricing measure from the initial
    state at time 0, with v0 from model.  This step returns

        W(omega) = integral over x in R of exp((1j*omega + alpha)*x) * Jhat(x) dx

    at every node of omega.

    Args:
        omega (np.ndarray): frequency nodes, shape (m,), non-empty.
        alpha (float): damping, strictly positive.
        t0 (float): fixing date, non-negative.
        u (float): date in the delivery window, not before t0.
        a (float): intercept of the exercise surface at u.
        b (float): slope of the exercise surface at u.
        model (np.ndarray): shape (8,), packed as [kappa, v0, rd_level,
            rd_amplitude, rd_decay, rf_level, rf_amplitude, rf_decay].
        term (np.ndarray): shape (13,), packed as the four interval edges
            followed by the three volatility-of-variance values, the three
            correlations and the three mean-reversion levels.

    Expected return:
        np.ndarray of shape (2*m,) packed as [Re W, Im W], each block of
        length m.

    Raises:
    ValueError: if omega is empty, if alpha <= 0, if t0 < 0, if u < t0, or if model does not have eight entries or term thirteen.
    """
    m = np.atleast_1d(np.asarray(omega, dtype=float)).reshape(-1).size
    return np.zeros(2 * m)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_forward_start_kernel(omega, alpha, t0, u, a, b, model, term):
    omega = np.atleast_1d(np.asarray(omega, dtype=float)).reshape(-1)
    alpha, t0, u = float(alpha), float(t0), float(u)
    a, b = float(a), float(b)
    model = np.asarray(model, dtype=float).reshape(-1)
    term = np.asarray(term, dtype=float).reshape(-1)
    if omega.size == 0:
        raise ValueError("omega must not be empty")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if t0 < 0.0:
        raise ValueError("t0 must be non-negative")
    if u < t0:
        raise ValueError("u must not precede t0")
    if model.size != 8 or term.size != 13:
        raise ValueError("model must have eight entries and term thirteen")
    m = omega.size
    v0 = float(model[1])
    rd_u = float(model[2] + model[3] * np.exp(-model[4] * u))
    rf_u = float(model[5] + model[6] * np.exp(-model[7] * u))
    g1 = _oracle_truncated_payoff_transform(omega, alpha, a, rd_u, 0.0)
    g2 = _oracle_truncated_payoff_transform(omega, alpha, a, 0.0, rf_u)
    leg1 = g1[:m] + 1j * g1[m:]
    leg2 = g2[:m] + 1j * g2[m:]
    z = 1j * omega + alpha
    u1 = -omega + 1j * alpha
    total = np.zeros(m, dtype=complex)
    for leg, shift in ((leg1, 0.0), (leg2, 1.0)):
        c0 = (shift + z) * b
        e = _oracle_characteristic_exponents(u1.real, u1.imag, c0.real, c0.imag, t0, u, model, term)
        h = _oracle_spot_weighted_variance_exponents(e[2], e[3], t0, model, term)
        total = total + leg * np.exp((e[0] + 1j * e[1]) + (h[0] + 1j * h[1])
                                     + (h[2] + 1j * h[3]) * v0)
    return np.concatenate((total.real, total.imag))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    cfg = ('import numpy as np\n'
           'MODEL = np.array([2.0, 0.09, 0.028, 0.004, 1.5, 0.034, 0.003, 0.8])\n'
           'TERM = np.array([0.0, 0.5, 1.0, 1.5, 0.40, 0.50, 0.45,\n'
           '                 -0.60, -0.40, -0.50, 0.09, 0.10, 0.08])\n'
           'A1, B1 = -0.40343006665908127, -1.1313708498984762\n'
           'def _rule(omax, n):\n'
           '    h = 2.0 * omax / n\n'
           '    om = -omax + h * np.arange(n + 1)\n'
           '    w = np.empty(n + 1)\n'
           '    w[0] = w[-1] = 1.0\n'
           '    w[1:-1:2] = 4.0\n'
           '    w[2:-1:2] = 2.0\n'
           '    return om, w * h / 3.0\n'
           'def _invert(K, om, ow, alpha, x):\n'
           '    m = om.size\n'
           '    W = K[:m] + 1j * K[m:]\n'
           '    return float((np.exp(-alpha * x) / (2.0 * np.pi) * np.sum(ow * np.exp(-1j * om * x) * W)).real)\n'
           'def _rint(level, amp, decay, lo, hi):\n'
           '    return level * (hi - lo) + (amp / decay) * (np.exp(-decay * lo) - np.exp(-decay * hi))\n')

    # pure helper: binds nothing gold-derived, so the setup survives redaction
    _verdict = (cfg
                + 'def verdict(f):\n'
                + '    try:\n'
                + '        f()\n'
                + '        return 0\n'
                + '    except ValueError:\n'
                + '        return 1\n'
                + '    except Exception:\n'
                + '        return 2\n')

    return [
        # interior date with a live slope, fixing across a breakpoint
        {'setup': cfg,
         'call': 'list(np.round(forward_start_kernel(np.array([0.0, 2.5]), 1.0, 0.625, 1.0, A1, B1, MODEL, TERM), 12))',
         'gold_call': 'list(np.round(_oracle_forward_start_kernel(np.array([0.0, 2.5]), 1.0, 0.625, 1.0, A1, B1, MODEL, TERM), 12))',
         'tol': 1e-10},
        # end of the window: the slope has vanished
        {'setup': cfg + 'AT = np.array([-0.20544016792684794, -0.0])\n',
         'call': 'list(np.round(forward_start_kernel(np.array([0.0, 1.0]), 1.0, 0.625, 1.5, float(AT[0]), float(AT[1]), MODEL, TERM), 12))',
         'gold_call': 'list(np.round(_oracle_forward_start_kernel(np.array([0.0, 1.0]), 1.0, 0.625, 1.5, float(AT[0]), float(AT[1]), MODEL, TERM), 12))',
         'tol': 1e-10},
        # a steep slope over a long horizon with an early fixing
        {'setup': cfg,
         'call': 'list(np.round(forward_start_kernel(np.array([0.0, 1.3, 4.0]), 1.0, 0.25, 1.5, -0.3, -3.0, MODEL, TERM), 12))',
         'gold_call': 'list(np.round(_oracle_forward_start_kernel(np.array([0.0, 1.3, 4.0]), 1.0, 0.25, 1.5, -0.3, -3.0, MODEL, TERM), 12))',
         'tol': 1e-10},
        # nothing random at all: fixing and observation both at time 0
        {'setup': (cfg
                   + 'def _case(F):\n'
                   + '    om = np.array([0.0, 0.8, -2.0])\n'
                   + '    k = np.asarray(F(om, 1.4, 0.0, 0.0, -0.4, -1.0, MODEL, TERM), dtype=float)\n'
                   + '    z = 1j * om + 1.4\n'
                   + '    zeta = -0.4 - 1.0 * 0.09\n'
                   + '    ex = 0.032 * np.exp(z * zeta) / z - 0.037 * np.exp((1.0 + z) * zeta) / (1.0 + z)\n'
                   + '    return bool(np.max(np.abs(k[:3] + 1j * k[3:] - ex)) < 1e-14)\n'),
         'call': '_case(forward_start_kernel)',
         'gold_call': '_case(_oracle_forward_start_kernel)'},
        {'setup': (cfg
                   + 'def _case(F):\n'
                   + '    k = F(np.array([1.3, -1.3]), 1.0, 0.625, 1.0, A1, B1, MODEL, TERM)\n'
                   + '    return bool(abs(float(k[0]) - float(k[1])) < 1e-14 and abs(float(k[2]) + float(k[3])) < 1e-14)\n'),
         'call': '_case(forward_start_kernel)',
         'gold_call': '_case(_oracle_forward_start_kernel)'},
        # the damping is a device: the inverted expectation does not see it
        {'setup': (cfg
                   + 'def _case(F):\n'
                   + '    om, ow = _rule(300.0, 8192)\n'
                   + '    vals = [_invert(np.asarray(F(om, al, 0.625, 1.0, A1, B1, MODEL, TERM), dtype=float), om, ow, al, 0.004)\n'
                   + '            for al in (1.0, 2.0)]\n'
                   + '    return bool(abs(vals[0] - vals[1]) < 1e-12)\n'),
         'call': '_case(forward_start_kernel)',
         'gold_call': '_case(_oracle_forward_start_kernel)'},
        # surface pushed out of the way: the untruncated forward-start mean
        {'setup': (cfg
                   + 'def _case(F):\n'
                   + '    om, ow = _rule(300.0, 8192)\n'
                   + '    got = _invert(np.asarray(F(om, 1.0, 0.625, 1.25, 3.0, 0.0, MODEL, TERM), dtype=float), om, ow, 1.0, 0.01)\n'
                   + '    rd = 0.028 + 0.004 * np.exp(-1.5 * 1.25)\n'
                   + '    rf = 0.034 + 0.003 * np.exp(-0.8 * 1.25)\n'
                   + '    carry = _rint(0.028, 0.004, 1.5, 0.625, 1.25) - _rint(0.034, 0.003, 0.8, 0.625, 1.25)\n'
                   + '    ex = rd - rf * np.exp(0.01) * np.exp(carry)\n'
                   + '    return bool(abs(got - ex) < 1e-10)\n'),
         'call': '_case(forward_start_kernel)',
         'gold_call': '_case(_oracle_forward_start_kernel)'},
        # observation date before the fixing date
        {'setup': _verdict,
         'call': 'verdict(lambda: forward_start_kernel(np.array([1.0]), 1.0, 1.0, 0.5, A1, B1, MODEL, TERM))',
         'gold_call': 'verdict(lambda: _oracle_forward_start_kernel(np.array([1.0]), 1.0, 1.0, 0.5, A1, B1, MODEL, TERM))'},
        # non-positive damping
        {'setup': _verdict,
         'call': 'verdict(lambda: forward_start_kernel(np.array([1.0]), -1.0, 0.625, 1.0, A1, B1, MODEL, TERM))',
         'gold_call': 'verdict(lambda: _oracle_forward_start_kernel(np.array([1.0]), -1.0, 0.625, 1.0, A1, B1, MODEL, TERM))'},
    ]
