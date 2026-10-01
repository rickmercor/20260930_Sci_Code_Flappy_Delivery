"""
Advance the variance exponent of the joint characteristic function of log-spot and variance across one interval on which the volatility of variance and the correlation are constant, and return both the value the exponent reaches and its integral over the interval. Every exponent used later in the pipeline is assembled from calls to this step, so an error here propagates everywhere downstream.

The exponent obeys a scalar quadratic ordinary differential equation in time to delivery, and its coefficients are fixed by the dynamics in the problem statement together with the frequency argument. That argument is complex because the evaluation runs on a contour shifted off the real axis, so the discriminant of the quadratic is complex as well, and the initial condition is a general complex number rather than a multiple of the imaginary unit.

Accuracy matters more than usual. The integral feeds an exponent that is chained across several intervals and then exponentiated, and the final answer is graded to a millionth of a pip, so an approximation that looks harmless on one interval is not harmless after the chain.

Returns
-------
np.ndarray of shape (4,) packed as [Re B(s), Im B(s), Re integral, Im integral], where integral is the integral of B over [0, s]. For s = 0 the result is [b0_re, b0_im, 0, 0]. If b0_re, b0_im, u1_re and u1_im are 1-D arrays of length m, the result has shape (4, m), one column per element.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def riccati_interval_advance(b0_re, b0_im, u1_re, u1_im, xi, rho, kappa, s):
    """The variance exponent B of the joint characteristic function obeys a scalar
    quadratic ordinary differential equation in time-to-delivery.  Its three
    coefficients follow from the dynamics given in the problem statement and
    from the frequency argument u1, and on this interval the
    volatility-of-volatility and the correlation are constant, so the
    coefficients are constant too.  Advance that equation from B(0) = b0 across
    a span of length s.  Both B(s) and the integral of B over [0, s] are
    returned, the second because the accumulated log-spot exponent consumes it.

    The frequency argument is complex, not real: the evaluation runs on a
    contour shifted off the real axis, so an implementation checked only at real
    u1 has not been checked.

    Args:
        b0_re (float): real part of the initial condition B(0).
        b0_im (float): imaginary part of the initial condition B(0).
        u1_re (float): real part of the frequency argument u1.
        u1_im (float): imaginary part of the frequency argument u1.
        xi (float): volatility of volatility on the interval, strictly positive.
        rho (float): correlation on the interval, in [-1, 1].
        kappa (float): mean-reversion speed, strictly positive.
        s (float): interval length, non-negative.

    Expected return:
        np.ndarray of shape (4,) packed as [Re B(s), Im B(s), Re integral,
        Im integral], where integral is the integral of B over [0, s].  For
        s = 0 the result is [b0_re, b0_im, 0, 0].

    Array inputs:
        b0_re, b0_im, u1_re and u1_im may also be 1-D arrays of one common
        length m (or broadcast against scalars).  The four quantities are
        then evaluated elementwise and the result has shape (4, m).

    Raises:
    ValueError: if xi <= 0, if rho lies outside [-1, 1], if kappa <= 0, or if s < 0.
    """
    return np.zeros(4)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_riccati_interval_advance(b0_re, b0_im, u1_re, u1_im, xi, rho, kappa, s):
    xi, rho, kappa, s = float(xi), float(rho), float(kappa), float(s)
    if xi <= 0.0:
        raise ValueError("xi must be positive")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must lie in [-1, 1]")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if s < 0.0:
        raise ValueError("s must be non-negative")
    b0 = np.asarray(b0_re, dtype=float) + 1j * np.asarray(b0_im, dtype=float)
    u1 = np.asarray(u1_re, dtype=float) + 1j * np.asarray(u1_im, dtype=float)
    b0, u1 = np.broadcast_arrays(b0, u1)
    if s == 0.0:
        zero = np.zeros(b0.shape)
        return np.array([b0.real, b0.imag, zero, zero])
    aa = 0.5 * xi * xi
    bb = 1j * u1 * rho * xi - kappa
    cc = -0.5 * (u1 * u1 + 1j * u1)
    d = np.sqrt(bb * bb - 4.0 * aa * cc + 0j)
    lam1 = (-bb + d) / (xi * xi)
    lam2 = (-bb - d) / (xi * xi)
    R = (b0 - lam2) / (b0 - lam1)
    e = np.exp(-d * s)
    B = (lam2 - lam1 * R * e) / (1.0 - R * e)
    I = lam2 * s - (2.0 / (xi * xi)) * np.log((1.0 - R * e) / (1.0 - R))
    return np.array([B.real, B.imag, I.real, I.imag])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {'setup': 'import numpy as np\n',
         'call': 'list(np.round(riccati_interval_advance(0.0, 0.0, -1.5, 1.0, 0.45, -0.50, 2.0, 0.5), 12))',
         'gold_call': 'list(np.round(_oracle_riccati_interval_advance(0.0, 0.0, -1.5, 1.0, 0.45, -0.50, 2.0, 0.5), 12))',
         'tol': 1e-10},
        {'setup': 'import numpy as np\n',
         'call': 'list(np.round(riccati_interval_advance(-0.3, 0.7, -4.0, 1.0, 0.40, -0.60, 2.0, 0.5), 12))',
         'gold_call': 'list(np.round(_oracle_riccati_interval_advance(-0.3, 0.7, -4.0, 1.0, 0.40, -0.60, 2.0, 0.5), 12))',
         'tol': 1e-10},
        {'setup': 'import numpy as np\n',
         'call': 'list(np.round(riccati_interval_advance(0.0, 0.0, 0.0, 0.0, 0.50, -0.40, 2.0, 0.25), 12))',
         'gold_call': 'list(np.round(_oracle_riccati_interval_advance(0.0, 0.0, 0.0, 0.0, 0.50, -0.40, 2.0, 0.25), 12))',
         'tol': 1e-12},
        {'setup': 'import numpy as np\n',
         'call': 'list(np.round(riccati_interval_advance(0.2, -0.1, 0.0, 0.0, 0.50, -0.40, 2.0, 0.0), 12))',
         'gold_call': 'list(np.round(_oracle_riccati_interval_advance(0.2, -0.1, 0.0, 0.0, 0.50, -0.40, 2.0, 0.0), 12))',
         'tol': 1e-12},
        {'setup': (
            'import numpy as np\n'
            'def _case(F):\n'
            '    r = F(0.0, 0.0, -1.5, 1.0, 0.45, -0.50, 2.0, 0.5)\n'
            '    half = F(0.0, 0.0, -1.5, 1.0, 0.45, -0.50, 2.0, 0.25)\n'
            '    chain = F(float(half[0]), float(half[1]), -1.5, 1.0, 0.45, -0.50, 2.0, 0.25)\n'
            '    ok = (abs(float(chain[0]) - float(r[0])) < 1e-12\n'
            '          and abs(float(chain[1]) - float(r[1])) < 1e-12\n'
            '          and abs(float(half[2]) + float(chain[2]) - float(r[2])) < 1e-12\n'
            '          and abs(float(half[3]) + float(chain[3]) - float(r[3])) < 1e-12)\n'
            '    return bool(ok)\n'),
         'call': '_case(riccati_interval_advance)',
         'gold_call': '_case(_oracle_riccati_interval_advance)'},
        # the spot-weighted variance transform uses a zero log-spot frequency with a shifted drift
        {'setup': 'import numpy as np\n',
         'call': 'list(np.round(riccati_interval_advance(-2.5, 7.0, 0.0, -1.0, 0.40, -0.60, 2.0, 0.5), 12))',
         'gold_call': 'list(np.round(_oracle_riccati_interval_advance(-2.5, 7.0, 0.0, -1.0, 0.40, -0.60, 2.0, 0.5), 12))',
         'tol': 1e-10},
        # deep on the shifted contour with a large complex initial condition
        {'setup': 'import numpy as np\n',
         'call': 'list(np.round(riccati_interval_advance(-1.0, -135.0, -120.0, 1.0, 0.45, -0.50, 2.0, 0.5), 10))',
         'gold_call': 'list(np.round(_oracle_riccati_interval_advance(-1.0, -135.0, -120.0, 1.0, 0.45, -0.50, 2.0, 0.5), 10))',
         'tol': 1e-8},
        # a second interval's parameters and a longer span
        {'setup': 'import numpy as np\n',
         'call': 'list(np.round(riccati_interval_advance(-0.6, 3.1, -40.0, 1.0, 0.50, -0.40, 2.0, 0.375), 11))',
         'gold_call': 'list(np.round(_oracle_riccati_interval_advance(-0.6, 3.1, -40.0, 1.0, 0.50, -0.40, 2.0, 0.375), 11))',
         'tol': 1e-9},
        {'setup': (
            'import numpy as np\n'
            'def _case(F):\n'
            '    vals = [F(0.0, 0.0, -w, 1.0, 0.45, -0.50, 2.0, 0.5) for w in (1.0, 10.0, 100.0, 300.0)]\n'
            '    return [np.round(np.asarray(v, dtype=float), 8).tolist() for v in vals]\n'),
         'call': '_case(riccati_interval_advance)',
         'gold_call': '_case(_oracle_riccati_interval_advance)',
         'tol': 1e-7},
        {'setup': (
            'import numpy as np\n'
            'def run_model():\n'
            '    try:\n'
            '        riccati_interval_advance(0.0, 0.0, -1.5, 1.0, 0.0, -0.50, 2.0, 0.5)\n'
            '        return 0\n'
            '    except ValueError:\n'
            '        return 1\n'
            '    except Exception:\n'
            '        return 2\n'
            'def run_gold():\n'
            '    try:\n'
            '        _oracle_riccati_interval_advance(0.0, 0.0, -1.5, 1.0, 0.0, -0.50, 2.0, 0.5)\n'
            '        return 0\n'
            '    except ValueError:\n'
            '        return 1\n'
            '    except Exception:\n'
            '        return 2\n'),
         'call': 'run_model()',
         'gold_call': 'run_gold()'},
        {'setup': (
            'import numpy as np\n'
            'def run_model():\n'
            '    try:\n'
            '        riccati_interval_advance(0.0, 0.0, -1.5, 1.0, 0.45, -1.40, 2.0, 0.5)\n'
            '        return 0\n'
            '    except ValueError:\n'
            '        return 1\n'
            '    except Exception:\n'
            '        return 2\n'
            'def run_gold():\n'
            '    try:\n'
            '        _oracle_riccati_interval_advance(0.0, 0.0, -1.5, 1.0, 0.45, -1.40, 2.0, 0.5)\n'
            '        return 0\n'
            '    except ValueError:\n'
            '        return 1\n'
            '    except Exception:\n'
            '        return 2\n'),
         'call': 'run_model()',
         'gold_call': 'run_gold()'},
        {'setup': 'import numpy as np\nOM = np.array([1.5, 10.0, 300.0])\n',
         'call': 'np.round(riccati_interval_advance(np.zeros(3), np.zeros(3), -OM, np.ones(3), 0.45, -0.50, 2.0, 0.5), 12).tolist()',
         'gold_call': 'np.round(_oracle_riccati_interval_advance(np.zeros(3), np.zeros(3), -OM, np.ones(3), 0.45, -0.50, 2.0, 0.5), 12).tolist()',
         'tol': 1e-10},
    ]
