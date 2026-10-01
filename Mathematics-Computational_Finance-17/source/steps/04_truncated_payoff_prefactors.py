"""
Return the damped Fourier transform of the amount the early-exercise decomposition collects on the exercise region, cut off sharply at a given exercise level. The amount is what accrues to the holder of a contract paying K − S at delivery, per unit strike and per unit time, while the state sits where immediate delivery is optimal.

A sharp cut-off is exactly what makes a series representation of the same density ring, and damping the integrand before transforming it is what removes the difficulty. With a strictly positive damping the integral converges at minus infinity and the transform is an elementary expression rather than an approximation; a damping that is zero or negative leaves a divergent integral, which is why it is validated rather than merely documented.

The amount itself is not the intrinsic value of the contract and not a single rate times the intrinsic value. It is whatever the decomposition of this particular payoff under this particular carry produces, and getting it wrong moves the answer by a large fraction of its size.

Returns
-------
np.ndarray of shape (2*m,) packed as [Re G, Im G], each block of length m.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def truncated_payoff_transform(omega, alpha, xstar, rd, rf):
    """Damped Fourier transform of the truncated early-exercise amount.

    Let g(x) be the amount the early-exercise decomposition collects, per
    unit strike and per unit time, while the state sits in the exercise
    region of a contract that pays K - S at delivery.  It is written in
    log-moneyness x = log(S/K) and uses the instantaneous domestic rate rd
    and foreign rate rf of the date in question.  This step returns

        G(omega) = integral over x from -inf to xstar of
                   g(x) * exp((1j*omega + alpha)*x) dx

    at every node of omega.

    Args:
        omega (np.ndarray): frequency nodes, shape (m,), non-empty.
        alpha (float): damping, strictly positive.
        xstar (float): cut-off, the exercise level in log-moneyness.
        rd (float): instantaneous domestic rate.
        rf (float): instantaneous foreign rate.

    Expected return:
        np.ndarray of shape (2*m,) packed as [Re G, Im G], each block of
        length m.

    Raises:
    ValueError: if omega is empty or if alpha <= 0.
    """
    m = np.atleast_1d(np.asarray(omega, dtype=float)).reshape(-1).size
    return np.zeros(2 * m)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_truncated_payoff_transform(omega, alpha, xstar, rd, rf):
    omega = np.atleast_1d(np.asarray(omega, dtype=float)).reshape(-1)
    alpha, xstar = float(alpha), float(xstar)
    rd, rf = float(rd), float(rf)
    if omega.size == 0:
        raise ValueError("omega must not be empty")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    z = 1j * omega + alpha
    g = rd * np.exp(z * xstar) / z - rf * np.exp((1.0 + z) * xstar) / (1.0 + z)
    return np.concatenate((g.real, g.imag))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    def _wrap(expr_model, expr_gold):
        return ('import numpy as np\n'
                'def run_model():\n'
                '    try:\n'
                '        ' + expr_model + '\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                'def run_gold():\n'
                '    try:\n'
                '        ' + expr_gold + '\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n')

    return [
        {'setup': 'import numpy as np\n',
         'call': 'list(np.round(truncated_payoff_transform(np.array([2.5]), 1.0, -0.4, 0.030, 0.035), 12))',
         'gold_call': 'list(np.round(_oracle_truncated_payoff_transform(np.array([2.5]), 1.0, -0.4, 0.030, 0.035), 12))',
         'tol': 1e-10},
        {'setup': 'import numpy as np\n',
         'call': 'list(np.round(truncated_payoff_transform(np.array([0.0, 1.0, -1.0, 7.5]), 1.0, -0.2, 0.0284, 0.0349), 12))',
         'gold_call': 'list(np.round(_oracle_truncated_payoff_transform(np.array([0.0, 1.0, -1.0, 7.5]), 1.0, -0.2, 0.0284, 0.0349), 12))',
         'tol': 1e-10},
        # a positive cut-off and a large damping
        {'setup': 'import numpy as np\n',
         'call': 'list(np.round(truncated_payoff_transform(np.array([-3.0, 0.5]), 2.7, 0.15, 0.041, 0.022), 12))',
         'gold_call': 'list(np.round(_oracle_truncated_payoff_transform(np.array([-3.0, 0.5]), 2.7, 0.15, 0.041, 0.022), 12))',
         'tol': 1e-10},
        # the closed form against a fine direct quadrature of the defining integral
        {'setup': ('import numpy as np\n'
                   'def _case(F):\n'
                   '    om = np.array([0.0, 1.7, -4.2])\n'
                   '    got = np.asarray(F(om, 1.3, -0.35, 0.030, 0.035), dtype=float)\n'
                   '    x = np.linspace(-60.0, -0.35, 600001)\n'
                   '    h = x[1] - x[0]\n'
                   '    w = np.full(x.size, h)\n'
                   '    w[0] *= 0.5\n'
                   '    w[-1] *= 0.5\n'
                   '    ok = True\n'
                   '    for k, o in enumerate(om):\n'
                   '        f = (0.030 - 0.035 * np.exp(x)) * np.exp((1j * o + 1.3) * x)\n'
                   '        ref = np.sum(w * f)\n'
                   '        ok = ok and abs(got[k] - ref.real) < 1e-8 and abs(got[k + 3] - ref.imag) < 1e-8\n'
                   '    return bool(ok)\n'),
         'call': '_case(truncated_payoff_transform)',
         'gold_call': '_case(_oracle_truncated_payoff_transform)'},
        {'setup': ('import numpy as np\n'
                   'def _case(F):\n'
                   '    p = F(np.array([0.7, -0.7]), 0.9, -0.25, 0.030, 0.035)\n'
                   '    return bool(abs(float(p[0]) - float(p[1])) < 1e-15 and abs(float(p[2]) + float(p[3])) < 1e-15)\n'),
         'call': '_case(truncated_payoff_transform)',
         'gold_call': '_case(_oracle_truncated_payoff_transform)'},
        {'setup': ('import numpy as np\n'
                   'def _case(F):\n'
                   '    return np.round(np.asarray(F(np.linspace(-4.0, 4.0, 9), 1.0, -0.4, 0.030, 0.035), dtype=float), 12).tolist()\n'),
         'call': '_case(truncated_payoff_transform)',
         'gold_call': '_case(_oracle_truncated_payoff_transform)',
         'tol': 1e-10},
        {'setup': _wrap('truncated_payoff_transform(np.array([1.0]), 0.0, -0.4, 0.030, 0.035)',
                        '_oracle_truncated_payoff_transform(np.array([1.0]), 0.0, -0.4, 0.030, 0.035)'),
         'call': 'run_model()', 'gold_call': 'run_gold()'},
        {'setup': _wrap('truncated_payoff_transform(np.array([]), 1.0, -0.4, 0.030, 0.035)',
                        '_oracle_truncated_payoff_transform(np.array([]), 1.0, -0.4, 0.030, 0.035)'),
         'call': 'run_model()', 'gold_call': 'run_gold()'},
    ]
