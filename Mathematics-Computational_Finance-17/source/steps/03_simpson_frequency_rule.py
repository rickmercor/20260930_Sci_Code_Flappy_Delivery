"""
Build the quadrature rule used for every frequency integral in the pipeline: a composite Simpson rule on a symmetric interval about the origin, returned as nodes and weights together because every later step consumes them as a pair.

Symmetry about the origin is not cosmetic. The integrands this rule is applied to are complex and are not even functions of the frequency, so the negative half-line carries information of its own and the familiar shortcut of integrating the positive half and doubling does not apply.

The rule needs an even number of sub-intervals, and an odd count is rejected rather than silently padded. The weights carry the step length, so they sum to the width of the interval and a constant is integrated exactly, which is the cheapest check on an implementation.

Returns
-------
np.ndarray of shape (2*(n+1),) packed as [nodes, weights], nodes in increasing order from -omega_max to +omega_max and weights already carrying the step length, so that they sum to 2*omega_max.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def simpson_frequency_rule(omega_max, n):
    """Composite Simpson rule on [-omega_max, omega_max].

    Args:
        omega_max (float): half-width of the frequency interval, strictly
            positive.
        n (int): number of equal sub-intervals, an even integer of at least two.

    Expected return:
        np.ndarray of shape (2*(n+1),) packed as [nodes, weights], nodes in
        increasing order from -omega_max to +omega_max and weights already
        carrying the step length, so that they sum to 2*omega_max.

    Raises:
    ValueError: if omega_max <= 0, if n < 2, or if n is odd.
    """
    return np.zeros(2 * (int(n) + 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_simpson_frequency_rule(omega_max, n):
    omega_max = float(omega_max)
    n = int(n)
    if omega_max <= 0.0:
        raise ValueError("omega_max must be positive")
    if n < 2 or n % 2 != 0:
        raise ValueError("n must be an even integer of at least two")
    h = 2.0 * omega_max / n
    om = -omega_max + h * np.arange(n + 1)
    w = np.empty(n + 1)
    w[0] = 1.0
    w[-1] = 1.0
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return np.concatenate((om, w * h / 3.0))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [{'setup': 'import numpy as np\n'
               'def _case(F):\n'
               '    r = F(1.0, 4)\n'
               '    return list(np.round(np.asarray(r, dtype=float), 15))\n',
      'call': '_case(simpson_frequency_rule)',
      'gold_call': '_case(_oracle_simpson_frequency_rule)',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n'
               'def _case(F):\n'
               '    r = F(300.0, 8192)\n'
               '    h = r.size // 2\n'
               '    return list(np.round(np.array([float(r.size), float(r[0]), float(r[1]), '
               'float(r[h]), float(r[h + 1]), float(r[h + 2]), float(r[h:].sum())]), 12))\n',
      'call': '_case(simpson_frequency_rule)',
      'gold_call': '_case(_oracle_simpson_frequency_rule)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'def _case(F):\n'
               '    r = F(12.5, 200)\n'
               '    h = r.size // 2\n'
               '    om, w = r[:h], r[h:]\n'
               '    cubic = float(np.sum(w * (om ** 3 - 3.0 * om ** 2 + 1.0)))\n'
               '    return bool(abs(cubic + 3881.25) < 1e-8)\n',
      'call': '_case(simpson_frequency_rule)',
      'gold_call': '_case(_oracle_simpson_frequency_rule)'},
     {'setup': 'import numpy as np\n'
               'def _case(F):\n'
               '    r = F(7.0, 50)\n'
               '    h = r.size // 2\n'
               '    om, w = r[:h], r[h:]\n'
               '    ok = bool(np.all(np.diff(om) > 0.0) and np.allclose(om, -om[::-1], atol=1e-13) '
               'and np.allclose(w, w[::-1], atol=1e-15))\n'
               '    return bool(ok)\n',
      'call': '_case(simpson_frequency_rule)',
      'gold_call': '_case(_oracle_simpson_frequency_rule)'},
     {'setup': 'import numpy as np\n'
               'def _case(F):\n'
               '    r = F(3.0, 2)\n'
               '    h = r.size // 2\n'
               '    return list(np.round(np.asarray(r, dtype=float), 15))\n',
      'call': '_case(simpson_frequency_rule)',
      'gold_call': '_case(_oracle_simpson_frequency_rule)',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n'
               'def run_model():\n'
               '    try:\n'
               '        simpson_frequency_rule(300.0, 8191)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_simpson_frequency_rule(300.0, 8191)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'def run_model():\n'
               '    try:\n'
               '        simpson_frequency_rule(-1.0, 8)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_simpson_frequency_rule(-1.0, 8)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
