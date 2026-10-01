"""
Turn a damped transform back into the expectation it encodes, at one value of the log-moneyness, by summing the kernel against the frequency rule. The result is real, because the function being recovered is real.

The sum is not real term by term: the imaginary parts cancel only across the whole symmetric grid, so taking the real part at the end is correct and taking the modulus is not. The damping that made the forward transform converge has to be removed again at the point where the function is recovered.

Returns
-------
float: the recovered value f(x_src), a real number.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def exercise_region_expectation(kernel_flat, rule_flat, alpha, x_src):
    """Recover an expectation from its damped transform at one log-moneyness.

    kernel_flat holds samples, at the nodes of rule_flat, of a damped transform

        W(omega) = integral over x in R of exp((1j*omega + alpha)*x) * f(x) dx

    of a real function f.  Using the quadrature rule for the frequency
    integral, return the value f(x_src).

    Args:
        kernel_flat (np.ndarray): shape (2*m,), packed as [Re W, Im W].
        rule_flat (np.ndarray): shape (2*m,), packed as [nodes, weights].
        alpha (float): damping, strictly positive, the same one used to build
            the kernel.
        x_src (float): log-moneyness at which f is recovered.

    Expected return:
        float: the recovered value f(x_src), a real number.

    Raises:
    ValueError: if either packed array has odd length, if the two imply a different number of nodes, or if alpha <= 0.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_exercise_region_expectation(kernel_flat, rule_flat, alpha, x_src):
    kernel_flat = np.asarray(kernel_flat, dtype=float).reshape(-1)
    rule_flat = np.asarray(rule_flat, dtype=float).reshape(-1)
    alpha, x_src = float(alpha), float(x_src)
    if kernel_flat.size % 2 != 0 or rule_flat.size % 2 != 0:
        raise ValueError("both packed arrays must have even length")
    m = kernel_flat.size // 2
    k = rule_flat.size // 2
    if m != k:
        raise ValueError("kernel and rule must hold the same number of nodes")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    w = kernel_flat[:m] + 1j * kernel_flat[m:]
    om, ow = rule_flat[:k], rule_flat[k:]
    val = np.exp(-alpha * x_src) / (2.0 * np.pi) * np.sum(
        ow * np.exp(-1j * om * x_src) * w)
    return float(val.real)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [{'setup': 'import numpy as np\n'
               'import math\n'
               'RD, RF = 0.030, 0.035\n'
               'def _rule(omax, n):\n'
               '    h = 2.0 * omax / n\n'
               '    om = -omax + h * np.arange(n + 1)\n'
               '    w = np.empty(n + 1)\n'
               '    w[0] = w[-1] = 1.0\n'
               '    w[1:-1:2] = 4.0\n'
               '    w[2:-1:2] = 2.0\n'
               '    return om, w * h / 3.0\n'
               'OM, OW = _rule(300.0, 8192)\n'
               'RULE = np.concatenate((OM, OW))\n'
               'def _pref(om, alpha, zeta):\n'
               '    z = 1j * om + alpha\n'
               '    return RD * np.exp(z * zeta) / z - RF * np.exp((1.0 + z) * zeta) / (1.0 + z)\n'
               'def _gauss(om, alpha, zeta, mu, sg):\n'
               '    u = -om + 1j * alpha\n'
               '    W = _pref(om, alpha, zeta) * np.exp(1j * u * mu - 0.5 * sg * sg * u * u)\n'
               '    return np.concatenate((W.real, W.imag))\n'
               'def _Phi(x):\n'
               '    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))\n'
               'def _exact(zeta, mu, sg):\n'
               '    d = (zeta - mu) / sg\n'
               '    return RD * _Phi(d) - RF * math.exp(mu + 0.5 * sg * sg) * _Phi(d - sg)\n'
               'K = _gauss(OM, 1.0, -0.20, 0.0, 0.30)\n',
      'call': 'round(float(exercise_region_expectation(K, RULE, 1.0, 0.0)), 15)',
      'gold_call': 'round(float(_oracle_exercise_region_expectation(K, RULE, 1.0, 0.0)), 15)',
      'tol': 1e-13},
     {'setup': 'import numpy as np\n'
               'import math\n'
               'RD, RF = 0.030, 0.035\n'
               'def _rule(omax, n):\n'
               '    h = 2.0 * omax / n\n'
               '    om = -omax + h * np.arange(n + 1)\n'
               '    w = np.empty(n + 1)\n'
               '    w[0] = w[-1] = 1.0\n'
               '    w[1:-1:2] = 4.0\n'
               '    w[2:-1:2] = 2.0\n'
               '    return om, w * h / 3.0\n'
               'OM, OW = _rule(300.0, 8192)\n'
               'RULE = np.concatenate((OM, OW))\n'
               'def _pref(om, alpha, zeta):\n'
               '    z = 1j * om + alpha\n'
               '    return RD * np.exp(z * zeta) / z - RF * np.exp((1.0 + z) * zeta) / (1.0 + z)\n'
               'def _gauss(om, alpha, zeta, mu, sg):\n'
               '    u = -om + 1j * alpha\n'
               '    W = _pref(om, alpha, zeta) * np.exp(1j * u * mu - 0.5 * sg * sg * u * u)\n'
               '    return np.concatenate((W.real, W.imag))\n'
               'def _Phi(x):\n'
               '    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))\n'
               'def _exact(zeta, mu, sg):\n'
               '    d = (zeta - mu) / sg\n'
               '    return RD * _Phi(d) - RF * math.exp(mu + 0.5 * sg * sg) * _Phi(d - sg)\n'
               'K = _gauss(OM, 1.0, 0.10, -0.05, 0.25)\n',
      'call': 'round(float(exercise_region_expectation(K, RULE, 1.0, 0.0)), 15)',
      'gold_call': 'round(float(_oracle_exercise_region_expectation(K, RULE, 1.0, 0.0)), 15)',
      'tol': 1e-13},
     {'setup': 'import numpy as np\n'
               'def _case(F):\n'
               '    import numpy as np\n'
               '    import math\n'
               '    RD, RF = 0.030, 0.035\n'
               '    def _rule(omax, n):\n'
               '        h = 2.0 * omax / n\n'
               '        om = -omax + h * np.arange(n + 1)\n'
               '        w = np.empty(n + 1)\n'
               '        w[0] = w[-1] = 1.0\n'
               '        w[1:-1:2] = 4.0\n'
               '        w[2:-1:2] = 2.0\n'
               '        return om, w * h / 3.0\n'
               '    OM, OW = _rule(300.0, 8192)\n'
               '    RULE = np.concatenate((OM, OW))\n'
               '    def _pref(om, alpha, zeta):\n'
               '        z = 1j * om + alpha\n'
               '        return RD * np.exp(z * zeta) / z - RF * np.exp((1.0 + z) * zeta) / (1.0 + '
               'z)\n'
               '    def _gauss(om, alpha, zeta, mu, sg):\n'
               '        u = -om + 1j * alpha\n'
               '        W = _pref(om, alpha, zeta) * np.exp(1j * u * mu - 0.5 * sg * sg * u * u)\n'
               '        return np.concatenate((W.real, W.imag))\n'
               '    def _Phi(x):\n'
               '        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))\n'
               '    def _exact(zeta, mu, sg):\n'
               '        d = (zeta - mu) / sg\n'
               '        return RD * _Phi(d) - RF * math.exp(mu + 0.5 * sg * sg) * _Phi(d - sg)\n'
               '    K = _gauss(OM, 1.0, -0.60, 0.0, 0.40)\n'
               '    got = float(F(K, RULE, 1.0, 0.0))\n'
               '    ex = _exact(-0.60, 0.0, 0.40)\n'
               '    return bool(abs(got - ex) < 1e-14)\n',
      'call': '_case(exercise_region_expectation)',
      'gold_call': '_case(_oracle_exercise_region_expectation)'},
     {'setup': 'import numpy as np\n'
               'def _case(F):\n'
               '    import numpy as np\n'
               '    import math\n'
               '    RD, RF = 0.030, 0.035\n'
               '    def _rule(omax, n):\n'
               '        h = 2.0 * omax / n\n'
               '        om = -omax + h * np.arange(n + 1)\n'
               '        w = np.empty(n + 1)\n'
               '        w[0] = w[-1] = 1.0\n'
               '        w[1:-1:2] = 4.0\n'
               '        w[2:-1:2] = 2.0\n'
               '        return om, w * h / 3.0\n'
               '    OM, OW = _rule(300.0, 8192)\n'
               '    RULE = np.concatenate((OM, OW))\n'
               '    def _pref(om, alpha, zeta):\n'
               '        z = 1j * om + alpha\n'
               '        return RD * np.exp(z * zeta) / z - RF * np.exp((1.0 + z) * zeta) / (1.0 + '
               'z)\n'
               '    def _gauss(om, alpha, zeta, mu, sg):\n'
               '        u = -om + 1j * alpha\n'
               '        W = _pref(om, alpha, zeta) * np.exp(1j * u * mu - 0.5 * sg * sg * u * u)\n'
               '        return np.concatenate((W.real, W.imag))\n'
               '    def _Phi(x):\n'
               '        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))\n'
               '    def _exact(zeta, mu, sg):\n'
               '        d = (zeta - mu) / sg\n'
               '        return RD * _Phi(d) - RF * math.exp(mu + 0.5 * sg * sg) * _Phi(d - sg)\n'
               '    vals = []\n'
               '    for al in (0.7, 1.0, 1.6, 2.4):\n'
               '        K = _gauss(OM, al, -0.20, 0.0, 0.30)\n'
               '        vals.append(float(F(K, RULE, al, 0.0)))\n'
               '    return [round(v, 12) for v in vals]\n',
      'call': '_case(exercise_region_expectation)',
      'gold_call': '_case(_oracle_exercise_region_expectation)',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'def _case(F):\n'
               '    import numpy as np\n'
               '    import math\n'
               '    RD, RF = 0.030, 0.035\n'
               '    def _rule(omax, n):\n'
               '        h = 2.0 * omax / n\n'
               '        om = -omax + h * np.arange(n + 1)\n'
               '        w = np.empty(n + 1)\n'
               '        w[0] = w[-1] = 1.0\n'
               '        w[1:-1:2] = 4.0\n'
               '        w[2:-1:2] = 2.0\n'
               '        return om, w * h / 3.0\n'
               '    OM, OW = _rule(300.0, 8192)\n'
               '    RULE = np.concatenate((OM, OW))\n'
               '    def _pref(om, alpha, zeta):\n'
               '        z = 1j * om + alpha\n'
               '        return RD * np.exp(z * zeta) / z - RF * np.exp((1.0 + z) * zeta) / (1.0 + '
               'z)\n'
               '    def _gauss(om, alpha, zeta, mu, sg):\n'
               '        u = -om + 1j * alpha\n'
               '        W = _pref(om, alpha, zeta) * np.exp(1j * u * mu - 0.5 * sg * sg * u * u)\n'
               '        return np.concatenate((W.real, W.imag))\n'
               '    def _Phi(x):\n'
               '        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))\n'
               '    def _exact(zeta, mu, sg):\n'
               '        d = (zeta - mu) / sg\n'
               '        return RD * _Phi(d) - RF * math.exp(mu + 0.5 * sg * sg) * _Phi(d - sg)\n'
               '    K = _gauss(OM, 1.0, -0.20, 0.0, 0.30)\n'
               '    got = float(F(K, RULE, 1.0, -0.30))\n'
               '    ex = _exact(-0.20, -0.30, 0.30)\n'
               '    return bool(abs(got - ex) < 1e-13)\n',
      'call': '_case(exercise_region_expectation)',
      'gold_call': '_case(_oracle_exercise_region_expectation)'},
     {'setup': 'import numpy as np\n'
               'import math\n'
               'RD, RF = 0.030, 0.035\n'
               'def _rule(omax, n):\n'
               '    h = 2.0 * omax / n\n'
               '    om = -omax + h * np.arange(n + 1)\n'
               '    w = np.empty(n + 1)\n'
               '    w[0] = w[-1] = 1.0\n'
               '    w[1:-1:2] = 4.0\n'
               '    w[2:-1:2] = 2.0\n'
               '    return om, w * h / 3.0\n'
               'OM, OW = _rule(300.0, 8192)\n'
               'RULE = np.concatenate((OM, OW))\n'
               'def _pref(om, alpha, zeta):\n'
               '    z = 1j * om + alpha\n'
               '    return RD * np.exp(z * zeta) / z - RF * np.exp((1.0 + z) * zeta) / (1.0 + z)\n'
               'def _gauss(om, alpha, zeta, mu, sg):\n'
               '    u = -om + 1j * alpha\n'
               '    W = _pref(om, alpha, zeta) * np.exp(1j * u * mu - 0.5 * sg * sg * u * u)\n'
               '    return np.concatenate((W.real, W.imag))\n'
               'def _Phi(x):\n'
               '    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))\n'
               'def _exact(zeta, mu, sg):\n'
               '    d = (zeta - mu) / sg\n'
               '    return RD * _Phi(d) - RF * math.exp(mu + 0.5 * sg * sg) * _Phi(d - sg)\n'
               'def run_model():\n'
               '    try:\n'
               '        exercise_region_expectation(np.zeros(10), np.zeros(8), 1.0, 0.0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_exercise_region_expectation(np.zeros(10), np.zeros(8), 1.0, 0.0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'import math\n'
               'RD, RF = 0.030, 0.035\n'
               'def _rule(omax, n):\n'
               '    h = 2.0 * omax / n\n'
               '    om = -omax + h * np.arange(n + 1)\n'
               '    w = np.empty(n + 1)\n'
               '    w[0] = w[-1] = 1.0\n'
               '    w[1:-1:2] = 4.0\n'
               '    w[2:-1:2] = 2.0\n'
               '    return om, w * h / 3.0\n'
               'OM, OW = _rule(300.0, 8192)\n'
               'RULE = np.concatenate((OM, OW))\n'
               'def _pref(om, alpha, zeta):\n'
               '    z = 1j * om + alpha\n'
               '    return RD * np.exp(z * zeta) / z - RF * np.exp((1.0 + z) * zeta) / (1.0 + z)\n'
               'def _gauss(om, alpha, zeta, mu, sg):\n'
               '    u = -om + 1j * alpha\n'
               '    W = _pref(om, alpha, zeta) * np.exp(1j * u * mu - 0.5 * sg * sg * u * u)\n'
               '    return np.concatenate((W.real, W.imag))\n'
               'def _Phi(x):\n'
               '    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))\n'
               'def _exact(zeta, mu, sg):\n'
               '    d = (zeta - mu) / sg\n'
               '    return RD * _Phi(d) - RF * math.exp(mu + 0.5 * sg * sg) * _Phi(d - sg)\n'
               'def run_model():\n'
               '    try:\n'
               '        exercise_region_expectation(np.zeros(8), np.zeros(8), 0.0, 0.0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_exercise_region_expectation(np.zeros(8), np.zeros(8), 0.0, 0.0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
