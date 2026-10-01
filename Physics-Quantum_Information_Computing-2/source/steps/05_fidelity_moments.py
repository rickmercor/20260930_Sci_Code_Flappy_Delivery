"""
Evaluate raw moments of the realized conditional fidelity.

The source evaluates nonlinear statistics before averaging over the recorded measurement outcome. Integrals use the joint branch measure and remain defined at zero-probability input poles.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def fidelity_moments(law: np.ndarray, orders: np.ndarray) -> np.ndarray:
    """Evaluate raw moments of the realized conditional fidelity.

    law : ndarray, shape (4,5)
        Physical conditional_fidelity_law output.
    orders : ndarray, shape (M,)
        Nonnegative integer exponents, in any order; M may be zero.
    Returns
    -------
    ndarray, shape (M,), float
        E[F**orders[i]], with zeroth moment equal to normalization.

    All quantities are dimensionless. Inputs are preserved.
    Raises
    ------
    ValueError
        For invalid input shapes or data outside the stated domain.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
from scipy.optimize import least_squares, linprog

def _oracle_fidelity_moments(law: np.ndarray, orders: np.ndarray) -> np.ndarray:
    law = np.asarray(law, dtype=float)
    orders = np.asarray(orders)
    if law.shape != (4, 5) or not np.isfinite(law).all() or orders.ndim != 1 or (not np.isfinite(orders).all()) or np.any(orders < 0) or np.any(orders != np.floor(orders)):
        raise ValueError('Expected a finite law and nonnegative integer orders.')

    def _validate_physical_law(law):
        """Validate the declared R/E family with a coefficient tolerance of 1e-10."""
        if law.shape != (4, 5) or not np.isfinite(law).all():
            raise ValueError('Expected a finite physical law of shape (4,5).')
        tol = 1e-10
        c = 8 * law[0, 0] - 1
        w = 8 * law[0, 2] + c
        a = 4 * law[0, 4]
        b = 8 * law[0, 1] - a
        if c >= -tol and -tol <= a <= 1 + tol and (-tol <= b <= 1 + tol):
            expected = np.array([[(1 + c) / 8, s * (a + b) / 8, (w - c) / 8, 0.25, s * a / 4] for s in (1, 1, -1, -1)])
            if abs(c * c - (1 - a) * (1 - b)) <= tol and abs(w - (1 - a - b + 2 * a * b)) <= tol and np.allclose(law, expected, rtol=0, atol=tol):
                return
        product = w - c * c
        total = 1 + product - c * c
        discriminant = total * total - 4 * product
        if c >= -tol and discriminant >= -tol:
            root = np.sqrt(max(0.0, discriminant))
            (a, b) = ((total + root) / 2, (total - root) / 2)
            expected = np.tile([(1 + c) / 8, 0.0, (w - c) / 8, 0.25, 0.0], (4, 1))
            if -tol <= a <= 1 + tol and -tol <= b <= 1 + tol and np.allclose(law, expected, rtol=0, atol=tol):
                return
        raise ValueError('Law is outside the declared recorded/erased physical family.')
    _validate_physical_law(law)
    result = []
    for order in orders:
        total = 0.0
        for (q0, q1, q2, p0, p1) in law:

            def _fun(z):
                p = p0 + p1 * z
                f = np.clip((q0 + q1 * z + q2 * z * z) / p, 0, 1)
                return 0.5 * p * f ** order
            total += quad(_fun, -1, 1, epsabs=2e-12, epsrel=2e-12)[0]
        result.append(total)
    return np.array(result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nL=np.array([[0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0], [0.25, -0.0, 0.0, 0.25, -0.0], [0.25, -0.0, 0.0, 0.25, -0.0]], dtype=float)\norders=np.array([0, 1, 2, 7], dtype=float)', 'call': 'fidelity_moments(L, orders)', 'gold_call': '_oracle_fidelity_moments(L, orders)', 'tol': 1e-08}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.125, 0.0, 0.25, 0.25], [0.125, 0.125, 0.0, 0.25, 0.25], [0.125, -0.125, 0.0, 0.25, -0.25], [0.125, -0.125, 0.0, 0.25, -0.25]], dtype=float)\norders=np.array([0, 1, 2, 9], dtype=float)', 'call': 'fidelity_moments(L, orders)', 'gold_call': '_oracle_fidelity_moments(L, orders)', 'tol': 1e-08}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.125, 0.0, 0.25, 0.0], [0.125, 0.125, 0.0, 0.25, 0.0], [0.125, -0.125, 0.0, 0.25, -0.0], [0.125, -0.125, 0.0, 0.25, -0.0]], dtype=float)\norders=np.array([0, 1, 2, 5], dtype=float)', 'call': 'fidelity_moments(L, orders)', 'gold_call': '_oracle_fidelity_moments(L, orders)', 'tol': 1e-08}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.25, 0.125, 0.25, 0.25], [0.125, 0.25, 0.125, 0.25, 0.25], [0.125, -0.25, 0.125, 0.25, -0.25], [0.125, -0.25, 0.125, 0.25, -0.25]], dtype=float)\norders=np.array([0, 1, 2, 6], dtype=float)', 'call': 'fidelity_moments(L, orders)', 'gold_call': '_oracle_fidelity_moments(L, orders)', 'tol': 1e-08}, {'setup': 'import numpy as np\nL=np.array([[0.175, 0.12500000000000003, -0.010000000000000002, 0.25, 0.05000000000000001], [0.175, 0.12500000000000003, -0.010000000000000002, 0.25, 0.05000000000000001], [0.175, -0.12500000000000003, -0.010000000000000002, 0.25, -0.05000000000000001], [0.175, -0.12500000000000003, -0.010000000000000002, 0.25, -0.05000000000000001]], dtype=float)\norders=np.array([1, 2, 3], dtype=float)', 'call': 'fidelity_moments(L, orders)', 'gold_call': '_oracle_fidelity_moments(L, orders)', 'tol': 1e-08}, {'setup': 'import numpy as np\nL=np.array([[0.175, 0.12500000000000003, -0.009999999999999995, 0.25, 0.20000000000000004], [0.175, 0.12500000000000003, -0.009999999999999995, 0.25, 0.20000000000000004], [0.175, -0.12500000000000003, -0.009999999999999995, 0.25, -0.20000000000000004], [0.175, -0.12500000000000003, -0.009999999999999995, 0.25, -0.20000000000000004]], dtype=float)\norders=np.array([1, 2, 3], dtype=float)', 'call': 'fidelity_moments(L, orders)', 'gold_call': '_oracle_fidelity_moments(L, orders)', 'tol': 1e-08}, {'setup': 'import numpy as np\nL=np.array([[0.1875, 0.125, 0.0, 0.25, 0.125], [0.1875, 0.125, 0.0, 0.25, 0.125], [0.1875, -0.125, 0.0, 0.25, -0.125], [0.1875, -0.125, 0.0, 0.25, -0.125]], dtype=float)\norders=np.array([8, 0, 2, 8], dtype=float)', 'call': 'fidelity_moments(L, orders)', 'gold_call': '_oracle_fidelity_moments(L, orders)', 'tol': 1e-08}, {'setup': 'import numpy as np\nL=np.array([[0.1464330352493528, 0.12375000000000001, -0.015333035249352813, 0.25, 0.24250000000000002], [0.1464330352493528, 0.12375000000000001, -0.015333035249352813, 0.25, 0.24250000000000002], [0.1464330352493528, -0.12375000000000001, -0.015333035249352813, 0.25, -0.24250000000000002], [0.1464330352493528, -0.12375000000000001, -0.015333035249352813, 0.25, -0.24250000000000002]], dtype=float)\norders=np.array([1, 4, 16], dtype=float)', 'call': 'fidelity_moments(L, orders)', 'gold_call': '_oracle_fidelity_moments(L, orders)', 'tol': 1e-08}, {'setup': 'import numpy as np\nL=np.array([[0.1464330352493528, 0.12375000000000001, -0.015333035249352808, 0.25, 0.004999999999999999], [0.1464330352493528, 0.12375000000000001, -0.015333035249352808, 0.25, 0.004999999999999999], [0.1464330352493528, -0.12375000000000001, -0.015333035249352808, 0.25, -0.004999999999999999], [0.1464330352493528, -0.12375000000000001, -0.015333035249352808, 0.25, -0.004999999999999999]], dtype=float)\norders=np.array([1, 4, 16], dtype=float)', 'call': 'fidelity_moments(L, orders)', 'gold_call': '_oracle_fidelity_moments(L, orders)', 'tol': 1e-08}, {'setup': 'import numpy as np\nL=np.array([[0.17492181386928965, 0.14500000000000002, 0.009953186130710348, 0.25, 0.11250000000000002], [0.17492181386928965, 0.14500000000000002, 0.009953186130710348, 0.25, 0.11250000000000002], [0.17492181386928965, -0.14500000000000002, 0.009953186130710348, 0.25, -0.11250000000000002], [0.17492181386928965, -0.14500000000000002, 0.009953186130710348, 0.25, -0.11250000000000002]], dtype=float)\norders=np.array([0, 12, 32], dtype=float)', 'call': 'fidelity_moments(L, orders)', 'gold_call': '_oracle_fidelity_moments(L, orders)', 'tol': 1e-08}, {'setup': 'import numpy as np\nL=np.array([[0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_moments(L, np.array([0,1,2,3,8]))', 'gold_call': '_oracle_fidelity_moments(L, np.array([0,1,2,3,8]))', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.0, 0.0, 0.25, 0.0], [0.125, 0.0, 0.0, 0.25, 0.0], [0.125, 0.0, 0.0, 0.25, 0.0], [0.125, 0.0, 0.0, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_moments(L, np.array([0,1,2,3,8]))', 'gold_call': '_oracle_fidelity_moments(L, np.array([0,1,2,3,8]))', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.0, 0.125, 0.25, 0.0], [0.125, 0.0, 0.125, 0.25, 0.0], [0.125, 0.0, 0.125, 0.25, 0.0], [0.125, 0.0, 0.125, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_moments(L, np.array([0,1,2,3,8]))', 'gold_call': '_oracle_fidelity_moments(L, np.array([0,1,2,3,8]))', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.175, 0.0, -0.010000000000000002, 0.25, 0.0], [0.175, 0.0, -0.010000000000000002, 0.25, 0.0], [0.175, 0.0, -0.010000000000000002, 0.25, 0.0], [0.175, 0.0, -0.010000000000000002, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_moments(L, np.array([0,1,2,3,8]))', 'gold_call': '_oracle_fidelity_moments(L, np.array([0,1,2,3,8]))', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.175, 0.0, -0.009999999999999995, 0.25, 0.0], [0.175, 0.0, -0.009999999999999995, 0.25, 0.0], [0.175, 0.0, -0.009999999999999995, 0.25, 0.0], [0.175, 0.0, -0.009999999999999995, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_moments(L, np.array([0,1,2,3,8]))', 'gold_call': '_oracle_fidelity_moments(L, np.array([0,1,2,3,8]))', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.225, 0.0, -0.015, 0.25, 0.0], [0.225, 0.0, -0.015, 0.25, 0.0], [0.225, 0.0, -0.015, 0.25, 0.0], [0.225, 0.0, -0.015, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_moments(L, np.array([0,1,2,3,8]))', 'gold_call': '_oracle_fidelity_moments(L, np.array([0,1,2,3,8]))', 'tol': 2e-07}, {'setup': 'import numpy as np\ndef _exception_code(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    except Exception:\n        return 2.0\n    return 0.0', 'call': '_exception_code(lambda: fidelity_moments(np.ones((4,5)), np.array([-1])))', 'gold_call': '_exception_code(lambda: _oracle_fidelity_moments(np.ones((4,5)), np.array([-1])))', 'tol': 0.0}, {'setup': 'import numpy as np\nL=np.zeros((4,5))\ndef _exception_code(f):\n    try:\n        f()\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0', 'call': '_exception_code(lambda: fidelity_moments(L, np.array([0,1])))', 'gold_call': '_exception_code(lambda: _oracle_fidelity_moments(L, np.array([0,1])))', 'tol': 0.0}]
