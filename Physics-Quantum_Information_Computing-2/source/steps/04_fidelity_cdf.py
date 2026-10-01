"""
Evaluate the realized-fidelity cumulative distribution.

Integrate the joint Born-Haar measure on the branch preimages. Compose fidelity_preimages and retain probability atoms. Threshold order is preserved.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def fidelity_cdf(law: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Evaluate the realized-fidelity cumulative distribution.

    law : ndarray, shape (4,5)
        Physical conditional_fidelity_law output.
    thresholds : ndarray, shape (T,)
        Finite thresholds, including values outside [0,1]; T may be zero.
    Returns
    -------
    ndarray, shape (T,), float
        Pr(F<=thresholds[i]) in supplied order, using all four Bell outcomes.

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

def _oracle_fidelity_cdf(law: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    law = np.asarray(law, dtype=float)
    if law.shape != (4, 5) or not np.isfinite(law).all():
        raise ValueError('Expected a finite law of shape (4,5).')

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
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not np.isfinite(thresholds).all():
        raise ValueError('Thresholds must be a finite one-dimensional array.')
    out = []
    for t in thresholds:
        spans = _oracle_fidelity_preimages(law, t)
        mass = 0.0
        for (j, edges) in enumerate(spans):
            (p0, p1) = law[j, 3:]
            for (lo, hi) in edges:
                mass += p0 * (hi - lo) / 2 + p1 * (hi * hi - lo * lo) / 4
        out.append(mass)
    return np.array(out)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nL=np.array([[0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0], [0.25, -0.0, 0.0, 0.25, -0.0], [0.25, -0.0, 0.0, 0.25, -0.0]], dtype=float)\nthresholds=np.array([-0.1, 0.9, 1.0, 1.1], dtype=float)', 'call': 'fidelity_cdf(L, thresholds)', 'gold_call': '_oracle_fidelity_cdf(L, thresholds)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.125, 0.0, 0.25, 0.25], [0.125, 0.125, 0.0, 0.25, 0.25], [0.125, -0.125, 0.0, 0.25, -0.25], [0.125, -0.125, 0.0, 0.25, -0.25]], dtype=float)\nthresholds=np.array([0.499999, 0.5, 0.500001], dtype=float)', 'call': 'fidelity_cdf(L, thresholds)', 'gold_call': '_oracle_fidelity_cdf(L, thresholds)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.125, 0.0, 0.25, 0.0], [0.125, 0.125, 0.0, 0.25, 0.0], [0.125, -0.125, 0.0, 0.25, -0.0], [0.125, -0.125, 0.0, 0.25, -0.0]], dtype=float)\nthresholds=np.array([0.0, 0.2, 0.7, 1.0], dtype=float)', 'call': 'fidelity_cdf(L, thresholds)', 'gold_call': '_oracle_fidelity_cdf(L, thresholds)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.25, 0.125, 0.25, 0.25], [0.125, 0.25, 0.125, 0.25, 0.25], [0.125, -0.25, 0.125, 0.25, -0.25], [0.125, -0.25, 0.125, 0.25, -0.25]], dtype=float)\nthresholds=np.array([0.0, 0.2, 0.7, 1.0], dtype=float)', 'call': 'fidelity_cdf(L, thresholds)', 'gold_call': '_oracle_fidelity_cdf(L, thresholds)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.175, 0.12500000000000003, -0.010000000000000002, 0.25, 0.05000000000000001], [0.175, 0.12500000000000003, -0.010000000000000002, 0.25, 0.05000000000000001], [0.175, -0.12500000000000003, -0.010000000000000002, 0.25, -0.05000000000000001], [0.175, -0.12500000000000003, -0.010000000000000002, 0.25, -0.05000000000000001]], dtype=float)\nthresholds=np.array([0.4, 0.6, 0.8, 0.9, 1.0], dtype=float)', 'call': 'fidelity_cdf(L, thresholds)', 'gold_call': '_oracle_fidelity_cdf(L, thresholds)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.175, 0.12500000000000003, -0.009999999999999995, 0.25, 0.20000000000000004], [0.175, 0.12500000000000003, -0.009999999999999995, 0.25, 0.20000000000000004], [0.175, -0.12500000000000003, -0.009999999999999995, 0.25, -0.20000000000000004], [0.175, -0.12500000000000003, -0.009999999999999995, 0.25, -0.20000000000000004]], dtype=float)\nthresholds=np.array([0.4, 0.6, 0.8, 0.9, 1.0], dtype=float)', 'call': 'fidelity_cdf(L, thresholds)', 'gold_call': '_oracle_fidelity_cdf(L, thresholds)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.225, 0.04999999999999999, -0.015, 0.25, 0.04999999999999999], [0.225, 0.04999999999999999, -0.015, 0.25, 0.04999999999999999], [0.225, -0.04999999999999999, -0.015, 0.25, -0.04999999999999999], [0.225, -0.04999999999999999, -0.015, 0.25, -0.04999999999999999]], dtype=float)\nthresholds=np.array([0.9, 0.82, 0.88, 0.9], dtype=float)', 'call': 'fidelity_cdf(L, thresholds)', 'gold_call': '_oracle_fidelity_cdf(L, thresholds)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.1464330352493528, 0.12375000000000001, -0.015333035249352813, 0.25, 0.24250000000000002], [0.1464330352493528, 0.12375000000000001, -0.015333035249352813, 0.25, 0.24250000000000002], [0.1464330352493528, -0.12375000000000001, -0.015333035249352813, 0.25, -0.24250000000000002], [0.1464330352493528, -0.12375000000000001, -0.015333035249352813, 0.25, -0.24250000000000002]], dtype=float)\nthresholds=np.array([0.52, 0.6, 0.7, 0.95], dtype=float)', 'call': 'fidelity_cdf(L, thresholds)', 'gold_call': '_oracle_fidelity_cdf(L, thresholds)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.1464330352493528, 0.12375000000000001, -0.015333035249352808, 0.25, 0.004999999999999999], [0.1464330352493528, 0.12375000000000001, -0.015333035249352808, 0.25, 0.004999999999999999], [0.1464330352493528, -0.12375000000000001, -0.015333035249352808, 0.25, -0.004999999999999999], [0.1464330352493528, -0.12375000000000001, -0.015333035249352808, 0.25, -0.004999999999999999]], dtype=float)\nthresholds=np.array([0.05, 0.3, 0.7, 0.99], dtype=float)', 'call': 'fidelity_cdf(L, thresholds)', 'gold_call': '_oracle_fidelity_cdf(L, thresholds)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.19625, 0.1075, -0.007525000000000004, 0.25, 0.1075], [0.19625, 0.1075, -0.007525000000000004, 0.25, 0.1075], [0.19625, -0.1075, -0.007525000000000004, 0.25, -0.1075], [0.19625, -0.1075, -0.007525000000000004, 0.25, -0.1075]], dtype=float)\nthresholds=np.array([-2.0, 2.0, 0.8], dtype=float)', 'call': 'fidelity_cdf(L, thresholds)', 'gold_call': '_oracle_fidelity_cdf(L, thresholds)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_cdf(L, np.array([-.1,.49,.5,.7,.9,1.,1.1]))', 'gold_call': '_oracle_fidelity_cdf(L, np.array([-.1,.49,.5,.7,.9,1.,1.1]))', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.0, 0.0, 0.25, 0.0], [0.125, 0.0, 0.0, 0.25, 0.0], [0.125, 0.0, 0.0, 0.25, 0.0], [0.125, 0.0, 0.0, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_cdf(L, np.array([-.1,.49,.5,.7,.9,1.,1.1]))', 'gold_call': '_oracle_fidelity_cdf(L, np.array([-.1,.49,.5,.7,.9,1.,1.1]))', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.0, 0.125, 0.25, 0.0], [0.125, 0.0, 0.125, 0.25, 0.0], [0.125, 0.0, 0.125, 0.25, 0.0], [0.125, 0.0, 0.125, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_cdf(L, np.array([-.1,.49,.5,.7,.9,1.,1.1]))', 'gold_call': '_oracle_fidelity_cdf(L, np.array([-.1,.49,.5,.7,.9,1.,1.1]))', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.175, 0.0, -0.010000000000000002, 0.25, 0.0], [0.175, 0.0, -0.010000000000000002, 0.25, 0.0], [0.175, 0.0, -0.010000000000000002, 0.25, 0.0], [0.175, 0.0, -0.010000000000000002, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_cdf(L, np.array([-.1,.49,.5,.7,.9,1.,1.1]))', 'gold_call': '_oracle_fidelity_cdf(L, np.array([-.1,.49,.5,.7,.9,1.,1.1]))', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.175, 0.0, -0.009999999999999995, 0.25, 0.0], [0.175, 0.0, -0.009999999999999995, 0.25, 0.0], [0.175, 0.0, -0.009999999999999995, 0.25, 0.0], [0.175, 0.0, -0.009999999999999995, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_cdf(L, np.array([-.1,.49,.5,.7,.9,1.,1.1]))', 'gold_call': '_oracle_fidelity_cdf(L, np.array([-.1,.49,.5,.7,.9,1.,1.1]))', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.225, 0.0, -0.015, 0.25, 0.0], [0.225, 0.0, -0.015, 0.25, 0.0], [0.225, 0.0, -0.015, 0.25, 0.0], [0.225, 0.0, -0.015, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_cdf(L, np.array([-.1,.49,.5,.7,.9,1.,1.1]))', 'gold_call': '_oracle_fidelity_cdf(L, np.array([-.1,.49,.5,.7,.9,1.,1.1]))', 'tol': 2e-07}, {'setup': 'import numpy as np\ndef _exception_code(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    except Exception:\n        return 2.0\n    return 0.0', 'call': '_exception_code(lambda: fidelity_cdf(np.ones((3,5)), np.array([.5])))', 'gold_call': '_exception_code(lambda: _oracle_fidelity_cdf(np.ones((3,5)), np.array([.5])))', 'tol': 0.0}, {'setup': 'import numpy as np\nL=np.zeros((4,5))\ndef _exception_code(f):\n    try:\n        f()\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0', 'call': '_exception_code(lambda: fidelity_cdf(L, np.array([0.5])))', 'gold_call': '_exception_code(lambda: _oracle_fidelity_cdf(L, np.array([0.5])))', 'tol': 0.0}]
