"""
Evaluate beta-weighted expectations of the fidelity distribution.

The importance function W is a normalized beta density on fidelity. Average W(F_j) over the joint experiment. For exponent zero, the continuous polynomial endpoint value is one.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def importance_expectations(law: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Evaluate beta-weighted expectations of the fidelity distribution.

    law : ndarray, shape (4,5)
        Physical conditional_fidelity_law output.
    priors : ndarray, shape (K,2)
        Finite [alpha,beta] rows with both shapes >=1; K may be zero.
    Returns
    -------
    ndarray, shape (K,), float
        E[F**(alpha-1)*(1-F)**(beta-1)/B(alpha,beta)] in row order.

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

def _oracle_importance_expectations(law: np.ndarray, priors: np.ndarray) -> np.ndarray:
    law = np.asarray(law, dtype=float)
    priors = np.asarray(priors, dtype=float)
    if law.shape != (4, 5) or not np.isfinite(law).all() or priors.ndim != 2 or (priors.shape[1] != 2) or (not np.isfinite(priors).all()) or np.any(priors < 1):
        raise ValueError('Expected a finite law and beta shapes alpha,beta >= 1.')

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
    for (alpha, beta) in priors:
        total = 0.0
        norm = np.exp(-betaln(alpha, beta))
        for (q0, q1, q2, p0, p1) in law:

            def _fun(z):
                p = p0 + p1 * z
                f = np.clip((q0 + q1 * z + q2 * z * z) / p, 0, 1)
                return 0.5 * p * norm * f ** (alpha - 1) * (1 - f) ** (beta - 1)
            total += quad(_fun, -1, 1, epsabs=2e-11, epsrel=2e-11)[0]
        result.append(total)
    return np.array(result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nL=np.array([[0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0], [0.25, -0.0, 0.0, 0.25, -0.0], [0.25, -0.0, 0.0, 0.25, -0.0]], dtype=float)\npriors=np.array([[1, 1], [3, 1], [3, 2]], dtype=float)', 'call': 'importance_expectations(L, priors)', 'gold_call': '_oracle_importance_expectations(L, priors)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.125, 0.0, 0.25, 0.25], [0.125, 0.125, 0.0, 0.25, 0.25], [0.125, -0.125, 0.0, 0.25, -0.25], [0.125, -0.125, 0.0, 0.25, -0.25]], dtype=float)\npriors=np.array([[1, 1], [2, 1], [6, 2]], dtype=float)', 'call': 'importance_expectations(L, priors)', 'gold_call': '_oracle_importance_expectations(L, priors)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.125, 0.0, 0.25, 0.0], [0.125, 0.125, 0.0, 0.25, 0.0], [0.125, -0.125, 0.0, 0.25, -0.0], [0.125, -0.125, 0.0, 0.25, -0.0]], dtype=float)\npriors=np.array([[1, 1], [2, 3], [7, 1]], dtype=float)', 'call': 'importance_expectations(L, priors)', 'gold_call': '_oracle_importance_expectations(L, priors)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.25, 0.125, 0.25, 0.25], [0.125, 0.25, 0.125, 0.25, 0.25], [0.125, -0.25, 0.125, 0.25, -0.25], [0.125, -0.25, 0.125, 0.25, -0.25]], dtype=float)\npriors=np.array([[2.0, 1.0], [5.0, 1.0], [12.0, 2.5]], dtype=float)', 'call': 'importance_expectations(L, priors)', 'gold_call': '_oracle_importance_expectations(L, priors)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.175, 0.12500000000000003, -0.010000000000000002, 0.25, 0.05000000000000001], [0.175, 0.12500000000000003, -0.010000000000000002, 0.25, 0.05000000000000001], [0.175, -0.12500000000000003, -0.010000000000000002, 0.25, -0.05000000000000001], [0.175, -0.12500000000000003, -0.010000000000000002, 0.25, -0.05000000000000001]], dtype=float)\npriors=np.array([[2, 1], [5, 1], [8, 1]], dtype=float)', 'call': 'importance_expectations(L, priors)', 'gold_call': '_oracle_importance_expectations(L, priors)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.175, 0.12500000000000003, -0.009999999999999995, 0.25, 0.20000000000000004], [0.175, 0.12500000000000003, -0.009999999999999995, 0.25, 0.20000000000000004], [0.175, -0.12500000000000003, -0.009999999999999995, 0.25, -0.20000000000000004], [0.175, -0.12500000000000003, -0.009999999999999995, 0.25, -0.20000000000000004]], dtype=float)\npriors=np.array([[2, 1], [5, 1], [8, 1]], dtype=float)', 'call': 'importance_expectations(L, priors)', 'gold_call': '_oracle_importance_expectations(L, priors)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.1875, 0.125, 0.0, 0.25, 0.125], [0.1875, 0.125, 0.0, 0.25, 0.125], [0.1875, -0.125, 0.0, 0.25, -0.125], [0.1875, -0.125, 0.0, 0.25, -0.125]], dtype=float)\npriors=np.array([[1.5, 1.5], [12.0, 2.5]], dtype=float)', 'call': 'importance_expectations(L, priors)', 'gold_call': '_oracle_importance_expectations(L, priors)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.1464330352493528, 0.12375000000000001, -0.015333035249352813, 0.25, 0.24250000000000002], [0.1464330352493528, 0.12375000000000001, -0.015333035249352813, 0.25, 0.24250000000000002], [0.1464330352493528, -0.12375000000000001, -0.015333035249352813, 0.25, -0.24250000000000002], [0.1464330352493528, -0.12375000000000001, -0.015333035249352813, 0.25, -0.24250000000000002]], dtype=float)\npriors=np.array([[30, 1], [1, 30]], dtype=float)', 'call': 'importance_expectations(L, priors)', 'gold_call': '_oracle_importance_expectations(L, priors)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.1464330352493528, 0.12375000000000001, -0.015333035249352808, 0.25, 0.004999999999999999], [0.1464330352493528, 0.12375000000000001, -0.015333035249352808, 0.25, 0.004999999999999999], [0.1464330352493528, -0.12375000000000001, -0.015333035249352808, 0.25, -0.004999999999999999], [0.1464330352493528, -0.12375000000000001, -0.015333035249352808, 0.25, -0.004999999999999999]], dtype=float)\npriors=np.array([[2.2, 1.7], [1.7, 2.2]], dtype=float)', 'call': 'importance_expectations(L, priors)', 'gold_call': '_oracle_importance_expectations(L, priors)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.1793139024560011, 0.13624999999999998, 0.004136097543998914, 0.25, 0.1025], [0.1793139024560011, 0.13624999999999998, 0.004136097543998914, 0.25, 0.1025], [0.1793139024560011, -0.13624999999999998, 0.004136097543998914, 0.25, -0.1025], [0.1793139024560011, -0.13624999999999998, 0.004136097543998914, 0.25, -0.1025]], dtype=float)\npriors=np.array([[1, 1], [7, 3], [7, 3], [25, 8]], dtype=float)', 'call': 'importance_expectations(L, priors)', 'gold_call': '_oracle_importance_expectations(L, priors)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0]], dtype=float)', 'call': 'importance_expectations(L, np.array([[3.,1.],[1.,3.],[4.,4.],[12.,2.5]]))', 'gold_call': '_oracle_importance_expectations(L, np.array([[3.,1.],[1.,3.],[4.,4.],[12.,2.5]]))', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.0, 0.0, 0.25, 0.0], [0.125, 0.0, 0.0, 0.25, 0.0], [0.125, 0.0, 0.0, 0.25, 0.0], [0.125, 0.0, 0.0, 0.25, 0.0]], dtype=float)', 'call': 'importance_expectations(L, np.array([[3.,1.],[1.,3.],[4.,4.],[12.,2.5]]))', 'gold_call': '_oracle_importance_expectations(L, np.array([[3.,1.],[1.,3.],[4.,4.],[12.,2.5]]))', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.0, 0.125, 0.25, 0.0], [0.125, 0.0, 0.125, 0.25, 0.0], [0.125, 0.0, 0.125, 0.25, 0.0], [0.125, 0.0, 0.125, 0.25, 0.0]], dtype=float)', 'call': 'importance_expectations(L, np.array([[3.,1.],[1.,3.],[4.,4.],[12.,2.5]]))', 'gold_call': '_oracle_importance_expectations(L, np.array([[3.,1.],[1.,3.],[4.,4.],[12.,2.5]]))', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.175, 0.0, -0.010000000000000002, 0.25, 0.0], [0.175, 0.0, -0.010000000000000002, 0.25, 0.0], [0.175, 0.0, -0.010000000000000002, 0.25, 0.0], [0.175, 0.0, -0.010000000000000002, 0.25, 0.0]], dtype=float)', 'call': 'importance_expectations(L, np.array([[3.,1.],[1.,3.],[4.,4.],[12.,2.5]]))', 'gold_call': '_oracle_importance_expectations(L, np.array([[3.,1.],[1.,3.],[4.,4.],[12.,2.5]]))', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.175, 0.0, -0.009999999999999995, 0.25, 0.0], [0.175, 0.0, -0.009999999999999995, 0.25, 0.0], [0.175, 0.0, -0.009999999999999995, 0.25, 0.0], [0.175, 0.0, -0.009999999999999995, 0.25, 0.0]], dtype=float)', 'call': 'importance_expectations(L, np.array([[3.,1.],[1.,3.],[4.,4.],[12.,2.5]]))', 'gold_call': '_oracle_importance_expectations(L, np.array([[3.,1.],[1.,3.],[4.,4.],[12.,2.5]]))', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.225, 0.0, -0.015, 0.25, 0.0], [0.225, 0.0, -0.015, 0.25, 0.0], [0.225, 0.0, -0.015, 0.25, 0.0], [0.225, 0.0, -0.015, 0.25, 0.0]], dtype=float)', 'call': 'importance_expectations(L, np.array([[3.,1.],[1.,3.],[4.,4.],[12.,2.5]]))', 'gold_call': '_oracle_importance_expectations(L, np.array([[3.,1.],[1.,3.],[4.,4.],[12.,2.5]]))', 'tol': 2e-07}, {'setup': 'import numpy as np\ndef _exception_code(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    except Exception:\n        return 2.0\n    return 0.0', 'call': '_exception_code(lambda: importance_expectations(np.ones((4,5)), np.array([[.5,2.]])))', 'gold_call': '_exception_code(lambda: _oracle_importance_expectations(np.ones((4,5)), np.array([[.5,2.]])))', 'tol': 0.0}, {'setup': 'import numpy as np\nL=np.zeros((4,5))\ndef _exception_code(f):\n    try:\n        f()\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0', 'call': '_exception_code(lambda: importance_expectations(L, np.array([[2.,1.]])))', 'gold_call': '_exception_code(lambda: _oracle_importance_expectations(L, np.array([[2.,1.]])))', 'tol': 0.0}]
