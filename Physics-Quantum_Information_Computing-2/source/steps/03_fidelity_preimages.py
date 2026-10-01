"""
Find each branch preimage of the cumulative-fidelity event.

Use the rational branch law to represent F_j(z)<=threshold over z in [-1,1]. The preimage representation supports disconnected intervals and atoms; isolated points have zero Haar measure.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def fidelity_preimages(law: np.ndarray, threshold: float) -> np.ndarray:
    """Find each branch preimage of the cumulative-fidelity event.

    law : ndarray, shape (4,5)
        A physical conditional_fidelity_law output with columns q0,q1,q2,p0,p1.
    threshold : float
        Any finite fidelity threshold.
    Returns
    -------
    ndarray, shape (4,2,2), float
        For each Bell row, up to two maximal intervals [lo,hi] with lo<hi,
        ordered by lo, whose union is the event up to sets of zero Haar
        measure. An interval may end at a zero-probability endpoint, but
        isolated points, including an isolated zero-probability endpoint,
        are omitted, so a row whose event is a single point is [[0,0],[0,0]].
        Unused slots are [0,0]; the full event is [[-1,1],[0,0]]. Equality
        includes a constant branch.

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

def _oracle_fidelity_preimages(law: np.ndarray, threshold: float) -> np.ndarray:
    law = np.asarray(law, dtype=float)
    if law.shape != (4, 5) or not np.isfinite(law).all() or (not np.isfinite(threshold)):
        raise ValueError('Expected a finite law of shape (4,5) and finite threshold.')

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
    out = np.zeros((4, 2, 2))
    for (j, (q0, q1, q2, p0, p1)) in enumerate(law):
        coeff = np.array([q2, q1 - threshold * p1, q0 - threshold * p0])
        scale = max(abs(coeff))
        if scale == 0:
            out[j, 0] = [-1, 1]
            continue
        coeff = coeff / scale
        if abs(coeff[0]) < 2e-14:
            roots = [] if abs(coeff[1]) < 2e-14 else [-coeff[2] / coeff[1]]
        else:
            roots = np.roots(coeff)
            roots = [r.real for r in roots if abs(r.imag) < 1e-10]
        bounds = sorted(set([-1.0, 1.0] + [float(r) for r in roots if -1 < r < 1]))
        parts = []
        for (lo, hi) in zip(bounds[:-1], bounds[1:]):
            if np.polyval(coeff, (lo + hi) / 2) <= 0:
                if parts and abs(parts[-1][1] - lo) < 1e-12:
                    parts[-1][1] = hi
                else:
                    parts.append([lo, hi])
        if parts:
            out[j, :len(parts)] = parts
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nL=np.array([[0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0], [0.25, -0.0, 0.0, 0.25, -0.0], [0.25, -0.0, 0.0, 0.25, -0.0]], dtype=float)', 'call': 'fidelity_preimages(L, 0.999)', 'gold_call': '_oracle_fidelity_preimages(L, 0.999)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0], [0.25, -0.0, 0.0, 0.25, -0.0], [0.25, -0.0, 0.0, 0.25, -0.0]], dtype=float)', 'call': 'fidelity_preimages(L, 1)', 'gold_call': '_oracle_fidelity_preimages(L, 1)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.125, 0.0, 0.25, 0.25], [0.125, 0.125, 0.0, 0.25, 0.25], [0.125, -0.125, 0.0, 0.25, -0.25], [0.125, -0.125, 0.0, 0.25, -0.25]], dtype=float)', 'call': 'fidelity_preimages(L, 0.5)', 'gold_call': '_oracle_fidelity_preimages(L, 0.5)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.125, 0.0, 0.25, 0.25], [0.125, 0.125, 0.0, 0.25, 0.25], [0.125, -0.125, 0.0, 0.25, -0.25], [0.125, -0.125, 0.0, 0.25, -0.25]], dtype=float)', 'call': 'fidelity_preimages(L, 0.49)', 'gold_call': '_oracle_fidelity_preimages(L, 0.49)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.125, 0.0, 0.25, 0.0], [0.125, 0.125, 0.0, 0.25, 0.0], [0.125, -0.125, 0.0, 0.25, -0.0], [0.125, -0.125, 0.0, 0.25, -0.0]], dtype=float)', 'call': 'fidelity_preimages(L, 0.3)', 'gold_call': '_oracle_fidelity_preimages(L, 0.3)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.25, 0.125, 0.25, 0.25], [0.125, 0.25, 0.125, 0.25, 0.25], [0.125, -0.25, 0.125, 0.25, -0.25], [0.125, -0.25, 0.125, 0.25, -0.25]], dtype=float)', 'call': 'fidelity_preimages(L, 0.7)', 'gold_call': '_oracle_fidelity_preimages(L, 0.7)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.225, 0.04999999999999999, -0.015, 0.25, 0.04999999999999999], [0.225, 0.04999999999999999, -0.015, 0.25, 0.04999999999999999], [0.225, -0.04999999999999999, -0.015, 0.25, -0.04999999999999999], [0.225, -0.04999999999999999, -0.015, 0.25, -0.04999999999999999]], dtype=float)', 'call': 'fidelity_preimages(L, 0.88)', 'gold_call': '_oracle_fidelity_preimages(L, 0.88)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.13545825033167594, 0.16124999999999998, 0.027541749668324053, 0.25, 0.24749999999999997], [0.13545825033167594, 0.16124999999999998, 0.027541749668324053, 0.25, 0.24749999999999997], [0.13545825033167594, -0.16124999999999998, 0.027541749668324053, 0.25, -0.24749999999999997], [0.13545825033167594, -0.16124999999999998, 0.027541749668324053, 0.25, -0.24749999999999997]], dtype=float)', 'call': 'fidelity_preimages(L, 0.51)', 'gold_call': '_oracle_fidelity_preimages(L, 0.51)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.13545825033167594, 0.16125, 0.02754174966832406, 0.25, 0.07499999999999998], [0.13545825033167594, 0.16125, 0.02754174966832406, 0.25, 0.07499999999999998], [0.13545825033167594, -0.16125, 0.02754174966832406, 0.25, -0.07499999999999998], [0.13545825033167594, -0.16125, 0.02754174966832406, 0.25, -0.07499999999999998]], dtype=float)', 'call': 'fidelity_preimages(L, 0.51)', 'gold_call': '_oracle_fidelity_preimages(L, 0.51)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.1372474487139159, 0.12875, -0.006097448713915885, 0.25, 0.2475], [0.1372474487139159, 0.12875, -0.006097448713915885, 0.25, 0.2475], [0.1372474487139159, -0.12875, -0.006097448713915885, 0.25, -0.2475], [0.1372474487139159, -0.12875, -0.006097448713915885, 0.25, -0.2475]], dtype=float)', 'call': 'fidelity_preimages(L, 0.64)', 'gold_call': '_oracle_fidelity_preimages(L, 0.64)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0], [0.25, 0.0, 0.0, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_preimages(L, 1.0)', 'gold_call': '_oracle_fidelity_preimages(L, 1.0)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.0, 0.0, 0.25, 0.0], [0.125, 0.0, 0.0, 0.25, 0.0], [0.125, 0.0, 0.0, 0.25, 0.0], [0.125, 0.0, 0.0, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_preimages(L, 0.5)', 'gold_call': '_oracle_fidelity_preimages(L, 0.5)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.125, 0.0, 0.125, 0.25, 0.0], [0.125, 0.0, 0.125, 0.25, 0.0], [0.125, 0.0, 0.125, 0.25, 0.0], [0.125, 0.0, 0.125, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_preimages(L, 0.8)', 'gold_call': '_oracle_fidelity_preimages(L, 0.8)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.175, 0.0, -0.010000000000000002, 0.25, 0.0], [0.175, 0.0, -0.010000000000000002, 0.25, 0.0], [0.175, 0.0, -0.010000000000000002, 0.25, 0.0], [0.175, 0.0, -0.010000000000000002, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_preimages(L, 0.7)', 'gold_call': '_oracle_fidelity_preimages(L, 0.7)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.175, 0.0, -0.009999999999999995, 0.25, 0.0], [0.175, 0.0, -0.009999999999999995, 0.25, 0.0], [0.175, 0.0, -0.009999999999999995, 0.25, 0.0], [0.175, 0.0, -0.009999999999999995, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_preimages(L, 0.7)', 'gold_call': '_oracle_fidelity_preimages(L, 0.7)', 'tol': 2e-07}, {'setup': 'import numpy as np\nL=np.array([[0.225, 0.0, -0.015, 0.25, 0.0], [0.225, 0.0, -0.015, 0.25, 0.0], [0.225, 0.0, -0.015, 0.25, 0.0], [0.225, 0.0, -0.015, 0.25, 0.0]], dtype=float)', 'call': 'fidelity_preimages(L, 0.85)', 'gold_call': '_oracle_fidelity_preimages(L, 0.85)', 'tol': 2e-07}, {'setup': 'import numpy as np\ndef _exception_code(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    except Exception:\n        return 2.0\n    return 0.0', 'call': '_exception_code(lambda: fidelity_preimages(np.ones((3,5)), .5))', 'gold_call': '_exception_code(lambda: _oracle_fidelity_preimages(np.ones((3,5)), .5))', 'tol': 0.0}, {'setup': 'import numpy as np\nL=np.zeros((4,5))\ndef _exception_code(f):\n    try:\n        f()\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0', 'call': '_exception_code(lambda: fidelity_preimages(L, 0.5))', 'gold_call': '_exception_code(lambda: _oracle_fidelity_preimages(L, 0.5))', 'tol': 0.0}]
