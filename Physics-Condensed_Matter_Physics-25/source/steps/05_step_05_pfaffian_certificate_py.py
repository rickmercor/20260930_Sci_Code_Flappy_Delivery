"""
Recover oriented parity, logarithmic Pfaffian magnitude and local protection.

Main paper Eqs. (8), (12), (14), (16)–(17) and Supplement I. The sign includes the full permutation orientation. The local gap is the distance of the Hermitian localizer spectrum from zero and is invariant under its unitary symmetry reduction, whereas nonorthogonal factor congruences preserve parity but change spectral distances.

Returns
-------
return result  # real ndarray (3,), ordered oriented sign, log(abs(Pf(skew))), local gap. Gap has the units of skew; logarithm uses the supplied numeric energy unit. Singular inputs return [0,0,0], with the middle zero a defined finite sentinel.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def pfaffian_certificate(skew: ArrayLike, factor: ArrayLike, orientation: int = 1) -> np.ndarray:
    'Recover oriented parity, logarithmic Pfaffian magnitude and local protection.\n\nParameters\n----------\nskew : real finite (n,n), skew-symmetric, positive even n.\nfactor : real (n+2,n), any valid scale/L/T/permutation factor from skew_factor.\norientation : integer +1 or -1, raw Pfaffian sign of the chosen trivial reference; default +1.\n\nRaises\n------\nValueError for incompatible shapes, numeric types, nonfinite values, or the stated domain violations. Numeric array-like inputs are accepted; real inputs have real numeric type, complex input is allowed only for layer, and booleans, strings and object arrays are invalid. Integer parameters and index arrays have integer type; integer entries packed into real factor/crossing records are integer-valued. Skew symmetry uses relative max-entry tolerance 1e-12. The packed permutation contains each integer index once, L is unit lower triangular within absolute tolerance 1e-12, scale is nonnegative, and reserved entries are zero. Zero scale requires a zero matrix and zero pivots. The factor identity must hold within relative max-entry tolerance 1e-9 (absolute tolerance 1e-9 for the zero matrix).\n\nReturns\n-------\nreal ndarray (3,), ordered oriented sign, log(abs(Pf(skew))), local gap. Gap has the units of skew; logarithm uses the supplied numeric energy unit. Singular inputs return [0,0,0], with the middle zero a defined finite sentinel.'
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def _oracle_pfaffian_certificate(skew: ArrayLike, factor: ArrayLike, orientation: int = 1) -> np.ndarray:
    def _checked_factor(factor, s):
        n = len(s)
        f = _checked_numeric(factor, 'factor')
        if f.shape != (n + 2, n):
            raise ValueError('factor must have shape (n+2,n)')
        p = _checked_permutation(f[n], n, packed=True)
        l = f[:n]
        if np.max(abs(np.triu(l, 1))) > 1e-12 or np.max(abs(np.diag(l) - 1)) > 1e-12:
            raise ValueError('factor L must be unit lower triangular within 1e-12')
        d, scale = f[n + 1, :n // 2], f[n + 1, n // 2]
        if scale < 0 or np.any(f[n + 1, n // 2 + 1:] != 0):
            raise ValueError('factor scale must be nonnegative and reserved entries must be zero')
        norm = float(np.max(abs(s)))
        if scale == 0 and (norm != 0 or np.any(d != 0)):
            raise ValueError('zero scale is valid only for a zero matrix and zero pivots')
        t = np.zeros((n, n))
        for i, pivot in enumerate(d):
            t[2*i, 2*i+1] = pivot
            t[2*i+1, 2*i] = -pivot
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            expected = s[np.ix_(p, p)] / (norm if norm else 1.)
            rebuilt = l @ (((scale / norm) if norm else scale) * t) @ l.T
        if not np.isfinite(rebuilt).all() or np.max(abs(rebuilt - expected)) > 1e-9:
            raise ValueError('factor must reconstruct the supplied matrix within relative max-entry tolerance 1e-9')
        return f, p

    def _checked_numeric(value, name, real=True):
        try:
            a = np.asarray(value)
            allowed = 'iuf' if real else 'iufc'
            if a.dtype.kind not in allowed or not np.isfinite(a).all():
                raise ValueError(name + ' must contain finite numeric values of the declared type')
            with np.errstate(over='ignore', invalid='ignore'):
                a = a.astype(float if real else complex)
            if not np.isfinite(a).all():
                raise ValueError(name + ' exceeds the supported floating-point range')
            return a
        except (TypeError, OverflowError) as exc:
            raise ValueError(name + ' must be a numeric scalar or array') from exc

    def _checked_orientation(orientation):
        sign = _checked_scalar(orientation, 'orientation', integer=True)
        if sign not in (-1, 1):
            raise ValueError('orientation must be integer +1 or -1')
        return sign

    def _checked_permutation(ordering, n, packed=False):
        a = _checked_numeric(ordering, 'permutation')
        if a.shape != (n,) or (not packed and np.asarray(ordering).dtype.kind not in 'iu'):
            raise ValueError('permutation must have the declared shape and integer type')
        if np.any(a != np.floor(a)) or np.any(a < 0) or np.any(a >= n):
            raise ValueError('permutation entries must be integer indices from 0 through n-1')
        p = a.astype(int)
        if not np.array_equal(np.sort(p), np.arange(n)):
            raise ValueError('permutation must contain each index exactly once')
        return p

    def _checked_scalar(value, name, minimum=None, strict=False, integer=False):
        a = _checked_numeric(value, name)
        if a.shape != ():
            raise ValueError(name + ' must be a scalar')
        x = float(a)
        if integer and (np.asarray(value).dtype.kind not in 'iu' or x != np.floor(x)):
            raise ValueError(name + ' must have integer type')
        if minimum is not None and (x < minimum or (strict and x == minimum)):
            raise ValueError(name + ' is outside its stated domain')
        return int(x) if integer else x

    def _checked_skew(skew):
        a = _checked_numeric(skew, 'skew')
        if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) == 0 or len(a) % 2:
            raise ValueError('skew must be square with positive even size')
        scale = float(np.max(abs(a)))
        if scale and np.max(abs(a / scale + a.T / scale)) > 1e-12:
            raise ValueError('matrix must be skew-symmetric to relative max-entry tolerance 1e-12')
        return a

    s=_checked_skew(skew);n=len(s);f,p=_checked_factor(factor,s);orientation=_checked_orientation(orientation)
    piv=f[n+1,:n//2];scale=f[n+1,n//2]
    if scale==0 or np.any(piv==0):return np.array([0.,0.,0.])
    parity=(-1)**sum(np.count_nonzero(p[i+1:]<p[i]) for i in range(n))
    sg=float(orientation*parity*np.prod(np.sign(piv)))
    logabs=float(np.sum(np.log(abs(piv)))+(n//2)*np.log(scale))
    # Scaling protects this diagnostic under uniform underflow/overflow regimes.
    gap=float(np.min(abs(eigvalsh(1j*(s/scale))))*scale)
    return np.array([sg,logabs,gap])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\ns=np.array([[0.0, 2.0], [-2.0, 0.0]],dtype=float)\nf=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [2.0, 1.0]],dtype=float)', 'call': 'pfaffian_certificate(s,f,1)', 'gold_call': '_oracle_pfaffian_certificate(s,f,1)'}, {'setup': 'import numpy as np\ns=np.array([[0.0, -2.0], [2.0, 0.0]],dtype=float)\nf=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-2.0, 1.0]],dtype=float)', 'call': 'pfaffian_certificate(s,f,1)', 'gold_call': '_oracle_pfaffian_certificate(s,f,1)'}, {'setup': 'import numpy as np\ns=np.array([[0.0, 2.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0], [0.0, 0.0, -3.0, 0.0]],dtype=float)\nf=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 1.0, 0.0]],dtype=float)', 'call': 'pfaffian_certificate(s,f,-1)', 'gold_call': '_oracle_pfaffian_certificate(s,f,-1)'}, {'setup': 'import numpy as np\ns=np.array([[0.0, -2.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0], [0.0, 0.0, -3.0, 0.0]],dtype=float)\nf=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 3.0, 2.0], [2.0, -3.0, 1.0, 0.0]],dtype=float)', 'call': 'pfaffian_certificate(s,f,1)', 'gold_call': '_oracle_pfaffian_certificate(s,f,1)'}, {'setup': 'import numpy as np\ns=np.array([[0.0, -5e+149, 0.0, 0.0, 0.0, 0.0], [5e+149, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2e+150, 0.0, 0.0], [0.0, 0.0, -2e+150, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 2.9999999999999998e+150], [0.0, 0.0, 0.0, 0.0, -2.9999999999999998e+150, 0.0]],dtype=float)\nf=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 2.0, 3.0, 4.0, 5.0], [0.5, 2.0, 3.0, 1e+150, 0.0, 0.0]],dtype=float)', 'call': '(pfaffian_certificate(s,f,1)*np.array([1.,1.,1e-150]))', 'gold_call': '(_oracle_pfaffian_certificate(s,f,1)*np.array([1.,1.,1e-150]))'}, {'setup': 'import numpy as np\ns=np.array([[0.0, 0.0, 5e-151, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2e-150, 0.0, 0.0], [-5e-151, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -2e-150, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 3e-150], [0.0, 0.0, 0.0, 0.0, -3e-150, 0.0]],dtype=float)\nf=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 2.0, 1.0, 3.0, 4.0, 5.0], [0.5, 2.0, 3.0, 1e-150, 0.0, 0.0]],dtype=float)', 'call': '(pfaffian_certificate(s,f,1)*np.array([1.,1.,1e+150]))', 'gold_call': '(_oracle_pfaffian_certificate(s,f,1)*np.array([1.,1.,1e+150]))'}, {'setup': 'import numpy as np\ns=np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0, -3.0, 0.0]],dtype=float)\nf=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 0.0, 3.0, 1.0, 0.0, 0.0]],dtype=float)', 'call': 'pfaffian_certificate(s,f,1)', 'gold_call': '_oracle_pfaffian_certificate(s,f,1)'}, {'setup': 'import numpy as np\ns=np.array([[0.0, -2.0, -0.019200000000000002, 0.0, 0.024, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0192, 0.0, 0.0, -0.4232, 0.0, 0.064], [0.0, 0.0, 0.4232, 0.0, 0.096, 0.0], [-0.024, 0.0, 0.0, -0.096, 0.0, -0.08], [0.0, 0.0, -0.064, 0.0, 0.08, 0.0]],dtype=float)\nf=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.2, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, -0.8, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.3, 0.0, 0.0, 0.0, 0.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.08, 0.5, 2.0, 1.0, 0.0, 0.0]],dtype=float)', 'call': 'pfaffian_certificate(s,f,1)', 'gold_call': '_oracle_pfaffian_certificate(s,f,1)'}]
