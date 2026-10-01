"""
Evaluate Cartesian derivatives of the softened gravitational kernel.

Radial derivatives K_q=(r^{-1} d/dr)^q g obey the Plummer recursion in
equations (11)-(13). Cartesian derivatives include contractions as well
as coordinate powers, which matters for tidal and higher-order terms.

Returns
-------
np.ndarray, shape ((p+1)(p+2)(p+3)/6,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def kernel_derivatives(r: "np.ndarray", epsilon: float, p: int) -> "np.ndarray":
    """Evaluate derivatives of g(r)=-1/sqrt(r dot r+epsilon**2).

    Parameters
    ----------
    r : np.ndarray
        Finite displacement vector of shape (3,).
    epsilon : float
        Strictly positive softening length.
    p : int
        Maximum total derivative order, 0 <= p <= 6.

    Returns
    -------
    result : np.ndarray
        Float64 vector of length (p+1)(p+2)(p+3)/6 containing the raw
        derivatives D_n. Order by increasing total degree, then decreasing
        n_x, then decreasing n_y, with n_z fixed by that degree.
        Thus the first four indices are 000,100,010,001 when p >= 1.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np

def _indices(p):
    return [(a,b,d-a-b) for d in range(p+1)
            for a in range(d,-1,-1) for b in range(d-a,-1,-1)]

def _factorial(n):
    return math.prod(math.factorial(int(a)) for a in n)

def _binomial(n, k):
    return math.prod(math.comb(int(a), int(b)) for a,b in zip(n,k))

def _power(x, n):
    return np.prod(np.asarray(x) ** np.asarray(n))

def _oracle_kernel_derivatives(r: "np.ndarray", epsilon: float, p: int) -> "np.ndarray":
    r = np.asarray(r, dtype=float)
    rho = np.dot(r, r) + epsilon**2
    radial = [-1.0 / np.sqrt(rho)]
    for q in range(1, p+1):
        radial.append(-(2*q-1) * radial[-1] / rho)
    values = []
    for n in _indices(p):
        value = 0.0
        for a in range(n[0]//2+1):
            for b in range(n[1]//2+1):
                for c in range(n[2]//2+1):
                    pairs = (a,b,c)
                    rest = tuple(n[j]-2*pairs[j] for j in range(3))
                    coefficient = _factorial(n) / (2**sum(pairs)*_factorial(pairs)*_factorial(rest))
                    value += coefficient * _power(r, rest) * radial[sum(n)-sum(pairs)]
        values.append(value)
    return np.array(values, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [{'setup': 'import numpy as np\nr=np.array([0.7, -0.2, 0.4])', 'call': 'kernel_derivatives(r.copy(), 0.18, 4)', 'gold_call': '_oracle_kernel_derivatives(r.copy(), 0.18, 4)', 'tol': 1e-10}, {'setup': 'import numpy as np\nr=np.array([0.0, 0.0, 0.0])', 'call': 'kernel_derivatives(r.copy(), 0.3, 0)', 'gold_call': '_oracle_kernel_derivatives(r.copy(), 0.3, 0)', 'tol': 1e-10}, {'setup': 'import numpy as np\nr=np.array([0.0, 0.0, 0.0])', 'call': 'kernel_derivatives(r.copy(), 0.2, 5)', 'gold_call': '_oracle_kernel_derivatives(r.copy(), 0.2, 5)', 'tol': 1e-10}, {'setup': 'import numpy as np\nr=np.array([1.1, 0.0, -0.3])', 'call': 'kernel_derivatives(r.copy(), 0.1, 6)', 'gold_call': '_oracle_kernel_derivatives(r.copy(), 0.1, 6)', 'tol': 1e-10}]
