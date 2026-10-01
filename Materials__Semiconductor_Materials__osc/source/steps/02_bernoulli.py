"""
Evaluate the Bernoulli function that the source's flux discretisation is built on, elementwise and without loss of accuracy near zero or at large magnitude. Raise ValueError for a non-finite argument.

This function carries the entire exponential structure of the discretised carrier flux. Evaluated naively it loses all significant digits as its argument approaches zero, where it must tend to one, and it overflows for large positive argument. Both limits occur inside a single device, so both have to be handled.

Returns
-------
np.ndarray of the same shape as the input, dimensionless
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bernoulli(x: np.ndarray) -> np.ndarray:
    """Evaluate the Bernoulli function that the source's flux discretisation is built on, elementwise and without loss of accuracy near zero or at large magnitude. Raise ValueError for a non-finite argument.

    Parameters
    ----------
    x : np.ndarray
        Dimensionless argument, elementwise.

    Returns
    -------
    np.ndarray of the same shape as the input, dimensionless
    """
    return x

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Canonical implementations for Material_Science-Semiconductor_Materials-10 (rebuild).
Source: arXiv:2609.05170 (CC BY 4.0) -- photocurrent versus effective voltage in organic
solar cells; one-dimensional drift-diffusion with Scharfetter-Gummel fluxes.

THE ONLY PLACE THE SCIENCE IS WRITTEN. build_osc.py slices this into sub_problems/NN_name.py.

Oracle chaining rule: every _oracle_* that needs an earlier step calls that step's _oracle_
twin, NEVER the public name.
"""
import numpy as np
from scipy.linalg import solve_banded

_Q = 1.602176634e-19
_KB = 1.380649e-23
_EPS0 = 8.8541878128e-12


def _phys():
    """Physical constants. A function, not module-level state: the harness keeps
    imports and function definitions and drops module-level assignments."""
    return 1.602176634e-19, 1.380649e-23, 8.8541878128e-12

# declared device configuration (SI)
_SPEC = dict(T=300.0, d=100e-9, Eg=1.4, Phi=0.25, mu_n=2e-8, mu_p=2e-8,
             eps_r=3.5, Nc=1e26, Nv=1e26, Gex=1e28, gamma=1.0)


def _tridiag(lo, di, up, rhs):
    ab = np.zeros((3, len(di)))
    ab[0, 1:] = up[:-1]
    ab[1] = di
    ab[2, :-1] = lo[1:]
    return solve_banded((1, 1), ab, rhs)


def _tridiag_dirichlet(lo, di, up, r, left, right):
    """Solve the interior rows only, with the two end values pinned exactly.

    Putting the Dirichlet rows inside the banded system gives them a unit
    diagonal beside interior diagonals of order 1e8, so partial pivoting swaps
    them out and the pinned value comes back perturbed - by 10% or more once the
    densities span fifteen orders of magnitude. Eliminating them by hand keeps
    the contact values bit-exact and conditions the remaining system better.
    """
    N = len(di)
    if N <= 2:
        return np.array([left, right], float)[:N]
    rr = r[1:-1].copy()
    rr[0] -= lo[1] * left
    rr[-1] -= up[N - 2] * right
    inner = _tridiag(lo[1:-1], di[1:-1], up[1:-1], rr)
    out = np.empty(N, float)
    out[0] = left; out[-1] = right; out[1:-1] = inner
    return out




def _oracle_bernoulli(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("the Bernoulli argument must be finite")
    out = np.empty_like(x)
    small = np.abs(x) < 1.0e-10
    out[small] = 1.0 - x[small] / 2.0 + x[small] ** 2 / 12.0
    big = ~small
    xb = np.clip(x[big], -700.0, 700.0)
    out[big] = xb / np.expm1(xb)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np',
         "call": 'bernoulli(np.array([-5.0,-1.0,-1e-12,0.0,1e-12,1.0,5.0,40.0]))',
         "gold_call": '_oracle_bernoulli(np.array([-5.0,-1.0,-1e-12,0.0,1e-12,1.0,5.0,40.0]))'},
        {"setup": 'import numpy as np',
         "call": 'bernoulli(np.linspace(-20.0,20.0,41))',
         "gold_call": '_oracle_bernoulli(np.linspace(-20.0,20.0,41))'},
        {"setup": 'import numpy as np',
         "call": 'bernoulli(np.array([0.0]))',
         "gold_call": '_oracle_bernoulli(np.array([0.0]))'},
        {"setup": 'import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ndef raises(f):\n    try:\n        f()\n    except ValueError:\n        return 1.0\n    return 0.0',
         "call": 'raises(lambda: bernoulli(np.array([0.0, np.inf])))',
         "gold_call": 'raises(lambda: _oracle_bernoulli(np.array([0.0, np.inf])))'},
    ]
