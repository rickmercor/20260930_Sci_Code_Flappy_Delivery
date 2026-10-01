"""
Evaluate a family of sup-norm-normalized piecewise-polynomial test functions and their exact first derivatives on a uniform grid.

Weak-form equation learning multiplies the governing equation by smooth, compactly supported test functions and integrates by parts, so that derivatives act on the test function instead of on noisy data. The separable test functions used for structured population models are products of one-dimensional bump functions of the form

 

$$\phi(x)=C\,(x-x_1)^p\,(x_2-x)^p,\qquad x_1<x<x_2,$$

 

and $\phi(x)=0$ outside the open interval $(x_1,x_2)$, with equal exponents $p$ on both sides and the constant $C$ chosen so that $\max_x\phi(x)=1$. Each bump has the same support length $\ell=x_2-x_1$ and is placed by its left endpoint $x_1$. With $p\ge 2$ the bump and its first derivative vanish at both ends of the support, so boundary terms from integration by parts disappear.

Returns
-------
np.ndarray of shape (2, K, n): test-function values [0] and exact derivatives [1] at the grid points
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def polynomial_test_functions(grid: "np.ndarray", left_endpoints: "np.ndarray", support_length: float, power: int) -> "np.ndarray":
    '''Evaluate normalized piecewise-polynomial test functions and their derivatives.
 
    For each left endpoint x1_k the test function is
 
        phi_k(x) = C (x - x1_k)^p (x1_k + ell - x)^p   on the open interval (x1_k, x1_k + ell),
 
    and zero elsewhere (including the two support endpoints), where ell = support_length,
    p = power and C is chosen so that max_x phi_k(x) = 1 (the maximum is attained at the
    support midpoint). The derivative d phi_k / dx must be the exact analytic derivative
    of this piecewise polynomial, evaluated at the grid points (also zero outside the open
    support). Grid points and endpoints are arbitrary real numbers; supports need not lie
    inside the grid range.
 
    Parameters
    ----------
    grid : np.ndarray
        1-D array of n evaluation points.
    left_endpoints : np.ndarray
        1-D array of K left support endpoints x1_k.
    support_length : float
        Common support length ell > 0.
    power : int
        Exponent p >= 2 applied to both factors.
 
    Returns
    -------
    values : np.ndarray
        Array of shape (2, K, n): values[0, k, i] = phi_k(grid[i]) and
        values[1, k, i] = phi_k'(grid[i]).
 
    Raises
    ------
    ValueError
        If grid or left_endpoints is not a nonempty 1-D array, support_length <= 0,
        or power is not an integer >= 2.
    '''
    return values

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _oracle_polynomial_test_functions(grid: "np.ndarray", left_endpoints: "np.ndarray", support_length: float, power: int) -> "np.ndarray":
    x = np.asarray(grid, dtype=float)
    x1 = np.asarray(left_endpoints, dtype=float)
    if x.ndim != 1 or x.size == 0 or x1.ndim != 1 or x1.size == 0:
        raise ValueError("grid and left_endpoints must be nonempty 1-D arrays")
    if not (np.isfinite(support_length) and support_length > 0):
        raise ValueError("support_length must be positive")
    if isinstance(power, bool) or int(power) != power or power < 2:
        raise ValueError("power must be an integer >= 2")
    p = int(power)
    ell = float(support_length)
    u = (x[None, :] - x1[:, None]) / ell
    inside = (u > 0.0) & (u < 1.0)
    q = np.where(inside, 4.0 * u * (1.0 - u), 0.0)
    phi = np.where(inside, q ** p, 0.0)
    dphi = np.where(inside, p * q ** (p - 1) * 4.0 * (1.0 - 2.0 * u) / ell, 0.0)
    return np.stack([phi, dphi])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
grid = np.linspace(0.0, 5.0, 101)
left = np.array([-1.0, 0.0, 0.7, 2.5, 4.0])
""",
            "call": 'polynomial_test_functions(grid.copy(), left.copy(), 2.5, 14)',
            "gold_call": '_oracle_polynomial_test_functions(grid.copy(), left.copy(), 2.5, 14)',
        },
        {
            "setup": """import numpy as np
grid = np.array([0.0, 0.1, 0.45, 0.5, 0.55, 0.9, 1.0, 1.3])
left = np.array([0.0, 0.3])
""",
            "call": 'polynomial_test_functions(grid.copy(), left.copy(), 1.0, 2)',
            "gold_call": '_oracle_polynomial_test_functions(grid.copy(), left.copy(), 1.0, 2)',
        },
        {
            "setup": """import numpy as np
grid = np.array([2.0, 2.0000001, 3.25, 4.4999999, 4.5])
left = np.array([2.0])
""",
            "call": 'polynomial_test_functions(grid.copy(), left.copy(), 2.5, 3)',
            "gold_call": '_oracle_polynomial_test_functions(grid.copy(), left.copy(), 2.5, 3)',
        },
        {
            "setup": """import numpy as np
grid = np.arange(501) * 0.05
left = np.linspace(0.0, 12.5, 11)
""",
            "call": 'polynomial_test_functions(grid.copy(), left.copy(), 12.5, 14)',
            "gold_call": '_oracle_polynomial_test_functions(grid.copy(), left.copy(), 12.5, 14)',
        },
        {
            "setup": """import numpy as np
grid = np.linspace(0.0, 1.0, 5)
left = np.array([0.0])
def run_model():
    try:
        polynomial_test_functions(grid.copy(), left.copy(), 1.0, 1)
        return 0
    except ValueError:
        return 1
""",
            "call": 'run_model()',
            "gold_call": '1',
        },
    ]
