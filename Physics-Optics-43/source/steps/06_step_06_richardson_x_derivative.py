"""
Differentiate a scalar function of the generating variables to the orders set by an input photon pattern.

The generating-function representation of a Fock projector requires n derivatives in x for n input photons, followed by division by n! = prod_i n_i!. Because the generating function is a ratio of matrix expressions rather than a closed form, these derivatives are taken numerically.

Derivatives are evaluated with central finite differences on a product stencil, refined by a second-order Richardson extrapolation:

    result(h) = product stencil evaluated at step h,

    answer    = ( 4 * result(h) - result(2h) ) / 3.

Supported orders are n_i in {0, 1, 2}. Order 0 contributes the single point x_i = 0, order 1 uses the two-point central stencil, and order 2 uses the three-point central stencil.

Plain central differences carry an O(h^2) truncation error; the Richardson refinement cancels that leading term and leaves O(h^4), which is what makes six significant figures attainable in double precision. The step size trades truncation against round-off: too large and the O(h^4) remainder dominates, too small and cancellation in the numerator destroys precision, more severely at higher order because the stencil divides by h^n. A useful structural check is that the generating variables enter D and A only at second order, so first derivatives at the origin see only the block that is linear in x.

Returns
-------
complex: the mixed partial derivative of func at x = 0 divided by prod_i factorial(n_i).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def richardson_x_derivative(func, n_pattern: list, h: float = 1e-3) -> complex:
    """Differentiate func at x = 0 to the orders given by n_pattern.
 
    Parameters
    ----------
    func : callable
        Function taking a list of M floats and returning a complex value.
    n_pattern : list
        Sequence of M integers in {0, 1, 2} giving the derivative order in each
        variable.
    h : float, optional
        Base finite-difference step; the extrapolation also evaluates at 2h.
 
    Returns
    -------
    value : complex
        The mixed partial derivative at x = 0 divided by
        prod_i factorial(n_i).
 
    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example ``import itertools``
    and ``from math import factorial``) inside the function body.
    """
    return 0j  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_richardson_x_derivative(func, n_pattern: list, h: float = 1e-3) -> complex:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import itertools
    from math import factorial
 
    if not callable(func):
        raise ValueError("func must be callable")
    if not (isinstance(h, (int, float)) and float(h) > 0.0):
        raise ValueError("h must be a positive number")
    for n in n_pattern:
        if int(n) != n or int(n) not in (0, 1, 2):
            raise ValueError("derivative orders must be 0, 1 or 2")
 
    def stencil_value(step):
        stencils = []
        for n in n_pattern:
            n = int(n)
            if n == 0:
                stencils.append([(0, 1.0)])
            elif n == 1:
                stencils.append([(1, 0.5 / step), (-1, -0.5 / step)])
            else:
                stencils.append([(1, 1.0 / step ** 2),
                                 (0, -2.0 / step ** 2),
                                 (-1, 1.0 / step ** 2)])
        total = 0.0 + 0j
        for combo in itertools.product(*stencils):
            weight = 1.0
            for _, w in combo:
                weight *= w
            total += weight * func([offset * step for offset, _ in combo])
        return total
 
    denom = 1
    for n in n_pattern:
        denom *= factorial(int(n))
    raw = (4.0 * stencil_value(float(h)) - stencil_value(2.0 * float(h))) / 3.0
    return complex(raw / denom)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: second derivative of a known analytic function (normal scenario) ---
        {
            "setup": """import numpy as np
f = lambda x: np.exp(2.0 * x[0]) * np.cos(x[1])
n = [2, 0]
EXPECTED = 4.0 / 2.0
""",
            "call": "bool(abs(richardson_x_derivative(f, n, 1e-3).real - EXPECTED) < 1e-6)",
            "gold_call": 'bool(abs(_oracle_richardson_x_derivative(f, n, 1e-3).real - EXPECTED) < 1e-6)',
        },
        # --- Valid: mixed first derivatives in two variables ---
        {
            "setup": """import numpy as np
f = lambda x: np.exp(0.7 * x[0] + 1.3 * x[1])
n = [1, 1]
EXPECTED = 0.7 * 1.3
""",
            "call": "bool(abs(richardson_x_derivative(f, n, 1e-3).real - EXPECTED) < 1e-6)",
            "gold_call": 'bool(abs(_oracle_richardson_x_derivative(f, n, 1e-3).real - EXPECTED) < 1e-6)',
        },
        # --- Edge: all-zero orders reduce to evaluation at the origin ---
        {
            "setup": """import numpy as np
f = lambda x: 3.5 + 0j
n = [0, 0, 0]
""",
            "call": '(lambda z: np.array([z.real, z.imag], dtype=float))(richardson_x_derivative(f, n, 1e-3))',
            "gold_call": '(lambda z: np.array([z.real, z.imag], dtype=float))(_oracle_richardson_x_derivative(f, n, 1e-3))',
        },
        # --- Boundary: complex-valued integrand with a nontrivial denominator ---
        {
            "setup": """import numpy as np
f = lambda x: np.exp(1j * 0.9 * x[0]) / (1.0 + 0.4 * x[1] ** 2)
n = [2, 0]
""",
            "call": '(lambda z: np.array([z.real, z.imag], dtype=float))(richardson_x_derivative(f, n, 1e-3))',
            "gold_call": '(lambda z: np.array([z.real, z.imag], dtype=float))(_oracle_richardson_x_derivative(f, n, 1e-3))',
        },
    ]
