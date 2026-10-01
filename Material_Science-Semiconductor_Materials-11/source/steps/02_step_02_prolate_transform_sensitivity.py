"""
Evaluate the normalized prolate-window transform and its analytic derivatives.

Inside the prolate band, combine a Legendre reconstruction with the finite-Fourier eigenfunction identity and differentiate the moving argument s/c. Outside the band, differentiate the finite-support cosine quadrature directly.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def prolate_transform_sensitivity(c: float, quadrature_order: int, s_values: "np.ndarray") -> "np.ndarray":
    """Return chi-hat(s), partial chi-hat/partial c, and partial chi-hat/partial s.

    Parameters
    ----------
    c : float
        Positive finite prolate bandwidth.
    quadrature_order : int
        Integer Gauss-Legendre order n, at least 8.
    s_values : np.ndarray
        Nonempty finite one-dimensional array of real transform arguments.

    Returns
    -------
    values : np.ndarray
        Real array of shape (M,3). Columns are chi-hat(s), the derivative with
        respect to c at fixed s, and the derivative with respect to s at fixed c.
        For |s| <= c use the finite-Fourier eigenfunction identity with a
        degree-(n-1) Legendre reconstruction; for |s| > c use direct quadrature.

    Raises
    ------
    ValueError
        If s_values is not a nonempty finite vector or an upstream PSWF input is invalid.
    """
    return values

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_prolate_transform_sensitivity(c: float, quadrature_order: int, s_values: "np.ndarray") -> "np.ndarray":
    s_values = np.asarray(s_values, dtype=float)
    if s_values.ndim != 1 or s_values.size == 0 or not np.all(np.isfinite(s_values)):
        raise ValueError("s_values must be a nonempty finite one-dimensional array")

    packed = _oracle_pswf_bandwidth_sensitivity(c, quadrature_order)
    n = int(quadrature_order)
    nodes = packed[:n]
    weights = packed[n:2*n]
    chi = packed[2*n:3*n]
    chi_c = packed[3*n:4*n]
    eigenvalue = float(packed[-2])
    eigenvalue_c = float(packed[-1])

    vandermonde = np.polynomial.legendre.legvander(nodes, n - 1)
    degrees = np.arange(n, dtype=float)
    factor = 0.5 * (2.0 * degrees + 1.0)
    coefficients = factor * (vandermonde.T @ (weights * chi))
    coefficients_c = factor * (vandermonde.T @ (weights * chi_c))
    coefficients_x = np.polynomial.legendre.legder(coefficients)

    result = np.empty((s_values.size, 3), dtype=float)
    inside = np.abs(s_values) <= float(c)

    if np.any(inside):
        z = s_values[inside] / float(c)
        chi_z = np.polynomial.legendre.legval(z, coefficients)
        chi_c_z = np.polynomial.legendre.legval(z, coefficients_c)
        chi_x_z = np.polynomial.legendre.legval(z, coefficients_x)
        result[inside, 0] = eigenvalue * chi_z
        result[inside, 1] = (
            eigenvalue_c * chi_z
            + eigenvalue * (chi_c_z - chi_x_z * z / float(c))
        )
        result[inside, 2] = eigenvalue * chi_x_z / float(c)

    if np.any(~inside):
        outside_s = s_values[~inside]
        cosine = np.cos(np.outer(outside_s, nodes))
        sine = np.sin(np.outer(outside_s, nodes))
        result[~inside, 0] = cosine @ (weights * chi)
        result[~inside, 1] = cosine @ (weights * chi_c)
        result[~inside, 2] = -(sine @ (weights * chi * nodes))

    zero = s_values == 0.0
    result[zero, :] = np.array([1.0, 0.0, 0.0])
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return cases spanning interior, boundary, and exterior transform arguments."""
    return [
        {"setup":"import numpy as np\nc = 7.15\nquadrature_order = 32\ns_values = np.array([0.0, 1.0, 4.0, 7.15, 9.0])","call":"prolate_transform_sensitivity(c, quadrature_order, s_values.copy())","gold_call":"_oracle_prolate_transform_sensitivity(c, quadrature_order, s_values.copy())","tol":1e-8},
        {"setup":"import numpy as np\nc = 4.0\nquadrature_order = 16\ns_values = np.array([-5.5, -4.0, -0.5, 0.0, 3.0])","call":"prolate_transform_sensitivity(c, quadrature_order, s_values.copy())","gold_call":"_oracle_prolate_transform_sensitivity(c, quadrature_order, s_values.copy())","tol":1e-8},
        {"setup":"import numpy as np\nc = 10.0\nquadrature_order = 24\ns_values = np.array([2.0, 9.5, 10.0, 10.5, 15.0])","call":"prolate_transform_sensitivity(c, quadrature_order, s_values.copy())","gold_call":"_oracle_prolate_transform_sensitivity(c, quadrature_order, s_values.copy())","tol":1e-8},
    ]
