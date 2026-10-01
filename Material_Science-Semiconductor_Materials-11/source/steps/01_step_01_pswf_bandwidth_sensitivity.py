"""
Construct the unit-integral prolate window samples and their analytic bandwidth sensitivity.

The real-even zeroth-order PSWF is discretized with a symmetric Gauss-Legendre Nyström matrix. Differentiate the symmetric eigenproblem with respect to c, use simple-eigenvalue perturbation theory for the leading eigenvector, then differentiate the unit-integral normalization.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pswf_bandwidth_sensitivity(c: float, quadrature_order: int) -> "np.ndarray":
    """Return the unit-integral window samples and their derivative with respect to c.

    Parameters
    ----------
    c : float
        Positive finite prolate bandwidth.
    quadrature_order : int
        Integer Gauss-Legendre order n, at least 8.

    Returns
    -------
    packed : np.ndarray
        One-dimensional array of length 4*n+2. Blocks are Gauss-Legendre nodes,
        weights, the unit-integral window samples chi = psi_0^c / int_{-1}^{1} psi_0^c
        (so that sum_j w_j chi_j = 1), dchi/dc, then the leading
        eigenvalue lambda_0 and dlambda_0/dc, where lambda_0 is the leading
        eigenvalue of the even finite-Fourier (cosine-kernel) operator,
        int_{-1}^{1} cos(c*x*t) chi(t) dt = lambda_0 * chi(x).

    Raises
    ------
    ValueError
        If c is not positive and finite or quadrature_order is not an integer at least 8.
    RuntimeError
        If the leading discrete eigenvalue is not numerically simple.
    """
    return packed

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pswf_bandwidth_sensitivity(c: float, quadrature_order: int) -> "np.ndarray":
    c = float(c)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError("c must be positive and finite")
    if isinstance(quadrature_order, bool) or int(quadrature_order) != quadrature_order:
        raise ValueError("quadrature_order must be an integer")
    n = int(quadrature_order)
    if n < 8:
        raise ValueError("quadrature_order must be at least 8")

    nodes, weights = np.polynomial.legendre.leggauss(n)
    sqrt_w = np.sqrt(weights)
    xx = np.outer(nodes, nodes)
    matrix = sqrt_w[:, None] * np.cos(c * xx) * sqrt_w[None, :]
    derivative_matrix = sqrt_w[:, None] * (-xx * np.sin(c * xx)) * sqrt_w[None, :]

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    eigenvalue = float(eigenvalues[0])
    vector = eigenvectors[:, 0].copy()
    integral = float(np.dot(weights, vector / sqrt_w))
    if integral < 0.0:
        vector = -vector
        eigenvectors[:, 0] = vector
        integral = -integral
    if integral == 0.0:
        raise RuntimeError("leading eigenvector has zero weighted integral")

    gaps = eigenvalue - eigenvalues[1:]
    if np.any(np.abs(gaps) <= 1.0e-12):
        raise RuntimeError("leading discrete eigenvalue is not simple")

    eigenvalue_derivative = float(vector @ (derivative_matrix @ vector))
    couplings = eigenvectors[:, 1:].T @ (derivative_matrix @ vector)
    vector_derivative = eigenvectors[:, 1:] @ (couplings / gaps)

    psi = vector / sqrt_w
    psi_derivative = vector_derivative / sqrt_w
    integral_derivative = float(np.dot(weights, psi_derivative))

    chi = psi / integral
    chi_derivative = (
        psi_derivative * integral - psi * integral_derivative
    ) / (integral * integral)

    return np.concatenate([
        nodes.astype(float),
        weights.astype(float),
        chi.astype(float),
        chi_derivative.astype(float),
        np.array([eigenvalue, eigenvalue_derivative], dtype=float),
    ])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential cases for the eigenpair sensitivity."""
    return [
        {"setup":"c = 7.15\nquadrature_order = 32","call":"pswf_bandwidth_sensitivity(c, quadrature_order)","gold_call":"_oracle_pswf_bandwidth_sensitivity(c, quadrature_order)","tol":1e-8},
        {"setup":"c = 3.2\nquadrature_order = 12","call":"pswf_bandwidth_sensitivity(c, quadrature_order)","gold_call":"_oracle_pswf_bandwidth_sensitivity(c, quadrature_order)","tol":1e-8},
        {"setup":"c = 10.5\nquadrature_order = 24","call":"pswf_bandwidth_sensitivity(c, quadrature_order)","gold_call":"_oracle_pswf_bandwidth_sensitivity(c, quadrature_order)","tol":1e-8},
    ]
