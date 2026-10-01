"""
Applies the scaled half-line quadrature rule to the test function f(u) = exp(-a*u). The rule uses the nodes and weights of the classical rule associated with the exponential weight of scale sigma on (0, infinity), applied so that the returned value approximates the full half-line integral of f multiplied by two (the doubling that the pricing formula absorbs into the quadrature). This step validates node/weight rescaling and the doubling convention. Deliberately excluded: the Fourier integrand, any model content, and any error estimation.

The valuation integral is an even-integrand integral over the real line, reduced to the half line and doubled. A quadrature rule with an exponential weight of freely chosen scale integrates it efficiently once the scale matches the integrand decay. Because the rule is exact for the weight function itself, applying the rule with scale sigma to f(u) = exp(-sigma u) returns exactly 2/sigma at every order N — a convention-pinning invariant. For sigma = 6.254084328929, n_quad = 16, a = 2.0 this step returns 0.999999998 (the exact half-line value, twice the integral of exp(-2u), is 1).

Returns
-------
float — the scaled-rule quadrature value approximating 2*integral_0^infty exp(-a u) du (a single scalar).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scaled_laguerre_exponential(sigma: float, n_quad: int, a: float) -> float:
    """Scaled half-line quadrature applied to f(u) = exp(-a*u).

    Parameters
    ----------
    sigma : float
        Scale of the exponential quadrature weight, sigma > 0.
    n_quad : int
        Number of quadrature nodes, n_quad >= 1.
    a : float
        Decay rate of the integrand, a > 0.

    Returns
    -------
    float
        The quadrature approximation of 2 * integral_0^inf exp(-a*u) du.

    Raises
    ------
    ValueError
        If sigma <= 0, a <= 0, or n_quad < 1.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_scaled_laguerre_exponential(sigma, n_quad, a):
    import numpy as np
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if a <= 0.0:
        raise ValueError("a must be positive")
    n_quad = int(n_quad)
    if n_quad < 1:
        raise ValueError("n_quad must be >= 1")
    x, w = np.polynomial.laguerre.laggauss(n_quad)
    u_n = x / sigma
    w_n = w / sigma
    return 2.0 * float(np.sum(w_n * np.exp(x) * np.exp(-a * u_n)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "scaled_laguerre_exponential(6.254084328929, 16, 2.0)",
            "gold_call": "_oracle_scaled_laguerre_exponential(6.254084328929, 16, 2.0)",
        },
        {
            "setup": "",
            "call": "scaled_laguerre_exponential(0.9, 8, 2.5)",
            "gold_call": "_oracle_scaled_laguerre_exponential(0.9, 8, 2.5)",
        },
        {
            "setup": "",
            "call": "scaled_laguerre_exponential(1.25, 6, 1.25)",
            "gold_call": "_oracle_scaled_laguerre_exponential(1.25, 6, 1.25)",
        },
    ]
