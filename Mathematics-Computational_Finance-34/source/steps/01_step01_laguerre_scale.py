"""
Computes the scale parameter of the exponential quadrature weight used for the half-line Fourier integral. The scale is matched to the estimated asymptotic exponential decay rate of the damped Fourier integrand implied by the model parameters and the maturity, following the source's prescription. This step validates the closed-form scale formula only; it performs no time stepping and no quadrature. Deliberately excluded: node construction, integrand evaluation, and any error estimation.

The magnitude of the rough Heston characteristic function along a damped Fourier contour is conjectured to decay exponentially in the integration variable, at a rate set by the model parameters and the maturity. Matching the quadrature weight's exponential scale to that decay places the nodes where the integrand actually carries mass. For the frozen benchmark configuration (alpha = 0.66, gamma = 0.50, nu = 0.12, rho = -0.55, V0 = 0.07, theta = 0.35, T = 2.0) this step returns 6.254084328929. A useful exact special case: with v0 = 0, gam = 0.35, nu = 0.25, rho = 0.6, theta = 0.2, big_t = 1.0 the value is exactly 0.64.

Returns
-------
float — the exponential weight scale of the scaled half-line quadrature rule (a single positive scalar).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def laguerre_scale(alpha: float, gam: float, nu: float, rho: float,
                   v0: float, theta: float, big_t: float) -> float:
    """Scale of the exponential quadrature weight for the Fourier integral.

    Parameters
    ----------
    alpha : float
        Roughness index of the fractional kernel, 0.5 < alpha < 1.
    gam : float
        Mean-reversion speed of the variance process, gam > 0.
    nu : float
        Vol-of-vol scale parameter, nu > 0 (the vol-of-vol is gam * nu).
    rho : float
        Spot-variance correlation, -1 < rho < 1.
    v0 : float
        Initial variance, v0 >= 0.
    theta : float
        Long-run variance level, theta >= 0.
    big_t : float
        Maturity in years, big_t > 0.

    Returns
    -------
    float
        The exponential weight scale used by the scaled quadrature rule.

    Raises
    ------
    ValueError
        If any parameter is outside its admissible range.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_laguerre_scale(alpha, gam, nu, rho, v0, theta, big_t):
    from math import sqrt, gamma as _gamma_fn
    if not (0.5 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0.5 < alpha < 1")
    if gam <= 0.0 or nu <= 0.0:
        raise ValueError("gam and nu must be positive")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must satisfy -1 < rho < 1")
    if v0 < 0.0 or theta < 0.0:
        raise ValueError("v0 and theta must be non-negative")
    if big_t <= 0.0:
        raise ValueError("big_t must be positive")
    return (sqrt(1.0 - rho * rho) / (gam * nu)
            * (gam * theta * big_t
               + v0 * big_t ** (1.0 - alpha) / _gamma_fn(2.0 - alpha)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "laguerre_scale(0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 2.0)",
            "gold_call": "_oracle_laguerre_scale(0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 2.0)",
        },
        {
            "setup": "",
            "call": "laguerre_scale(0.75, 0.5, 0.3, -0.7, 0.04, 0.15, 0.5)",
            "gold_call": "_oracle_laguerre_scale(0.75, 0.5, 0.3, -0.7, 0.04, 0.15, 0.5)",
        },
        {
            "setup": "",
            "call": "laguerre_scale(0.8, 0.35, 0.25, 0.6, 0.0, 0.2, 1.0)",
            "gold_call": "_oracle_laguerre_scale(0.8, 0.35, 0.25, 0.6, 0.0, 0.2, 1.0)",
        },
    ]
