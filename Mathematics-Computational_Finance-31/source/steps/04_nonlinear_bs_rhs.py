"""
Evaluate the specified nonlinear Black-Scholes local time derivative.



Use the transformed time-to-maturity convention

$$F=\frac12\sigma_0^2(1+z)X^2w_{xx}+(r-q)Xw_x-rw_0.$$

The liquidity correction and spatial derivatives are already supplied.

Evaluate this expression without intermediate rounding.

The transformed time-to-maturity pricing equation combines diffusion $\tfrac12\sigma_0^2(1+z)X^2w_{xx}$, carry $(r-q)Xw_x$, and discounting $-rw_0$. The liquidity correction changes the diffusion coefficient; the dividend yield appears in the carry term.

Returns
-------
return rhs
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nonlinear_bs_rhs(
    wx: float,
    wxx: float,
    w0: float,
    X: float,
    r: float,
    q: float,
    sigma0: float,
    z: float,
) -> float:
    r"""Evaluate the specified nonlinear Black-Scholes local time derivative.

    Use the transformed time-to-maturity convention
    $$F=\frac12\sigma_0^2(1+z)X^2w_{xx}+(r-q)Xw_x-rw_0.$$
    The liquidity correction and spatial derivatives are already supplied.
    Evaluate this expression without intermediate rounding.

    Parameters
    ----------
    wx : float
        Finite local first derivative $w_x$; either sign is allowed.
    wxx : float
        Finite local second derivative $w_{xx}$; either sign is allowed.
    w0 : float
        Finite central option value $w_0$.
    X : float
        Finite underlying coordinate $X$, normally positive in the model.
    r : float
        Finite interest rate $r$.
    q : float
        Finite continuous dividend yield $q$.
    sigma0 : float
        Finite baseline volatility $\sigma_0$, squared in the expression.
    z : float
        Finite precomputed liquidity correction $z$; the financial positive
        branch has $z\geq0$, including the zero-liquidity limit.
        This algebraic evaluator does not enforce financial consistency.
        Inputs must keep all products and the sum finite in binary64 arithmetic.

    Returns
    -------
    float
        The unrounded local rate $F$, not an updated option value or a
        sensitivity coefficient. Its discount contribution is $-rw_0$.

    Raises
    ------
    No exception is required for inputs in the stated valid domain.
    Outside that domain, validation behavior is unspecified and ordinary
    Python arithmetic or type exceptions may propagate.
    """
    rhs = 0.0
    return rhs

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_nonlinear_bs_rhs(
    wx: float,
    wxx: float,
    w0: float,
    X: float,
    r: float,
    q: float,
    sigma0: float,
    z: float,
) -> float:
    diffusion = 0.5 * sigma0 ** 2 * (1 + z) * X ** 2 * wxx
    drift = (r - q) * X * wx
    discount = -r * w0
    return diffusion + drift + discount

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "nonlinear_bs_rhs(0.7332311111111111, 0.015642222222222222, 14.6457, 100.0, 0.1, 0.0, 0.2, 0.4076834105977857)",
            "gold_call": "_oracle_nonlinear_bs_rhs(0.7332311111111111, 0.015642222222222222, 14.6457, 100.0, 0.1, 0.0, 0.2, 0.4076834105977857)",
        },
        {
            "setup": "",
            "call": "nonlinear_bs_rhs(-0.25, 0.03, 11.0, 80.0, 0.04, 0.01, 0.3, 0.2)",
            "gold_call": "_oracle_nonlinear_bs_rhs(-0.25, 0.03, 11.0, 80.0, 0.04, 0.01, 0.3, 0.2)",
        },
        {
            "setup": "",
            "call": "nonlinear_bs_rhs(0.0, 0.0, 7.5, 50.0, 0.0, 0.0, 0.2, 0.0)",
            "gold_call": "_oracle_nonlinear_bs_rhs(0.0, 0.0, 7.5, 50.0, 0.0, 0.0, 0.2, 0.0)",
        },
    ]
