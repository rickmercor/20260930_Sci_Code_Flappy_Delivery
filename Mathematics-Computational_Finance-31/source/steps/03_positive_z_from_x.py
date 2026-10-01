"""
Return the positive liquidity inverse and normalized sensitivities.



The unique positive inverse $Z=Z(x)$ is defined by

$$x=\left(\sqrt{Z}-\frac{\operatorname{arsinh}(\sqrt{Z})}

{\sqrt{1+Z}}\right)^2,\qquad x>0,\quad Z>0.$$

Also compute the analytic inverse responses

$$E=\frac{xZ'(x)}{Z(x)},\qquad C=\frac{x^2Z''(x)}{Z(x)}.$$

Derivatives refer to this mathematical inverse. Finite differences and

differentiation of numerical solver iterations are not accepted.

Cover the complete positive finite binary64 range, including subnormals,

using ordinary Python float arithmetic and stable evaluation where direct

subtraction cancels. No particular initial guess or solver is required.



With $y=\sqrt{Z}$ and

$g(y)=y-\operatorname{arsinh}(y)/\sqrt{1+y^2}$, acceptance requires

$$|g(y)-\sqrt{x}|\leq\mathrm{residual\_tol}\sqrt{x}.$$

A candidate must pass the residual check within the specified iteration

budget, including the candidate checked at the start of each iteration.

Do not silently relax the tolerance, exceed the budget, or mistake

cancellation or underflow for convergence. Audit the root with a stable

evaluation of the defining relation. All returned quantities must be finite.

The positive liquidity map is flat near its origin, making cancellation-safe inversion necessary. Dimensionless elasticity $E=xZ'/Z$ and curvature $C=x^2Z''/Z$ describe the conditioning of the mathematical inverse without returning potentially extreme unscaled derivatives. The scalar inverse also supplies the baseline for higher-order Taylor lifting.

Returns
-------
return z, elasticity, curvature
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def positive_z_from_x(
    x: float,
    residual_tol: float = 1e-13,
    max_iter: int = 24,
) -> tuple[float, float, float]:
    r"""Return the positive liquidity inverse and normalized sensitivities.

    The unique positive inverse $Z=Z(x)$ is defined by
    $$x=\left(\sqrt{Z}-\frac{\operatorname{arsinh}(\sqrt{Z})}
    {\sqrt{1+Z}}\right)^2,\qquad x>0,\quad Z>0.$$
    Also compute the analytic inverse responses
    $$E=\frac{xZ'(x)}{Z(x)},\qquad C=\frac{x^2Z''(x)}{Z(x)}.$$
    Derivatives refer to this mathematical inverse. Finite differences and
    differentiation of numerical solver iterations are not accepted.
    Cover the complete positive finite binary64 range, including subnormals,
    using ordinary Python float arithmetic and stable evaluation where direct
    subtraction cancels. No particular initial guess or solver is required.

    With $y=\sqrt{Z}$ and
    $g(y)=y-\operatorname{arsinh}(y)/\sqrt{1+y^2}$, acceptance requires
    $$|g(y)-\sqrt{x}|\leq\mathrm{residual\_tol}\sqrt{x}.$$
    A candidate must pass the residual check within the specified iteration
    budget, including the candidate checked at the start of each iteration.
    Do not silently relax the tolerance, exceed the budget, or mistake
    cancellation or underflow for convergence. Audit the root with a stable
    evaluation of the defining relation. All returned quantities must be finite.

    Parameters
    ----------
    x : float
        Finite positive binary64 liquidity argument $x$, including subnormals.
    residual_tol : float, default 1e-13
        Finite relative tolerance $0<\mathrm{residual\_tol}<1$ for the
        unsquared transformed relation. It is not an absolute tolerance on $x$.
    max_iter : int, default 24
        Positive integer budget $\mathrm{max\_iter}\geq1$ for nonlinear
        iterations with residual checks. Unattainable tolerance may fail.

    Returns
    -------
    tuple[float, float, float]
        Exactly $(Z,E,C)$, where $Z>0$ and all entries are finite.
        The last two outputs are normalized responses, not $Z'$ and $Z''$.

    Raises
    ------
    ValueError
        If $x$ is nonfinite or nonpositive, the tolerance is nonfinite or
        outside the open interval $(0,1)$, or the budget is not a positive integer.
    ArithmeticError
        If no acceptable root is found within the budget or finite root and
        response values cannot be produced. Do not return an unconverged result.
    """
    z = 0.0
    elasticity = 0.0
    curvature = 0.0
    return z, elasticity, curvature

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_positive_z_from_x(
    x: float,
    residual_tol: float = 1e-13,
    max_iter: int = 24,
) -> tuple[float, float, float]:
    import math

    x = float(x)
    if not math.isfinite(x) or x <= 0.0:
        raise ValueError("x must be finite and positive")
    if not math.isfinite(residual_tol) or not 0.0 < residual_tol < 1.0:
        raise ValueError("residual_tol must lie in (0, 1)")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    target = math.sqrt(x)
    coefficients = (
        2.0 / 3.0,
        -8.0 / 15.0,
        16.0 / 35.0,
        -128.0 / 315.0,
        256.0 / 693.0,
        -1024.0 / 3003.0,
        2048.0 / 6435.0,
    )

    def _g_jet(y: float) -> tuple[float, float, float]:
        if y < 0.05:
            g = math.fsum(
                coefficient * y ** (2 * index + 3)
                for index, coefficient in enumerate(coefficients)
            )
            dg = math.fsum(
                (2 * index + 3) * coefficient * y ** (2 * index + 2)
                for index, coefficient in enumerate(coefficients)
            )
            d2g = math.fsum(
                (2 * index + 3)
                * (2 * index + 2)
                * coefficient
                * y ** (2 * index + 1)
                for index, coefficient in enumerate(coefficients)
            )
            return g, dg, d2g

        scale = math.hypot(1.0, y)
        inverse_scale = 1.0 / scale
        ratio = y * inverse_scale
        inverse_hyperbolic = math.asinh(y)
        g = y - inverse_hyperbolic * inverse_scale
        dg = ratio * ratio + inverse_hyperbolic * ratio * inverse_scale * inverse_scale
        d2g = inverse_scale ** 3 * (
            3.0 * ratio
            + inverse_hyperbolic
            * (inverse_scale * inverse_scale - 2.0 * ratio * ratio)
        )
        return g, dg, d2g

    span = max(1.0, math.ulp(target))
    lower, upper = target, target + span
    initial = math.exp((math.log(target) + math.log(1.5)) / 3.0)
    y = min(max(initial, lower), upper)

    for _ in range(max_iter):
        g, dg, d2g = _g_jet(y)
        residual = g - target
        if abs(residual) <= residual_tol * target:
            z = y * y
            elasticity = target / (y * dg)
            curvature = 0.5 * elasticity * (
                elasticity - 1.0 - elasticity * y * d2g / dg
            )
            result = (z, elasticity, curvature)
            if all(math.isfinite(value) for value in result):
                return result
            break

        if residual < 0.0:
            lower = y
        else:
            upper = y

        trial = y - residual / dg if dg > 0.0 else math.nan
        if not (math.isfinite(trial) and lower < trial < upper):
            trial = 0.5 * (lower + upper)
        y = trial

    raise ArithmeticError("positive-root solve did not meet the residual tolerance")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    audit_setup = """
import math

def _audit_positive_z(x, result, z_scale, residual_tol):
    z, elasticity, curvature = result
    if not (
        math.isfinite(z)
        and math.isfinite(elasticity)
        and math.isfinite(curvature)
        and z > 0.0
    ):
        return (math.nan, math.nan, math.nan, False)

    y = math.sqrt(z)
    if y < 0.05:
        coefficients = (
            2.0 / 3.0,
            -8.0 / 15.0,
            16.0 / 35.0,
            -128.0 / 315.0,
            256.0 / 693.0,
            -1024.0 / 3003.0,
            2048.0 / 6435.0,
        )
        g = math.fsum(
            coefficient * y ** (2 * index + 3)
            for index, coefficient in enumerate(coefficients)
        )
    else:
        g = y - math.asinh(y) / math.hypot(1.0, y)

    relative_residual = abs(g * g - x) / x
    return (
        z / z_scale,
        elasticity,
        curvature,
        relative_residual <= 8.0 * residual_tol,
    )
"""
    return [
        {
            "setup": audit_setup,
            "call": "_audit_positive_z(5e-324,positive_z_from_x(5e-324,2e-15,3),1e-108,2e-15)",
            "gold_call": "_audit_positive_z(5e-324,_oracle_positive_z_from_x(5e-324,2e-15,3),1e-108,2e-15)",
            "tol": 2e-13,
        },
        {
            "setup": audit_setup,
            "call": "_audit_positive_z(6.916753717622085e-9,positive_z_from_x(6.916753717622085e-9,2e-15,5),0.0025,2e-15)",
            "gold_call": "_audit_positive_z(6.916753717622085e-9,_oracle_positive_z_from_x(6.916753717622085e-9,2e-15,5),0.0025,2e-15)",
            "tol": 5e-14,
        },
        {
            "setup": audit_setup,
            "call": "_audit_positive_z(0.0172873290940766,positive_z_from_x(0.0172873290940766,2e-15,5),1.0,2e-15)",
            "gold_call": "_audit_positive_z(0.0172873290940766,_oracle_positive_z_from_x(0.0172873290940766,2e-15,5),1.0,2e-15)",
            "tol": 5e-14,
        },
        {
            "setup": audit_setup,
            "call": "_audit_positive_z(1e12,positive_z_from_x(1e12,2e-15,3),1e12,2e-15)",
            "gold_call": "_audit_positive_z(1e12,_oracle_positive_z_from_x(1e12,2e-15,3),1e12,2e-15)",
            "tol": 5e-14,
        },
        {
            "setup": audit_setup,
            "call": "_audit_positive_z(1.7976931348623157e308,positive_z_from_x(1.7976931348623157e308,2e-15,3),1.7976931348623157e308,2e-15)",
            "gold_call": "_audit_positive_z(1.7976931348623157e308,_oracle_positive_z_from_x(1.7976931348623157e308,2e-15,3),1.7976931348623157e308,2e-15)",
            "tol": 5e-14,
        },
    ]
