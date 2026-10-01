"""
Apply two supplied derivative-weight triples to three nodal values.



In the common node order $(-,0,+)$, compute

$$w_x=a_-W_-+a_0W_0+a_+W_+,\qquad

w_{xx}=b_-W_-+b_0W_0+b_+W_+.$$

The weights already include all mesh-spacing factors. Apply them exactly

as supplied, without intermediate rounding or additional division.

In the transformed pricing equation, $w_x$ is local Delta and $w_{xx}$ is local Gamma. This step performs the two specified weighted sums. The supplied weights already contain all stencil scaling and may be supplied independently of the RBF-FD rule.

Returns
-------
return (wx, wxx)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spatial_derivatives(
    values: tuple[float, float, float],
    first_weights: tuple[float, float, float],
    second_weights: tuple[float, float, float],
) -> tuple[float, float]:
    r"""Apply two supplied derivative-weight triples to three nodal values.

    In the common node order $(-,0,+)$, compute
    $$w_x=a_-W_-+a_0W_0+a_+W_+,\qquad
    w_{xx}=b_-W_-+b_0W_0+b_+W_+.$$
    The weights already include all mesh-spacing factors. Apply them exactly
    as supplied, without intermediate rounding or additional division.

    Parameters
    ----------
    values : tuple[float, float, float]
        Exactly three finite nodal values $(W_-,W_0,W_+)$.
    first_weights : tuple[float, float, float]
        Exactly three finite first-derivative weights $(a_-,a_0,a_+)$.
    second_weights : tuple[float, float, float]
        Exactly three finite second-derivative weights $(b_-,b_0,b_+)$.
        Signed inputs and zero outputs are permitted. Inputs must keep
        products and sums finite in binary64 arithmetic.

    Returns
    -------
    tuple[float, float]
        The pair $(w_x,w_{xx})$: local Delta followed by local Gamma.

    Raises
    ------
    No exception is required for inputs in the stated valid domain.
    Outside that domain, validation behavior is unspecified and ordinary
    Python arithmetic or type exceptions may propagate.
    """
    wx = 0.0
    wxx = 0.0
    return (wx, wxx)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_spatial_derivatives(
    values: tuple[float, float, float],
    first_weights: tuple[float, float, float],
    second_weights: tuple[float, float, float],
) -> tuple[float, float]:
    wx = sum(weight * value for weight, value in zip(first_weights, values))
    wxx = sum(weight * value for weight, value in zip(second_weights, values))
    return wx, wxx

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "spatial_derivatives((8.4032, 14.6457, 22.2960), (-0.05277777777777778, 0.0, 0.05277777777777778), (0.011111111111111111, -0.022222222222222222, 0.011111111111111111))",
            "gold_call": "_oracle_spatial_derivatives((8.4032, 14.6457, 22.2960), (-0.05277777777777778, 0.0, 0.05277777777777778), (0.011111111111111111, -0.022222222222222222, 0.011111111111111111))",
        },
        {
            "setup": "",
            "call": "spatial_derivatives((1.0, 4.0, 9.0), (-0.1, 0.0, 0.1), (0.01, -0.02, 0.01))",
            "gold_call": "_oracle_spatial_derivatives((1.0, 4.0, 9.0), (-0.1, 0.0, 0.1), (0.01, -0.02, 0.01))",
        },
        {
            "setup": "",
            "call": "spatial_derivatives((0.0, 0.0, 0.0), (-0.5, 0.0, 0.5), (1.0, -2.0, 1.0))",
            "gold_call": "_oracle_spatial_derivatives((0.0, 0.0, 0.0), (-0.5, 0.0, 0.5), (1.0, -2.0, 1.0))",
        },
    ]
