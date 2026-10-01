"""
Return the ensemble exact-exchange Hartree-exchange

potential of the N-centered dimer ensemble at a given site-0 occupation and

set of weights, and its derivatives with respect to both weights at fixed

occupation.

In ensemble DFT the Hartree-exchange-correlation

energy depends on the ensemble weights as well as on the density. The

exact-exchange level keeps the part of the ensemble universal functional that

is first order in the interaction strength, at fixed density and weights. For

the Hubbard dimer this ensemble exact-exchange (EEXX) functional can be

obtained in closed form within the published N-centered formulation. Its

potential follows the site-occupation (SOFT) sign convention, in which a

potential is minus the occupation derivative of the corresponding energy.

Returns
-------
a tuple ``(v_Hx, dv_Hx_dxi_plus, dv_Hx_dxi_minus)`` of     three floats. Raise ``ValueError`` if ``t <= 0``, if the weights are not     an admissible N-centered weight set, or if ``n`` is not representable with     these weights.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ensemble_eexx_hx_potential(t: float, U: float, n: float, xi_plus: float, xi_minus: float) -> tuple[float, float, float]:
    """Return the N-centered ensemble EEXX Hartree-exchange potential and its weight derivatives.

    Expected return: a tuple ``(v_Hx, dv_Hx_dxi_plus, dv_Hx_dxi_minus)`` of
    three floats. Raise ``ValueError`` if ``t <= 0``, if the weights are not
    an admissible N-centered weight set, or if ``n`` is not representable with
    these weights.
    """
    return (0.0, 0.0, 0.0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_ensemble_eexx_hx_potential(t: float, U: float, n: float, xi_plus: float, xi_minus: float) -> tuple[float, float, float]:
    if t <= 0.0:
        raise ValueError("The hopping t must be positive.")
    if xi_plus < 0.0 or xi_minus < 0.0 or 3.0 * xi_plus + xi_minus > 2.0:
        raise ValueError("The weights are not an admissible N-centered weight set.")
    central_weight = 1.0 - 0.5 * (3.0 * xi_plus + xi_minus)
    offset = n - 1.0
    scale = 1.0 - xi_plus
    if abs(offset) > scale:
        raise ValueError("The occupation is not ensemble representable.")

    potential = -U * central_weight * offset / (scale * scale)
    d_plus = -U * offset * (-1.5 / (scale * scale) + 2.0 * central_weight / scale**3)
    d_minus = 0.5 * U * offset / (scale * scale)
    return (potential, d_plus, d_minus)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return zero-weight, symmetric-density and nonzero-weight cases."""
    return [
        # Normal: zero weights, occupation above 1.
        {
            "setup": "",
            "call": "ensemble_eexx_hx_potential(1.0, 2.5, 1.25, 0.0, 0.0)",
            "gold_call": "_oracle_ensemble_eexx_hx_potential(1.0, 2.5, 1.25, 0.0, 0.0)",
        },
        # Boundary: symmetric occupation.
        {
            "setup": "",
            "call": "ensemble_eexx_hx_potential(1.0, 1.0, 1.0, 0.1, 0.2)",
            "gold_call": "_oracle_ensemble_eexx_hx_potential(1.0, 1.0, 1.0, 0.1, 0.2)",
        },
        # Normal: both weights nonzero.
        {
            "setup": "",
            "call": "ensemble_eexx_hx_potential(1.0, 1.5, 1.3, 0.1, 0.05)",
            "gold_call": "_oracle_ensemble_eexx_hx_potential(1.0, 1.5, 1.3, 0.1, 0.05)",
        },
        # Normal: occupation below 1, addition weight only, hopping different from 1.
        {
            "setup": "",
            "call": "ensemble_eexx_hx_potential(2.0, 2.0, 0.7, 0.2, 0.0)",
            "gold_call": "_oracle_ensemble_eexx_hx_potential(2.0, 2.0, 0.7, 0.2, 0.0)",
        },
        # Edge: removal weight only, large U, small hopping.
        {
            "setup": "",
            "call": "ensemble_eexx_hx_potential(0.5, 3.0, 1.4, 0.0, 0.3)",
            "gold_call": "_oracle_ensemble_eexx_hx_potential(0.5, 3.0, 1.4, 0.0, 0.3)",
        },
    ]
