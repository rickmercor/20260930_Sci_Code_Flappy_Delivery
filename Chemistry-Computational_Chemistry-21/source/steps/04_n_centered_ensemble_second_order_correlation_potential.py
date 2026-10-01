"""
Return the second-order (PT2) correlation potential of the

N-centered dimer ensemble at a given site-0 occupation and set of weights,

and its derivatives with respect to both weights at fixed occupation.

Görling-Levy perturbation theory expands the

correlation energy in powers of the interaction strength at fixed density.

Its second-order term (PT2) is the leading correlation contribution in the

weakly correlated regime. For ensembles the expansion is carried out at fixed

density and fixed weights, so the PT2 term becomes weight dependent. For the

Hubbard dimer it can be obtained in closed form within the published

N-centered formulation. Its potential follows the site-occupation (SOFT) sign

convention, in which a potential is minus the occupation derivative of the

corresponding energy.

Returns
-------
a tuple ``(v_c, dv_c_dxi_plus, dv_c_dxi_minus)`` of three     floats. Raise ``ValueError`` if ``t <= 0``, if the weights are not an     admissible N-centered weight set, or if ``n`` does not lie strictly inside     the range representable with these weights.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ensemble_pt2_correlation_potential(t: float, U: float, n: float, xi_plus: float, xi_minus: float) -> tuple[float, float, float]:
    """Return the N-centered ensemble PT2 correlation potential and its weight derivatives.

    Expected return: a tuple ``(v_c, dv_c_dxi_plus, dv_c_dxi_minus)`` of three
    floats. Raise ``ValueError`` if ``t <= 0``, if the weights are not an
    admissible N-centered weight set, or if ``n`` does not lie strictly inside
    the range representable with these weights.
    """
    return (0.0, 0.0, 0.0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_ensemble_pt2_correlation_potential(t: float, U: float, n: float, xi_plus: float, xi_minus: float) -> tuple[float, float, float]:
    if t <= 0.0:
        raise ValueError("The hopping t must be positive.")
    if xi_plus < 0.0 or xi_minus < 0.0 or 3.0 * xi_plus + xi_minus > 2.0:
        raise ValueError("The weights are not an admissible N-centered weight set.")
    m = n - 1.0
    x = 1.0 - xi_plus
    if abs(m) >= x:
        raise ValueError("The occupation must lie strictly inside the representable range.")

    # G(n, xi) = d^2 F^xi / dU^2 at U = 0 = a * b * s**3, with the factors below.
    a = (2.0 - xi_minus - 3.0 * xi_plus) / (16.0 * t)
    k = 1.0 - 2.0 * xi_minus - 3.0 * xi_plus
    s2 = 1.0 - m * m / (x * x)
    s = s2**0.5
    b = m * m * k / x**3 - 1.0

    # dG/dn = a * m * s * h.
    h = 2.0 * k * s2 / x**3 - 3.0 * b / x**2
    dg_dn = a * m * s * h

    # Weight derivatives of dG/dn at fixed n.
    h_minus = -4.0 * s2 / x**3 + 6.0 * m * m / x**5
    dg_dn_minus = -m * s * h / (16.0 * t) + a * m * s * h_minus
    s_plus = -m * m / (x**3 * s)
    b_plus = m * m * (-3.0 / x**3 + 3.0 * k / x**4)
    h_plus = -6.0 * s2 / x**3 - 4.0 * k * m * m / x**6 + 6.0 * k * s2 / x**4 - 3.0 * b_plus / x**2 - 6.0 * b / x**3
    dg_dn_plus = m * (-3.0 * s * h / (16.0 * t) + a * s_plus * h + a * s * h_plus)

    prefactor = -0.5 * U * U
    return (prefactor * dg_dn, prefactor * dg_dn_plus, prefactor * dg_dn_minus)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return zero-weight, symmetric-density and nonzero-weight cases on both sides of n = 1."""
    return [
        # Normal: zero weights, occupation above 1.
        {
            "setup": "",
            "call": "ensemble_pt2_correlation_potential(1.0, 2.5, 1.25, 0.0, 0.0)",
            "gold_call": "_oracle_ensemble_pt2_correlation_potential(1.0, 2.5, 1.25, 0.0, 0.0)",
        },
        # Boundary: symmetric occupation.
        {
            "setup": "",
            "call": "ensemble_pt2_correlation_potential(1.0, 1.0, 1.0, 0.1, 0.1)",
            "gold_call": "_oracle_ensemble_pt2_correlation_potential(1.0, 1.0, 1.0, 0.1, 0.1)",
        },
        # Normal: both weights nonzero, occupation above 1.
        {
            "setup": "",
            "call": "ensemble_pt2_correlation_potential(1.0, 1.5, 1.3, 0.1, 0.05)",
            "gold_call": "_oracle_ensemble_pt2_correlation_potential(1.0, 1.5, 1.3, 0.1, 0.05)",
        },
        # Normal: occupation below 1, addition weight only, hopping different from 1.
        {
            "setup": "",
            "call": "ensemble_pt2_correlation_potential(2.0, 2.0, 0.8, 0.2, 0.0)",
            "gold_call": "_oracle_ensemble_pt2_correlation_potential(2.0, 2.0, 0.8, 0.2, 0.0)",
        },
        # Edge: removal weight only.
        {
            "setup": "",
            "call": "ensemble_pt2_correlation_potential(1.0, 3.0, 1.1, 0.0, 0.15)",
            "gold_call": "_oracle_ensemble_pt2_correlation_potential(1.0, 3.0, 1.1, 0.0, 0.15)",
        },
    ]
