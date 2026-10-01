"""
Approximate the zero-weight derivatives of the N-centered

ensemble Hxc potential with respect to the two ensemble weights, using the

published double-scaling approximation at the second-order (PT2) level.

Regular ground-state density functionals carry no

weight dependence, so on their own they cannot supply the weight derivatives

that N-centered ensemble DFT needs. Scaled ensemble approximations recover

that dependence with weight-dependent scaling functions that equal 1 at zero

weight. The double-scaling variant treats the Hartree-exchange and

correlation parts with separate scaling functions. At the PT2 level they are

built from the ensemble exact-exchange and ensemble second-order functionals

of the preceding steps, and every regular ground-state ingredient stays

exact.

Returns
-------
a tuple ``(w_plus, w_minus)`` of two floats. Raise     ``ValueError`` if ``U <= 0``, or if the ground-state occupation equals 1     (symmetric dimer), where the approximation is undefined.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pt2_double_scaled_weight_derivatives(t: float, U: float, dv: float) -> tuple[float, float]:
    """Return the PT2 double-scaled weight derivatives of the ensemble Hxc potential.

    Expected return: a tuple ``(w_plus, w_minus)`` of two floats. Raise
    ``ValueError`` if ``U <= 0``, or if the ground-state occupation equals 1
    (symmetric dimer), where the approximation is undefined.
    """
    return (0.0, 0.0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_pt2_double_scaled_weight_derivatives(t: float, U: float, dv: float) -> tuple[float, float]:
    if U <= 0.0:
        raise ValueError("The approximation needs U > 0.")
    n0, chi, f_hxc, v_hx, v_c = _oracle_exact_ground_state_hxc(t, U, dv)
    if abs(n0 - 1.0) < 1e-12:
        raise ValueError("The approximation is undefined for the symmetric dimer.")

    hx_zero, hx_plus, hx_minus = _oracle_ensemble_eexx_hx_potential(t, U, n0, 0.0, 0.0)
    c_zero, c_plus, c_minus = _oracle_ensemble_pt2_correlation_potential(t, U, n0, 0.0, 0.0)
    w_plus = (hx_plus / hx_zero) * v_hx + (c_plus / c_zero) * v_c
    w_minus = (hx_minus / hx_zero) * v_hx + (c_minus / c_zero) * v_c
    return (w_plus, w_minus)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return canonical and further asymmetric, interacting dimers."""
    return [
        # Normal: canonical fixture.
        {
            "setup": "",
            "call": "pt2_double_scaled_weight_derivatives(1.0, 2.5, 1.5)",
            "gold_call": "_oracle_pt2_double_scaled_weight_derivatives(1.0, 2.5, 1.5)",
        },
        # Normal: moderate correlation, strong asymmetry.
        {
            "setup": "",
            "call": "pt2_double_scaled_weight_derivatives(1.0, 1.5, 3.0)",
            "gold_call": "_oracle_pt2_double_scaled_weight_derivatives(1.0, 1.5, 3.0)",
        },
        # Normal: intermediate correlation.
        {
            "setup": "",
            "call": "pt2_double_scaled_weight_derivatives(1.0, 2.0, 2.4)",
            "gold_call": "_oracle_pt2_double_scaled_weight_derivatives(1.0, 2.0, 2.4)",
        },
        # Edge: negative dv, hopping different from 1.
        {
            "setup": "",
            "call": "pt2_double_scaled_weight_derivatives(2.0, 3.0, -1.0)",
            "gold_call": "_oracle_pt2_double_scaled_weight_derivatives(2.0, 3.0, -1.0)",
        },
        # Boundary: weak correlation and strong asymmetry.
        {
            "setup": "",
            "call": "pt2_double_scaled_weight_derivatives(1.0, 0.5, 4.0)",
            "gold_call": "_oracle_pt2_double_scaled_weight_derivatives(1.0, 0.5, 4.0)",
        },
    ]
