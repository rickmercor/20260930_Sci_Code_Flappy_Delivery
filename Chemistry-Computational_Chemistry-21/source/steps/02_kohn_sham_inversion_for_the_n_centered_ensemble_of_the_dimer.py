"""
For a given ensemble site-0 occupation and set of ensemble

weights, return the Kohn-Sham potential difference that reproduces that

occupation in the N-centered ensemble, together with the Kohn-Sham ensemble

response and the Kohn-Sham Fukui functions.

Ensemble DFT maps an ensemble of interacting states

onto an ensemble of non-interacting (Kohn-Sham) states with the same weights

and the same ensemble density. For Fukui functions of the two-electron dimer,

the ensemble combines the central two-electron ground state with the three-

and one-electron ground states. The member weights follow the published

N-centered formulation. The Kohn-Sham Fukui functions are the site-0 weights

of the frontier orbitals: the lowest unoccupied orbital for electron addition

and the highest occupied orbital for electron removal.

Returns
-------
a tuple ``(dv_s, chi_s, f_s_plus, f_s_minus)`` of four     floats. Raise ``ValueError`` if ``t <= 0``, if the weights are not an     admissible N-centered weight set, or if ``n`` cannot be reproduced by the     Kohn-Sham ensemble with these weights.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nc_ks_inversion(t: float, n: float, xi_plus: float, xi_minus: float) -> tuple[float, float, float, float]:
    """Return the N-centered Kohn-Sham potential, response and Fukui functions.

    Expected return: a tuple ``(dv_s, chi_s, f_s_plus, f_s_minus)`` of four
    floats. Raise ``ValueError`` if ``t <= 0``, if the weights are not an
    admissible N-centered weight set, or if ``n`` cannot be reproduced by the
    Kohn-Sham ensemble with these weights.
    """
    return (0.0, 0.0, 0.0, 0.0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_nc_ks_inversion(t: float, n: float, xi_plus: float, xi_minus: float) -> tuple[float, float, float, float]:
    if t <= 0.0:
        raise ValueError("The hopping t must be positive.")
    if xi_plus < 0.0 or xi_minus < 0.0 or 3.0 * xi_plus + xi_minus > 2.0:
        raise ValueError("The weights are not an admissible N-centered weight set.")
    offset = n - 1.0
    scale = 1.0 - xi_plus
    if abs(offset) >= scale:
        raise ValueError("The occupation is not non-interacting ensemble representable.")

    dv_s = 2.0 * t * offset / (scale * scale - offset * offset) ** 0.5
    root = (t * t + 0.25 * dv_s * dv_s) ** 0.5
    chi_s = scale * t * t / (2.0 * root**3)
    f_s_plus = 0.5 - dv_s / (4.0 * root)
    f_s_minus = 0.5 + dv_s / (4.0 * root)
    return (dv_s, chi_s, f_s_plus, f_s_minus)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return zero-weight, symmetric and nonzero-weight inversions."""
    return [
        # Normal: zero weights, occupation above 1.
        {
            "setup": "",
            "call": "nc_ks_inversion(1.0, 1.25, 0.0, 0.0)",
            "gold_call": "_oracle_nc_ks_inversion(1.0, 1.25, 0.0, 0.0)",
        },
        # Boundary: symmetric occupation with a nonzero removal weight.
        {
            "setup": "",
            "call": "nc_ks_inversion(1.0, 1.0, 0.0, 0.3)",
            "gold_call": "_oracle_nc_ks_inversion(1.0, 1.0, 0.0, 0.3)",
        },
        # Normal: both weights nonzero.
        {
            "setup": "",
            "call": "nc_ks_inversion(1.0, 1.3, 0.2, 0.1)",
            "gold_call": "_oracle_nc_ks_inversion(1.0, 1.3, 0.2, 0.1)",
        },
        # Normal: occupation below 1, hopping different from 1.
        {
            "setup": "",
            "call": "nc_ks_inversion(2.0, 0.6, 0.1, 0.5)",
            "gold_call": "_oracle_nc_ks_inversion(2.0, 0.6, 0.1, 0.5)",
        },
        # Edge: occupation close to the representability limit.
        {
            "setup": "",
            "call": "nc_ks_inversion(1.0, 1.5, 0.4, 0.2)",
            "gold_call": "_oracle_nc_ks_inversion(1.0, 1.5, 0.4, 0.2)",
        },
    ]
