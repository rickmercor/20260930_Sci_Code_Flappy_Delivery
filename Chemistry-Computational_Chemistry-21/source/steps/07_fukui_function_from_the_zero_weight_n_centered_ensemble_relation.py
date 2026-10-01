"""
Given the zero-weight derivative of the ensemble Hxc

potential for one branch, return the corresponding site-0 Fukui function of

the two-electron dimer. Use the exact relation of the published N-centered

ensemble formulation in its zero-weight limit, with every other ingredient

exact.

Fukui functions measure how the density responds to

adding or removing an electron. In fractional-electron-number DFT, evaluating

them from the Kohn-Sham system requires a derivative discontinuity of the Hxc

kernel, which regular approximations lack. Ensemble formulations that keep a

fixed central electron number express the same physics through derivatives of

the ensemble Hxc potential with respect to the ensemble weights. In the

zero-weight limit, every other ingredient reduces to a regular ground-state

quantity.

Returns
-------
a single float, the Fukui function for the requested     branch. Raise ``ValueError`` if ``branch`` is neither +1 nor -1, or for     invalid dimer parameters.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nc_zero_weight_fukui(t: float, U: float, dv: float, w: float, branch: int) -> float:
    """Return the site-0 Fukui function from the zero-weight N-centered ensemble relation.

    Expected return: a single float, the Fukui function for the requested
    branch. Raise ``ValueError`` if ``branch`` is neither +1 nor -1, or for
    invalid dimer parameters.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_nc_zero_weight_fukui(t: float, U: float, dv: float, w: float, branch: int) -> float:
    if branch != 1 and branch != -1:
        raise ValueError("branch must be +1 (affinity) or -1 (ionization).")
    n0, chi, f_hxc, v_hx, v_c = _oracle_exact_ground_state_hxc(t, U, dv)
    dv_s, chi_s, f_s_plus, f_s_minus = _oracle_nc_ks_inversion(t, n0, 0.0, 0.0)
    if branch == 1:
        f_s = f_s_plus
    else:
        f_s = f_s_minus
    return (1.0 + chi * f_hxc) * f_s - 0.5 * chi * f_hxc * n0 + branch * chi * w

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return one exact-weight-derivative case and four cases with other weight derivatives."""
    return [
        # Normal: ionization branch with the exact weight derivative of this dimer.
        {
            "setup": "",
            "call": "nc_zero_weight_fukui(1.0, 2.0, 2.4, 0.672463035844448, -1)",
            "gold_call": "_oracle_nc_zero_weight_fukui(1.0, 2.0, 2.4, 0.672463035844448, -1)",
        },
        # Normal: same dimer, a different weight derivative.
        {
            "setup": "",
            "call": "nc_zero_weight_fukui(1.0, 2.0, 2.4, 0.8, -1)",
            "gold_call": "_oracle_nc_zero_weight_fukui(1.0, 2.0, 2.4, 0.8, -1)",
        },
        # Normal: affinity branch, negative weight derivative.
        {
            "setup": "",
            "call": "nc_zero_weight_fukui(1.0, 1.5, 3.0, -0.2, 1)",
            "gold_call": "_oracle_nc_zero_weight_fukui(1.0, 1.5, 3.0, -0.2, 1)",
        },
        # Edge: negative dv, hopping different from 1, ionization branch.
        {
            "setup": "",
            "call": "nc_zero_weight_fukui(2.0, 3.0, -1.0, 0.5, -1)",
            "gold_call": "_oracle_nc_zero_weight_fukui(2.0, 3.0, -1.0, 0.5, -1)",
        },
        # Boundary: weak interaction, zero weight derivative, affinity branch.
        {
            "setup": "",
            "call": "nc_zero_weight_fukui(1.0, 0.3, 1.0, 0.0, 1)",
            "gold_call": "_oracle_nc_zero_weight_fukui(1.0, 0.3, 1.0, 0.0, 1)",
        },
    ]
