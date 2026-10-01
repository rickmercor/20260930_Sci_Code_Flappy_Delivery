"""
Return the exact zero-weight Hxc ingredients of the dimer:

the ground-state site-0 occupation, the interacting response, the Hxc

kernel, and the Hartree-exchange and correlation potentials at that

occupation.

In Kohn-Sham DFT the Hartree-exchange-correlation

(Hxc) potential is the difference between the Kohn-Sham potential and the

true external potential that give the same density. Its density derivative

is the Hxc kernel, which links the interacting and Kohn-Sham responses. For

the two-electron dimer all of these quantities are known exactly. The

Hartree-exchange part is the part first order in the interaction, and the

correlation part is the remainder. Potentials follow the site-occupation

(SOFT) sign convention.

Returns
-------
a tuple ``(n_0, chi, f_Hxc, v_Hx, v_c)`` of five floats.     Raise ``ValueError`` for invalid parameters.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exact_ground_state_hxc(t: float, U: float, dv: float) -> tuple[float, float, float, float, float]:
    """Return the exact ground-state occupation, response, Hxc kernel and Hx and c potentials.

    Expected return: a tuple ``(n_0, chi, f_Hxc, v_Hx, v_c)`` of five floats.
    Raise ``ValueError`` for invalid parameters.
    """
    return (0.0, 0.0, 0.0, 0.0, 0.0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_exact_ground_state_hxc(t: float, U: float, dv: float) -> tuple[float, float, float, float, float]:
    energy, n0, chi = _oracle_two_electron_ground_state(t, U, dv)
    dv_s, chi_s, f_s_plus, f_s_minus = _oracle_nc_ks_inversion(t, n0, 0.0, 0.0)
    f_hxc = 1.0 / chi_s - 1.0 / chi
    v_hx, d_plus, d_minus = _oracle_ensemble_eexx_hx_potential(t, U, n0, 0.0, 0.0)
    v_c = (dv_s - dv) - v_hx
    return (n0, chi, f_hxc, v_hx, v_c)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return canonical, weakly interacting, symmetric and strongly asymmetric cases."""
    return [
        # Normal: canonical fixture.
        {
            "setup": "",
            "call": "exact_ground_state_hxc(1.0, 2.5, 1.5)",
            "gold_call": "_oracle_exact_ground_state_hxc(1.0, 2.5, 1.5)",
        },
        # Boundary: weak interaction.
        {
            "setup": "",
            "call": "exact_ground_state_hxc(1.0, 0.2, 1.0)",
            "gold_call": "_oracle_exact_ground_state_hxc(1.0, 0.2, 1.0)",
        },
        # Normal: moderate correlation, strong asymmetry.
        {
            "setup": "",
            "call": "exact_ground_state_hxc(1.0, 1.5, 3.0)",
            "gold_call": "_oracle_exact_ground_state_hxc(1.0, 1.5, 3.0)",
        },
        # Boundary: symmetric dimer.
        {
            "setup": "",
            "call": "exact_ground_state_hxc(1.0, 1.0, 0.0)",
            "gold_call": "_oracle_exact_ground_state_hxc(1.0, 1.0, 0.0)",
        },
        # Edge: hopping different from 1, strongly correlated.
        {
            "setup": "",
            "call": "exact_ground_state_hxc(2.0, 4.0, 3.0)",
            "gold_call": "_oracle_exact_ground_state_hxc(2.0, 4.0, 3.0)",
        },
    ]
