"""
Implement infer_gravity_phase, which recovers the gravity-mode phase offset
eps_g of the quadrupolar mixed modes from one observed m = 0 mixed-mode
frequency once the coupling factor q is known.

The asymptotic mixed-mode description is the one of infer_coupling_factor
(uniform acoustic comb of spacing delta_nu anchored at nu_p, pure gravity
modes of periods (n + 1/2 + eps_g) * delta_pi, coupling factor q). The
coupling relation determines eps_g only modulo 1; the representative in
[0, 1) is the conventional phase of the pure gravity modes.

Returns
-------
float, the gravity phase eps_g in [0, 1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def infer_gravity_phase(nu_a: float, q: float, delta_nu: float, nu_p: float, delta_pi: float) -> float:
    '''Gravity phase eps_g in [0, 1) from one m = 0 mixed-mode frequency.

    Parameters
    ----------
    nu_a : float
        Frequency of an observed m = 0 mixed mode, in microhertz, strictly positive.
    q : float
        Coupling factor of the l = 2 mixed modes, in the open interval (0, 1).
    delta_nu : float
        Large frequency separation of the acoustic comb, in microhertz,
        strictly positive.
    nu_p : float
        Frequency of one l = 2 pure acoustic mode, in microhertz.
    delta_pi : float
        Asymptotic period spacing of the l = 2 gravity modes, in seconds,
        strictly positive.

    Returns
    -------
    eps_g : float
        The gravity phase in [0, 1) for which nu_a satisfies the coupling
        relation with the supplied q.

    Raises
    ------
    ValueError
        If q is not in (0, 1), or if nu_a, delta_nu or delta_pi is not
        strictly positive.
    '''
    return eps_g

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_infer_gravity_phase(nu_a: float, q: float, delta_nu: float, nu_p: float, delta_pi: float) -> float:
    """Reference implementation: invert the coupling relation for the gravity phase."""
    nu_a, q, delta_nu, nu_p, delta_pi = (float(v) for v in (nu_a, q, delta_nu, nu_p, delta_pi))
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie in the open interval (0, 1)")
    if nu_a <= 0.0 or delta_nu <= 0.0 or delta_pi <= 0.0:
        raise ValueError("nu_a, delta_nu and delta_pi must be strictly positive")
    theta_p = _acoustic_phase(nu_a, delta_nu, nu_p)
    # theta_g modulo pi from tan(theta_g) = tan(theta_p) / q.
    theta_g = np.arctan(np.tan(theta_p) / q)
    eps_g = 1.0 / (delta_pi * nu_a * 1e-6) - theta_g / np.pi
    return float(eps_g % 1.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    invalid_setup = """
def run_model():
    try:
        infer_gravity_phase(nu_a, q, delta_nu, nu_p, delta_pi)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_infer_gravity_phase(nu_a, q, delta_nu, nu_p, delta_pi)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Normal: the task's lower mode with its inferred coupling factor ---
        {
            "setup": "nu_a, q = 366.805, 0.0360054005\ndelta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850\n",
            "call": "infer_gravity_phase(nu_a, q, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_infer_gravity_phase(nu_a, q, delta_nu, nu_p, delta_pi)",
        },
        # --- Normal: the task's upper mode must give the same phase ---
        {
            "setup": "nu_a, q = 368.675, 0.0360054005\ndelta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850\n",
            "call": "infer_gravity_phase(nu_a, q, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_infer_gravity_phase(nu_a, q, delta_nu, nu_p, delta_pi)",
        },
        # --- Boundary: mode exactly on the acoustic comb (theta_p = 0, so the phase is the pure g-mode value) ---
        {
            "setup": "nu_a, q = 368.100, 0.05\ndelta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850\n",
            "call": "infer_gravity_phase(nu_a, q, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_infer_gravity_phase(nu_a, q, delta_nu, nu_p, delta_pi)",
        },
        # --- Edge: strong coupling, mode far from the comb ---
        {
            "setup": "nu_a, q = 393.50, 0.85\ndelta_nu, nu_p, delta_pi = 29.00, 381.101, 64.422\n",
            "call": "infer_gravity_phase(nu_a, q, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_infer_gravity_phase(nu_a, q, delta_nu, nu_p, delta_pi)",
        },
        # --- Invalid: coupling factor outside (0, 1) ---
        {
            "setup": "nu_a, q = 366.805, 1.0\ndelta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850\n" + invalid_setup,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
