"""
Implement compute_trapping_fraction, which evaluates the asymptotic trapping
fraction zeta of a quadrupolar mixed mode, the ratio of the mode inertia in
the gravity-mode cavity to the total mode inertia.

zeta is close to 1 for g-dominated modes and small for p-dominated modes. Use
the standard asymptotic closed form of the mixed-mode description, evaluated at an unperturbed mixed-mode frequency nu of the model
with coupling factor q, large separation delta_nu, acoustic comb anchored at
nu_p and gravity-mode period spacing delta_pi.

Returns
-------
float, the trapping fraction zeta in (0, 1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_trapping_fraction(nu: float, q: float, delta_nu: float, nu_p: float, delta_pi: float) -> float:
    '''Asymptotic trapping fraction zeta of an l = 2 mixed mode.

    Parameters
    ----------
    nu : float
        Frequency of an unperturbed mixed mode of the model, in microhertz,
        strictly positive.
    q : float
        Coupling factor in (0, 1).
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
    zeta : float
        Trapping fraction in the open interval (0, 1).

    Raises
    ------
    ValueError
        If nu, delta_nu or delta_pi is not strictly positive, or if q is not
        in (0, 1).
    '''
    return zeta

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_trapping_fraction(nu: float, q: float, delta_nu: float, nu_p: float, delta_pi: float) -> float:
    """Reference implementation of the asymptotic zeta function."""
    nu, q, delta_nu, nu_p, delta_pi = (float(v) for v in (nu, q, delta_nu, nu_p, delta_pi))
    if nu <= 0.0 or delta_nu <= 0.0 or delta_pi <= 0.0:
        raise ValueError("nu, delta_nu and delta_pi must be strictly positive")
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie in the open interval (0, 1)")
    theta_p = _acoustic_phase(nu, delta_nu, nu_p)
    nu_hz, delta_nu_hz = nu * 1e-6, delta_nu * 1e-6
    ratio = nu_hz ** 2 * delta_pi / delta_nu_hz
    denominator = q * np.cos(theta_p) ** 2 + np.sin(theta_p) ** 2 / q
    return float(1.0 / (1.0 + ratio / denominator))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    task = "q = 0.0360054005\ndelta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850\n"
    invalid_setup = """
def run_model():
    try:
        compute_trapping_fraction(nu, q, delta_nu, nu_p, delta_pi)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_compute_trapping_fraction(nu, q, delta_nu, nu_p, delta_pi)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Normal: the task's lower observed mode (partly p-dominated) ---
        {
            "setup": task + "nu = 366.805\n",
            "call": "compute_trapping_fraction(nu, q, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_compute_trapping_fraction(nu, q, delta_nu, nu_p, delta_pi)",
        },
        # --- Normal: a g-dominated neighbour of the pair ---
        {
            "setup": task + "nu = 359.3149105125\n",
            "call": "compute_trapping_fraction(nu, q, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_compute_trapping_fraction(nu, q, delta_nu, nu_p, delta_pi)",
        },
        # --- Boundary: the most p-dominated mode of the task's window (closest to the comb) ---
        {
            "setup": task + "nu = 341.4943423495\n",
            "call": "compute_trapping_fraction(nu, q, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_compute_trapping_fraction(nu, q, delta_nu, nu_p, delta_pi)",
        },
        # --- Edge: a strongly coupled star (q = 0.35), a low-frequency mixed mode ---
        {
            "setup": "q = 0.35\ndelta_nu, nu_p, delta_pi = 29.00, 381.101, 64.422\nnu = 151.42329751\n",
            "call": "compute_trapping_fraction(nu, q, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_compute_trapping_fraction(nu, q, delta_nu, nu_p, delta_pi)",
        },
        # --- Edge: the same star, a mode near the top of its spectrum ---
        {
            "setup": "q = 0.35\ndelta_nu, nu_p, delta_pi = 29.00, 381.101, 64.422\nnu = 258.95493277\n",
            "call": "compute_trapping_fraction(nu, q, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_compute_trapping_fraction(nu, q, delta_nu, nu_p, delta_pi)",
        },
        # --- Invalid: non-positive frequency ---
        {
            "setup": task + "nu = 0.0\n" + invalid_setup,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
