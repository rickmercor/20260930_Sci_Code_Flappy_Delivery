"""
Implement compute_multiplet_asymmetry, the end-to-end pipeline that predicts
the near-degeneracy asymmetry of one rotational multiplet of quadrupolar
mixed modes from two observed m = 0 frequencies, the global seismic
parameters and the two-zone rotation rates.

The pipeline combines the earlier functions of this task: the coupling
factor and gravity phase inferred from the observed pair, the set of
unperturbed mixed modes in the requested window, the rotational coupling
matrices of that set for the azimuthal orders +m and -m and the exact
solution of the rotating eigenvalue problem for both. It returns, for the
multiplet whose m = 0 component is closest to nu_target, the asymmetry
nu(+m) + nu(-m) - 2 nu(0), where nu(0) is the m = 0 component (unshifted,
because the coupling matrix vanishes for m = 0).

Returns
-------
float, the asymmetry nu(+m) + nu(-m) - 2 nu(0) in nanohertz
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_multiplet_asymmetry(nu_a: float, nu_b: float, delta_nu: float, nu_p: float, delta_pi: float, nu_core: float, nu_env: float, nu_lo: float, nu_hi: float, nu_target: float, m: int = 2) -> float:
    '''Near-degeneracy asymmetry nu(+m) + nu(-m) - 2 nu(0) of one l = 2 multiplet, in nanohertz.

    Parameters
    ----------
    nu_a : float
        Frequency of the first observed m = 0 mixed mode, in microhertz.
    nu_b : float
        Frequency of the second observed m = 0 mixed mode, in microhertz,
        different from nu_a.
    delta_nu : float
        Large frequency separation of the acoustic comb, in microhertz,
        strictly positive.
    nu_p : float
        Frequency of one l = 2 pure acoustic mode, in microhertz.
    delta_pi : float
        Asymptotic period spacing of the l = 2 gravity modes, in seconds,
        strictly positive.
    nu_core : float
        Core rotation rate Omega_core / (2 pi), in nanohertz.
    nu_env : float
        Envelope rotation rate Omega_env / (2 pi), in nanohertz.
    nu_lo : float
        Lower edge of the window of coupled modes, in microhertz, strictly positive.
    nu_hi : float
        Upper edge of the window, in microhertz, greater than nu_lo.
    nu_target : float
        Frequency, in microhertz, identifying the multiplet: the unperturbed
        mode closest to it is used.
    m : int, optional
        Positive azimuthal order of the pair of components, 1 or 2 (default 2).

    Returns
    -------
    asymmetry : float
        nu(+m) + nu(-m) - 2 nu(0) for the selected multiplet, in nanohertz;
        it has the same value for m and -m.

    Raises
    ------
    ValueError
        If m is not 1 or 2, if the window contains no mixed mode, or if any
        upstream step rejects its input (see the earlier functions).
    '''
    return asymmetry

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_multiplet_asymmetry(nu_a: float, nu_b: float, delta_nu: float, nu_p: float, delta_pi: float, nu_core: float, nu_env: float, nu_lo: float, nu_hi: float, nu_target: float, m: int = 2) -> float:
    """Reference end-to-end pipeline."""
    if isinstance(m, bool) or int(m) != m or int(m) not in (1, 2):
        raise ValueError("m must be 1 or 2")
    m = int(m)
    q = _oracle_infer_coupling_factor(nu_a, nu_b, delta_nu, nu_p, delta_pi)
    eps_g = _oracle_infer_gravity_phase(nu_a, q, delta_nu, nu_p, delta_pi)
    nu_modes = _oracle_find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)
    if nu_modes.size == 0:
        raise ValueError("the window contains no mixed mode")
    index = int(np.argmin(np.abs(nu_modes - float(nu_target))))
    nu_plus = _oracle_solve_rotational_multiplets(
        nu_modes, _oracle_assemble_rotation_matrix(nu_modes, q, delta_nu, nu_p, delta_pi, nu_core, nu_env, m))
    nu_minus = _oracle_solve_rotational_multiplets(
        nu_modes, _oracle_assemble_rotation_matrix(nu_modes, q, delta_nu, nu_p, delta_pi, nu_core, nu_env, -m))
    return float((nu_plus[index] + nu_minus[index] - 2.0 * nu_modes[index]) * 1e3)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    task = """nu_a, nu_b = 366.805, 368.675
delta_nu, nu_p, delta_pi = 26.50, 371.204 - 3.104, 60.850
nu_core, nu_env = 745.0, 61.0
nu_lo, nu_hi = 371.204 - 1.5 * 26.50, 371.204 + 1.5 * 26.50
"""
    invalid_setup = """
def run_model():
    try:
        compute_multiplet_asymmetry(nu_a, nu_b, delta_nu, nu_p, delta_pi, nu_core, nu_env, nu_lo, nu_hi, nu_target, m)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_compute_multiplet_asymmetry(nu_a, nu_b, delta_nu, nu_p, delta_pi, nu_core, nu_env, nu_lo, nu_hi, nu_target, m)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Normal: the task's multiplet a (lower observed mode), m = 2 ---
        {
            "setup": task + "nu_target, m = 366.805, 2\n",
            "call": "compute_multiplet_asymmetry(nu_a, nu_b, delta_nu, nu_p, delta_pi, nu_core, nu_env, nu_lo, nu_hi, nu_target, m)",
            "gold_call": "_oracle_compute_multiplet_asymmetry(nu_a, nu_b, delta_nu, nu_p, delta_pi, nu_core, nu_env, nu_lo, nu_hi, nu_target, m)",
        },
        # --- Normal: the partner multiplet b with the m = 1 components ---
        {
            "setup": task + "nu_target, m = 368.675, 1\n",
            "call": "compute_multiplet_asymmetry(nu_a, nu_b, delta_nu, nu_p, delta_pi, nu_core, nu_env, nu_lo, nu_hi, nu_target, m)",
            "gold_call": "_oracle_compute_multiplet_asymmetry(nu_a, nu_b, delta_nu, nu_p, delta_pi, nu_core, nu_env, nu_lo, nu_hi, nu_target, m)",
        },
        # --- Boundary: a far, g-dominated multiplet whose asymmetry is small ---
        {
            "setup": task + "nu_target, m = 344.5364, 2\n",
            "call": "compute_multiplet_asymmetry(nu_a, nu_b, delta_nu, nu_p, delta_pi, nu_core, nu_env, nu_lo, nu_hi, nu_target, m)",
            "gold_call": "_oracle_compute_multiplet_asymmetry(nu_a, nu_b, delta_nu, nu_p, delta_pi, nu_core, nu_env, nu_lo, nu_hi, nu_target, m)",
        },
        # --- Edge: window reduced to the observed pair only (two-mode coupling), slow rotation ---
        {
            "setup": """nu_a, nu_b = 366.805, 368.675
delta_nu, nu_p, delta_pi = 26.50, 371.204 - 3.104, 60.850
nu_core, nu_env = 200.0, 20.0
nu_lo, nu_hi = 366.0, 369.0
nu_target, m = 366.805, 2
""",
            "call": "compute_multiplet_asymmetry(nu_a, nu_b, delta_nu, nu_p, delta_pi, nu_core, nu_env, nu_lo, nu_hi, nu_target, m)",
            "gold_call": "_oracle_compute_multiplet_asymmetry(nu_a, nu_b, delta_nu, nu_p, delta_pi, nu_core, nu_env, nu_lo, nu_hi, nu_target, m)",
        },
        # --- Normal: a luminous giant (Delta nu = 4 microhertz, dense gravity modes 0.02-0.1
        #     microhertz apart, 152 modes in the window) rotating slowly ---
        {
            "setup": """nu_a, nu_b = 29.979546, 30.013518
delta_nu, nu_p, delta_pi = 4.0, 30.0, 60.0
nu_core, nu_env = 20.0, 2.0
nu_lo, nu_hi = 26.0, 34.0
nu_target, m = 29.979546, 2
""",
            "call": "compute_multiplet_asymmetry(nu_a, nu_b, delta_nu, nu_p, delta_pi, nu_core, nu_env, nu_lo, nu_hi, nu_target, m)",
            "gold_call": "_oracle_compute_multiplet_asymmetry(nu_a, nu_b, delta_nu, nu_p, delta_pi, nu_core, nu_env, nu_lo, nu_hi, nu_target, m)",
        },
        # --- Edge: fast rotation (core 4000 nHz, envelope 333.3 nHz) on the task's star, where the
        #     perturbed components of neighbouring multiplets cross in frequency ---
        {
            "setup": task + "nu_core, nu_env = 4000.0, 333.3\nnu_target, m = 366.805, 2\n",
            "call": "compute_multiplet_asymmetry(nu_a, nu_b, delta_nu, nu_p, delta_pi, nu_core, nu_env, nu_lo, nu_hi, nu_target, m)",
            "gold_call": "_oracle_compute_multiplet_asymmetry(nu_a, nu_b, delta_nu, nu_p, delta_pi, nu_core, nu_env, nu_lo, nu_hi, nu_target, m)",
        },
        # --- Invalid: azimuthal order 3 ---
        {
            "setup": task + "nu_target, m = 366.805, 3\n" + invalid_setup,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
