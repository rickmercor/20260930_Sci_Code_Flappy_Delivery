"""
Implement infer_coupling_factor, which recovers the p-g coupling factor q of
the quadrupolar (l = 2) mixed modes of an evolved solar-like star from the
frequencies of two observed m = 0 mixed modes of the same radial-order region.

The asymptotic description used throughout this task is the standard one
for mixed modes of evolved solar-like stars: the l = 2 acoustic
modes form a uniform comb of spacing delta_nu anchored at nu_p (the
frequency of the closest radial mode minus the small separation delta_nu02),
the pure gravity modes have periods (n + 1/2 + eps_g) * delta_pi, and the
mixed modes are the frequencies at which the acoustic phase and the
gravity-mode phase satisfy the p-g coupling relation with coupling factor
0 < q < 1. Two mixed modes of the same star share q and eps_g.

Returns
-------
float, the l = 2 coupling factor q in (0, 1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def infer_coupling_factor(nu_a: float, nu_b: float, delta_nu: float, nu_p: float, delta_pi: float) -> float:
    '''Coupling factor q from two m = 0 mixed-mode frequencies.

    Parameters
    ----------
    nu_a : float
        Frequency of the first observed m = 0 mixed mode, in microhertz.
    nu_b : float
        Frequency of the second observed m = 0 mixed mode, in microhertz;
        must differ from nu_a.
    delta_nu : float
        Large frequency separation of the acoustic comb, in microhertz,
        strictly positive.
    nu_p : float
        Frequency of one l = 2 pure acoustic mode, in microhertz; the comb is
        nu_p + k * delta_nu for every integer k.
    delta_pi : float
        Asymptotic period spacing of the l = 2 gravity modes, in seconds,
        strictly positive.

    Returns
    -------
    q : float
        The unique coupling factor in (0, 1) for which a single gravity
        phase makes both observed modes exact solutions of the coupling
        relation.

    Raises
    ------
    ValueError
        If delta_nu or delta_pi is not strictly positive, if nu_a equals
        nu_b, or if no such coupling factor, or more than one, exists in
        (0, 1).
    '''
    return q

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _acoustic_phase(nu: float, delta_nu: float, nu_p: float) -> float:
    """theta_p = pi (nu - nu_p) / delta_nu with frequencies in microhertz."""
    return np.pi * (float(nu) - float(nu_p)) / float(delta_nu)


def _oracle_infer_coupling_factor(nu_a: float, nu_b: float, delta_nu: float, nu_p: float, delta_pi: float) -> float:
    """Reference implementation of the phase-free quadratic for q."""
    nu_a, nu_b, delta_nu, nu_p, delta_pi = (float(v) for v in (nu_a, nu_b, delta_nu, nu_p, delta_pi))
    if delta_nu <= 0.0 or delta_pi <= 0.0:
        raise ValueError("delta_nu and delta_pi must be strictly positive")
    if nu_a == nu_b:
        raise ValueError("the two mixed modes must have different frequencies")
    tan_a = np.tan(_acoustic_phase(nu_a, delta_nu, nu_p))
    tan_b = np.tan(_acoustic_phase(nu_b, delta_nu, nu_p))
    # Frequencies in hertz for the gravity-phase difference.
    kappa = np.tan(np.pi / delta_pi * (1.0 / (nu_b * 1e-6) - 1.0 / (nu_a * 1e-6)))
    # Quadratic A u^2 - B u + kappa = 0 in u = 1 / q.
    coef_a = kappa * tan_a * tan_b
    coef_b = tan_b - tan_a
    if coef_a == 0.0:
        # A mode on the comb (tan = 0) or kappa = 0 makes the equation linear in u.
        roots_u = [kappa / coef_b] if coef_b != 0.0 else []
    else:
        disc = coef_b * coef_b - 4.0 * coef_a * kappa
        if disc < 0.0:
            raise ValueError("the two frequencies do not determine a real coupling factor")
        sqrt_disc = np.sqrt(disc)
        roots_u = [(coef_b + sqrt_disc) / (2.0 * coef_a), (coef_b - sqrt_disc) / (2.0 * coef_a)]
    candidates = []
    for root_u in roots_u:
        if np.isfinite(root_u) and root_u != 0.0:
            q_candidate = 1.0 / root_u
            if 0.0 < q_candidate < 1.0:
                candidates.append(float(q_candidate))
    if len(candidates) != 1:
        raise ValueError("exactly one admissible coupling factor in (0, 1) is required")
    return candidates[0]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    invalid_setup = """
def run_model():
    try:
        infer_coupling_factor(nu_a, nu_b, delta_nu, nu_p, delta_pi)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_infer_coupling_factor(nu_a, nu_b, delta_nu, nu_p, delta_pi)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Normal: the task's pair of m = 0 modes straddling the l = 2 acoustic mode ---
        {
            "setup": "nu_a, nu_b = 366.805, 368.675\ndelta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850\n",
            "call": "infer_coupling_factor(nu_a, nu_b, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_infer_coupling_factor(nu_a, nu_b, delta_nu, nu_p, delta_pi)",
        },
        # --- Normal: a more evolved star with a wider pair and a stronger coupling ---
        {
            "setup": "nu_a, nu_b = 380.382, 382.627\ndelta_nu, nu_p, delta_pi = 29.00, 381.101, 64.422\n",
            "call": "infer_coupling_factor(nu_a, nu_b, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_infer_coupling_factor(nu_a, nu_b, delta_nu, nu_p, delta_pi)",
        },
        # --- Boundary: modes given in reversed order (the result must not depend on ordering) ---
        {
            "setup": "nu_a, nu_b = 368.675, 366.805\ndelta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850\n",
            "call": "infer_coupling_factor(nu_a, nu_b, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_infer_coupling_factor(nu_a, nu_b, delta_nu, nu_p, delta_pi)",
        },
        # --- Edge: two g-dominated modes on the same side of the acoustic mode ---
        {
            "setup": "nu_a, nu_b = 351.7109413225, 359.3149105125\ndelta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850\n",
            "call": "infer_coupling_factor(nu_a, nu_b, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_infer_coupling_factor(nu_a, nu_b, delta_nu, nu_p, delta_pi)",
        },
        # --- Edge: one observed mode exactly on the acoustic comb (a star whose gravity phase
        #     puts a mixed mode at nu_p), paired with a g-dominated mode of the same star ---
        {
            "setup": "nu_a, nu_b = 368.100, 380.90494005742016\ndelta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850\n",
            "call": "infer_coupling_factor(nu_a, nu_b, delta_nu, nu_p, delta_pi)",
            "gold_call": "_oracle_infer_coupling_factor(nu_a, nu_b, delta_nu, nu_p, delta_pi)",
        },
        # --- Invalid: a pair that no coupling factor in (0, 1) can reproduce with a common phase ---
        {
            "setup": "nu_a, nu_b = 368.049, 375.878\ndelta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850\n" + invalid_setup,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: two g-dominated modes on the same side of the comb that admit TWO coupling
        #     factors in (0, 1) with different gravity phases (no unique solution) ---
        {
            "setup": "nu_a, nu_b = 311.5117096136005, 315.0370124420079\ndelta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850\n" + invalid_setup,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: identical frequencies ---
        {
            "setup": "nu_a, nu_b = 366.805, 366.805\ndelta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850\n" + invalid_setup,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: non-positive period spacing ---
        {
            "setup": "nu_a, nu_b = 366.805, 368.675\ndelta_nu, nu_p, delta_pi = 26.50, 368.100, 0.0\n" + invalid_setup,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
