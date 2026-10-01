"""
Find the inverse temperature at which the modulus of the global twist expectation value falls to a prescribed small value, and confirm the topological sector locally.

This is the end-to-end pipeline step. The ensemble geometric phase is well defined at every temperature, but its practical use rests on the modulus of the twist expectation value, which collapses exponentially with system size once the temperature is nonzero. The inverse temperature at which the modulus reaches a small threshold marks where the global diagnostic stops being measurable for a chain of given length, and at that temperature the local twist indicators must still identify the same sector as the geometric phase.

Returns
-------
beta_star : float -- Inverse temperature at which |<T>| = target.
"""

import numpy as np
import math

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def twist_crossing_beta(n_cells: int, t1: float, t2: float, t3: float, target: float,
                        beta_bracket: "np.ndarray", delta_x: float, n_local: int) -> float:
    '''Return beta* in the bracket with |<T>| = target, after a local consistency check.

    With the step-05 value, the bracket (b_lo, b_hi) satisfies ln|<T>| < ln(target) at b_lo
    and > ln(target) at b_hi, and for every input used with this function ln|<T>| crosses
    ln(target) exactly once inside the bracket. All global-twist evaluations must lie
    in step 05's supported overlap/cancellation domain. Reduce the bisection bracket width
    to 1e-12, or stop when its endpoints
    are adjacent representable floating-point values if that width is unattainable.
    Return the midpoint of the final bracket.
    Then, with the step-05 phase phi at beta* and the step-09 indicators at beta* for a
    chain of n_local cells with the same hoppings and the given delta_x, verify consistency.
    For n_local <= 151, also require agreement of steps 08 and 09 to absolute 1e-10
    and use the step-08 indicators in the consistency check.
    Let p = 0 for odd N and p = pi for even N (the parity factor (-1)^(N-1)). If phi is within
    1e-6 of p (modulo 2 pi) the chain must be trivial, Delta_T3 > 0; if phi is within 1e-6
    of p + pi (modulo 2 pi) it must be topological, Delta_T3 < 0; any other phase, or a
    failed check, raises ValueError.

    Parameters
    ----------
    n_cells : int
        Number of unit cells N >= 2 for the global twist.
    t1, t2, t3 : float
        Hopping amplitudes as in step 01.
    target : float
        Threshold modulus, 0 < target < 1.
    beta_bracket : np.ndarray
        Inverse temperatures (b_lo, b_hi) with 0 <= b_lo < b_hi < inf.
    delta_x : float
        Intracell half-separation for the local indicators, as in step 07.
    n_local : int
        Odd number of unit cells >= 3 of the chain used for the local indicators (step 09).

    Returns
    -------
    beta_star : float
        Inverse temperature at which |<T>| = target.

    Raises
    ------
    ValueError
        If target is not in (0, 1), the bracket is not increasing or does not enclose the
        crossing, either consistency check fails, or the inputs are invalid as in steps 05
        and 09 (with n_local in place of n_cells for step 09).
    '''
    return beta_star

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_twist_crossing_beta(n_cells: int, t1: float, t2: float, t3: float, target: float,
                                beta_bracket: "np.ndarray", delta_x: float, n_local: int) -> float:
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie in (0, 1)")
    lo, hi = (float(b) for b in beta_bracket)
    if not 0.0 <= lo < hi < np.inf:
        raise ValueError("beta_bracket must be increasing and finite")
    goal = np.log(target)

    def _excess(beta):
        return _oracle_global_twist_expectation(n_cells, t1, t2, t3, beta)[0] - goal

    if not (_excess(lo) < 0.0 < _excess(hi)):
        raise ValueError("bracket does not enclose the crossing")
    while hi - lo > 1e-12:
        mid = 0.5 * lo + 0.5 * hi
        if mid == lo or mid == hi:
            break
        if _excess(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    beta_star = 0.5 * lo + 0.5 * hi
    phase = _oracle_global_twist_expectation(n_cells, t1, t2, t3, beta_star)[1]
    local = _oracle_bloch_local_indicators(n_local, t1, t2, t3, beta_star, delta_x)
    if n_local <= 151:
        dense_local = _oracle_local_indicators(n_local, t1, t2, t3, beta_star, delta_x)
        if np.max(np.abs(local - dense_local)) > 1e-10:
            raise ValueError("local representations disagree")
        local = dense_local
    delta_t3 = local[0]
    parity = 0.0 if int(n_cells) % 2 == 1 else np.pi

    def _near(value, reference):
        return abs(np.angle(np.exp(1j * (value - reference)))) < 1e-6

    if _near(phase, parity):
        consistent = delta_t3 > 0.0
    elif _near(phase, parity + np.pi):
        consistent = delta_t3 < 0.0
    else:
        consistent = False
    if not consistent:
        raise ValueError("local indicators disagree with the geometric phase")
    return float(beta_star)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
"""
    guard = setup + """
def raises_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    return 0.0
"""
    return [
        {"setup": setup,
         "call": "twist_crossing_beta(31, 1e-7, 0.0, 0.0, 0.01, np.array([1e6, 5e8]), 0.25, 31)",
         "gold_call": "_oracle_twist_crossing_beta(31, 1e-7, 0.0, 0.0, 0.01, np.array([1e6, 5e8]), 0.25, 31)"},
        {"setup": setup,
         "call": "twist_crossing_beta(31, 1.0, 0.6, 1.5, 0.01, np.array([0.2, 50.0]), 0.25, 31)",
         "gold_call": "_oracle_twist_crossing_beta(31, 1.0, 0.6, 1.5, 0.01, np.array([0.2, 50.0]), 0.25, 31)"},
        {"setup": setup,
         "call": "twist_crossing_beta(31, 3.0, 2.0, 0.0, 0.01, np.array([0.2, 50.0]), 0.25, 31)",
         "gold_call": "_oracle_twist_crossing_beta(31, 3.0, 2.0, 0.0, 0.01, np.array([0.2, 50.0]), 0.25, 31)"},
        {"setup": setup,
         "call": "twist_crossing_beta(21, 1.0, 1.5, 0.6, 0.2, np.array([0.2, 40.0]), 0.25, 21)",
         "gold_call": "_oracle_twist_crossing_beta(21, 1.0, 1.5, 0.6, 0.2, np.array([0.2, 40.0]), 0.25, 21)"},
        {"setup": setup,
         "call": "twist_crossing_beta(400, 1.0, 0.6, 1.5, 0.01, np.array([2.0, 30.0]), 0.25, 20001)",
         "gold_call": "_oracle_twist_crossing_beta(400, 1.0, 0.6, 1.5, 0.01, np.array([2.0, 30.0]), 0.25, 20001)"},
        {"setup": setup,
         "call": "twist_crossing_beta(2001, 3.0, 2.0, 0.0, 0.05, np.array([1.0, 30.0]), 0.1, 19999)",
         "gold_call": "_oracle_twist_crossing_beta(2001, 3.0, 2.0, 0.0, 0.05, np.array([1.0, 30.0]), 0.1, 19999)"},
        {"setup": guard,
         "call": "raises_value_error(twist_crossing_beta, 31, 1.0, 0.6, 1.5, 1.5, np.array([0.1, 50.0]), 0.25, 31)",
         "gold_call": "raises_value_error(_oracle_twist_crossing_beta, 31, 1.0, 0.6, 1.5, 1.5, np.array([0.1, 50.0]), 0.25, 31)"},
        {"setup": guard,
         "call": "raises_value_error(twist_crossing_beta, 31, 1.0, 0.6, 1.5, 0.01, np.array([40.0, 50.0]), 0.25, 31)",
         "gold_call": "raises_value_error(_oracle_twist_crossing_beta, 31, 1.0, 0.6, 1.5, 0.01, np.array([40.0, 50.0]), 0.25, 31)"},
        {"setup": guard,
         "call": "raises_value_error(twist_crossing_beta, 31, 1.0, 0.6, 1.5, 0.01, np.array([0.2, 50.0]), 0.25, 20000)",
         "gold_call": "raises_value_error(_oracle_twist_crossing_beta, 31, 1.0, 0.6, 1.5, 0.01, np.array([0.2, 50.0]), 0.25, 20000)"},
    ]
