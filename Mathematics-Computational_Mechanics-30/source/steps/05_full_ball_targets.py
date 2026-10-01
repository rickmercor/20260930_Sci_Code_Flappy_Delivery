"""
Return the target vector the source fits against: the derivative operators of Eq (60) and the energy operator, each applied to the five non-constant scaled monomial test functions at the node, evaluated over the reference neighbourhood the source prescribes for the targets, and concatenated as x-derivative targets, y-derivative targets, x-lifted energy targets, y-lifted energy targets, in that order, giving a length-20 array. Evaluate every entry exactly, with the same prefactor convention as the energy block. Raise ValueError if delta or q is not positive, or if an entry is not finite.

The weights are chosen so that the truncated discrete operators reproduce a reference. What that reference is, and how it is evaluated, is the source's central design decision.

Returns
-------
ndarray of float64 with shape (20,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def full_ball_targets(delta, q):
    """ndarray of float64 with shape (20,)."""
    return np.zeros(20, dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_full_ball_targets(delta, q):
    """Eqs (60)-(63): the operators applied to the tests over the FULL ball, analytically.

    Returns a 1-D array of length 4*n_T = 20: [d_x*, d_y*, e_x*, e_y*] for the five
    non-constant scaled monomial tests x/d, y/d, x^2/d^2, xy/d^2, y^2/d^2. Uses the
    closed-form ball moments
        I_q(r1, r2) = int_{|xi|<=d} xi_x^r1 xi_y^r2 / |xi|^q dA
                    = d^(r1+r2+2-q) / (r1+r2+2-q) * int_0^{2pi} cos^r1 sin^r2 dtheta,
    which is the source's eq (63); it vanishes unless both exponents are even.
    """
    import numpy as np
    from math import gamma
    if delta <= 0.0 or q <= 0.0:
        raise ValueError("delta and q must be positive")
    mono = [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]

    def angular(p, s):
        if p % 2 or s % 2:
            return 0.0
        return 2.0 * gamma((p + 1) / 2.0) * gamma((s + 1) / 2.0) / gamma((p + s) / 2.0 + 1.0)

    def moment(r1, r2):
        e = r1 + r2 + 2 - q
        if e <= 0:
            raise ValueError("non-integrable ball moment")
        return delta ** e / e * angular(r1, r2)

    d = [[], []]
    e = [[], []]
    for a, b in mono:
        sc = delta ** (a + b)
        d[0].append(moment(a + 1, b) / sc)
        d[1].append(moment(a, b + 1) / sc)
        e[0].append(moment(2 * a + 2, 2 * b) / sc ** 2)
        e[1].append(moment(2 * a, 2 * b + 2) / sc ** 2)
    out = np.array(d[0] + d[1] + e[0] + e[1], dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError("targets must be finite")
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "delta, q = 0.25, 3",
            "call": "full_ball_targets(delta, q)",
            "gold_call": "_oracle_full_ball_targets(delta, q)",
        },
        {
            "setup": "delta, q = 1.0, 3",
            "call": "full_ball_targets(delta, q)",
            "gold_call": "_oracle_full_ball_targets(delta, q)",
        },
        {
            "setup": "delta, q = 0.5, 2",
            "call": "full_ball_targets(delta, q)",
            "gold_call": "_oracle_full_ball_targets(delta, q)",
        },
    ]
