"""
Read both messages of a connected pair off the stationary distribution of its chain, one value per susceptible stage in each direction.

Both directions are read from the same pair distribution. With the pair ordered as (first, second), row 0 of the returned array holds beta P(I_second | S(x)_first), the message received by the first node, and row 1 holds beta P(I_first | S(x)_second), the message received by the second node.

The pair closure is made self-consistent by reading the messages back off the stationary distribution of each pair. The message from j to i in stage x is the rate at which node i, given that it is in susceptible stage S(x), is infected by j.

In the pair chain j infects a susceptible i at rate beta whenever j is infectious, so the message is beta times the conditional probability that j is infectious given that i is in stage S(x):

phi[i, j, x] = beta P(I_j | S(x)_i) = beta P(S(x)_i, I_j) / (P(S(x)_i, I_j) + sum over y of P(S(x)_i, S(y)_j)).

The denominator is the marginal probability that i is in stage S(x), summed over every state of j. The conditioning is stage by stage: a node that recovered recently, and so sits in a young stage, is more likely to have an infectious partner than one that has been susceptible for a long time, and it is this dependence that a single memoryless susceptible class cannot express. With K = 1 the formula reduces to the message of the ordinary pair approximation.

Returns
-------
np.ndarray of shape (2, K): row 0 the stage-resolved messages received by the first node, row 1 those received by the second node, each in units of rate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stage_conditional_infection_rates(
    external_first: np.ndarray,
    external_second: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Read the stage-conditional infection messages of both ends off the stationary pair distribution.

    Parameters
    ----------
    external_first : np.ndarray
        External infection rates a[x] of the first node in stage S(x), length K.
    external_second : np.ndarray
        External infection rates b[y] of the second node in stage S(y), length K.
    beta : float
        Per-contact transmission rate, above zero.
    gamma : float
        Ageing rate between successive susceptible stages, non-negative.

    Returns
    -------
    np.ndarray
        Array of shape (2, K): row 0 is beta P(I_second | S(x)_first), row 1 is beta P(I_first | S(x)_second).

    Raises
    ------
    ValueError
        When the inputs are invalid as for the pair generator, or when the pair chain has no unique stationary distribution.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_stage_conditional_infection_rates(
    external_first: np.ndarray,
    external_second: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Reference implementation."""
    k = np.asarray(external_first).size
    if np.asarray(external_first).ndim != 1 or np.asarray(external_second).shape != (k,) or k < 1:
        raise ValueError("external rates must be one-dimensional arrays of equal length at least 1")
    p = _oracle_pair_stationary_distribution(external_first, external_second, beta, gamma)  # noqa: F821
    joint = p.reshape(k + 1, k + 1)
    beta = float(beta)
    # joint[u, v]: u is the state of the first node, v of the second; column 0 and row 0 are I
    to_first = beta * joint[1:, 0] / joint[1:, :].sum(axis=1)
    to_second = beta * joint[0, 1:] / joint[:, 1:].sum(axis=0)
    return np.vstack([to_first, to_second])

# =============================================================================
# TEST CASES
# =============================================================================

SETUP = """
import numpy as np

def flat(x):
    return tuple(round(float(v), 12) + 0.0 for v in np.asarray(x, dtype=float).ravel())

def verdict(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
"""


def test_cases():
    return [
        {
            # eight stages with a gradient of external pressure, the size used by the graded task
            "setup": SETUP + """
a = np.linspace(0.6, 1.3, 8)
b = np.linspace(1.1, 0.4, 8)
""",
            "call": "flat(stage_conditional_infection_rates(a, b, 0.3, 2.85))",
            "gold_call": "flat(_oracle_stage_conditional_infection_rates(a, b, 0.3, 2.85))",
        },
        {
            # symmetry and the regular-graph closed form: on a (q + 1)-regular graph with q = 2 the
            # one-stage fixed point is phi = beta - 1 / q, so feeding x = q phi to both ends must
            # return phi in both rows; exchanging the ends exchanges the rows
            "setup": SETUP + """
def regular(fn, beta):
    phi = beta - 0.5
    r = fn(np.array([2.0 * phi]), np.array([2.0 * phi]), beta, 0.0)
    return (round(float(r[0, 0] - phi), 12) + 0.0, round(float(r[1, 0] - phi), 12) + 0.0)
def exchange(fn):
    a = np.array([0.2, 0.9, 0.4]); b = np.array([0.7, 0.1, 0.3])
    r = fn(a, b, 0.45, 1.2); s = fn(b, a, 0.45, 1.2)
    return (round(float(np.abs(r - s[::-1]).max()), 12),)
""",
            "call": "regular(stage_conditional_infection_rates, 0.8) + regular(stage_conditional_infection_rates, 0.65) + exchange(stage_conditional_infection_rates)",
            "gold_call": "regular(_oracle_stage_conditional_infection_rates, 0.8) + regular(_oracle_stage_conditional_infection_rates, 0.65) + exchange(_oracle_stage_conditional_infection_rates)",
        },
        {
            # boundary: messages lie between zero and beta, the youngest stage of each node receives
            # the largest message when external pressure is uniform across stages, and very fast
            # ageing collapses the stage dependence onto the one-stage message; the rates themselves
            # are returned as well, so that a wrong implementation cannot pass on the indicators alone
            "setup": SETUP + """
def bounds(fn):
    r = fn(np.full(5, 0.5), np.full(5, 0.5), 0.7, 0.9)
    return (int(np.all(r > 0.0)), int(np.all(r < 0.7)), int(np.argmax(r[0])), int(np.argmax(r[1]))) + flat(r)
def fast_ageing(fn):
    r = fn(np.full(4, 0.3), np.full(4, 0.6), 0.5, 1.0e5)
    one = fn(np.array([0.3]), np.array([0.6]), 0.5, 0.0)
    return (round(float(r[0, 0] - one[0, 0]), 4) + 0.0, round(float(r[1, 0] - one[1, 0]), 4) + 0.0) + flat(one)
""",
            "call": "bounds(stage_conditional_infection_rates) + fast_ageing(stage_conditional_infection_rates)",
            "gold_call": "bounds(_oracle_stage_conditional_infection_rates) + fast_ageing(_oracle_stage_conditional_infection_rates)",
        },
        {
            # malformed inputs and one valid call
            "setup": SETUP + """
a = np.array([0.3, 0.4])
""",
            "call": "(verdict(stage_conditional_infection_rates, a, np.array([0.3]), 0.5, 1.0), verdict(stage_conditional_infection_rates, a, a, float('inf'), 1.0), verdict(stage_conditional_infection_rates, a, -a, 0.5, 1.0), verdict(stage_conditional_infection_rates, a, a, 0.5, float('nan')), verdict(stage_conditional_infection_rates, a, a, 0.5, 1.0))",
            "gold_call": "(verdict(_oracle_stage_conditional_infection_rates, a, np.array([0.3]), 0.5, 1.0), verdict(_oracle_stage_conditional_infection_rates, a, a, float('inf'), 1.0), verdict(_oracle_stage_conditional_infection_rates, a, -a, 0.5, 1.0), verdict(_oracle_stage_conditional_infection_rates, a, a, 0.5, float('nan')), verdict(_oracle_stage_conditional_infection_rates, a, a, 0.5, 1.0))",
        },
    ]
