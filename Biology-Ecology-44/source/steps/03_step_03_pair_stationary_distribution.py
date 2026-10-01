"""
Solve the balance equations of the pair chain for its stationary distribution, replacing one redundant equation by the normalisation, and return the distribution in the flat state ordering of the previous step.

Solving the linear system in floating point can leave entries of order the machine precision with either sign. The distribution is returned with every entry floored at 1e-16 and then renormalised to unit sum, so that the conditional probabilities formed from it in the next step are always defined. At any endemic operating point the floor lies many orders of magnitude below every entry.

The pair closure assumes that the joint state of each connected pair has relaxed to the stationary distribution of its own Markov chain, given the external infection rates acting on its two ends.

That distribution P is the row vector that satisfies the global balance equations P Q = 0 for the pair generator Q of the previous step, together with the normalisation that its entries sum to one. It is the left null vector of Q.

The balance equations are linearly dependent, because every row of Q sums to zero, so one of them is redundant and is replaced by the normalisation; the resulting square system has a unique solution whenever the chain has a single closed communicating class. When the external rates on the oldest stage both vanish the state (S(1), S(1)) is absorbing and the stationary distribution is concentrated on it, which is the disease-free state of the pair.

Returns
-------
np.ndarray of length (K + 1)^2, the stationary distribution of the pair chain in the flat ordering u * (K + 1) + v, with 0 for I and x for S(x), non-negative and summing to one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pair_stationary_distribution(
    external_first: np.ndarray,
    external_second: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Compute the stationary distribution of the memory-augmented SIS chain of one connected pair.

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
        Stationary distribution of length (K + 1)^2 in the flat ordering u * (K + 1) + v.

    Raises
    ------
    ValueError
        When the inputs are invalid as for the pair generator, or when the chain has no unique stationary distribution.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pair_stationary_distribution(
    external_first: np.ndarray,
    external_second: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Reference implementation."""
    q = _oracle_pair_transition_matrix(external_first, external_second, beta, gamma)  # noqa: F821
    size = q.shape[0]
    system = q.T.copy()
    system[-1, :] = 1.0
    rhs = np.zeros(size)
    rhs[-1] = 1.0
    try:
        p = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the pair chain has no unique stationary distribution") from exc
    if not np.all(np.isfinite(p)) or np.abs(p @ q).max() > 1e-9 * max(1.0, np.abs(q).max()):
        raise ValueError("the pair chain has no unique stationary distribution")
    p = np.maximum(p, 1e-16)
    return p / p.sum()

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
            # four stages with unequal external pressure on the two ends
            "setup": SETUP + """
a = np.array([0.42, 0.55, 0.61, 0.9])
b = np.array([0.18, 0.2, 0.35, 0.47])
""",
            "call": "flat(pair_stationary_distribution(a, b, 0.35, 1.7))",
            "gold_call": "flat(_oracle_pair_stationary_distribution(a, b, 0.35, 1.7))",
        },
        {
            # K = 1 against the ordinary SIS pair: the conditional rate at which the first node is
            # infected by the second, beta P(S, I) / (P(S, I) + P(S, S)), must equal the closed form
            # beta (2 y + x y + y^2 + beta y + beta x) / ((2 + x + y)(1 + y + beta)), where x is the
            # external rate on the first node and y that on the second; checked as residuals at two
            # operating points together with the four probabilities of a symmetric pair
            "setup": SETUP + """
def closed_form_residual(fn, beta, x, y):
    p = fn(np.array([x]), np.array([y]), beta, 0.0)
    # flat index u * 2 + v: (S, I) is 2 and (S, S) is 3
    rate = beta * p[2] / (p[2] + p[3])
    psi = beta * (2 * y + x * y + y * y + beta * y + beta * x) / ((2 + x + y) * (1 + y + beta))
    return (round(float(rate - psi), 12) + 0.0,)
""",
            "call": "flat(pair_stationary_distribution(np.array([0.3]), np.array([0.3]), 0.6, 0.0)) + closed_form_residual(pair_stationary_distribution, 0.6, 0.3, 0.3) + closed_form_residual(pair_stationary_distribution, 0.9, 0.2, 0.7)",
            "gold_call": "flat(_oracle_pair_stationary_distribution(np.array([0.3]), np.array([0.3]), 0.6, 0.0)) + closed_form_residual(_oracle_pair_stationary_distribution, 0.6, 0.3, 0.3) + closed_form_residual(_oracle_pair_stationary_distribution, 0.9, 0.2, 0.7)",
        },
        {
            # boundary: no external pressure on the oldest stage makes (S(1), S(1)) absorbing, so the
            # distribution sits there up to the floor; lumping the stages of an eight-stage pair whose
            # external rates are equal in every stage reproduces the one-stage distribution
            "setup": SETUP + """
def lumped(fn):
    p8 = fn(np.full(8, 0.4), np.full(8, 0.25), 0.5, 3.0).reshape(9, 9)
    p1 = fn(np.array([0.4]), np.array([0.25]), 0.5, 0.0).reshape(2, 2)
    l = np.array([[p8[0, 0], p8[0, 1:].sum()], [p8[1:, 0].sum(), p8[1:, 1:].sum()]])
    return (round(float(np.abs(l - p1).max()), 12),)
def disease_free(fn):
    p = fn(np.array([0.0, 0.4, 0.6]), np.array([0.0, 0.2, 0.1]), 0.8, 1.1)
    return (round(float(p[1 * 4 + 1]), 12), round(float(p.sum()), 12))
""",
            "call": "flat(lumped(pair_stationary_distribution) + disease_free(pair_stationary_distribution))",
            "gold_call": "flat(lumped(_oracle_pair_stationary_distribution) + disease_free(_oracle_pair_stationary_distribution))",
        },
        {
            # no unique stationary distribution: zero pressure and zero ageing leave every joint
            # susceptible state absorbing; plus malformed inputs and one valid call
            "setup": SETUP + """
z = np.zeros(3)
a = np.array([0.1, 0.2, 0.3])
""",
            "call": "(verdict(pair_stationary_distribution, z, z, 0.5, 0.0), verdict(pair_stationary_distribution, a, a[:2], 0.5, 1.0), verdict(pair_stationary_distribution, a, a, -0.5, 1.0), verdict(pair_stationary_distribution, a, a, 0.5, 1.0))",
            "gold_call": "(verdict(_oracle_pair_stationary_distribution, z, z, 0.5, 0.0), verdict(_oracle_pair_stationary_distribution, a, a[:2], 0.5, 1.0), verdict(_oracle_pair_stationary_distribution, a, a, -0.5, 1.0), verdict(_oracle_pair_stationary_distribution, a, a, 0.5, 1.0))",
        },
    ]
