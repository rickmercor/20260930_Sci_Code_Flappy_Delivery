"""
Two quantities follow from it. The first is the stationary probability that the node is infectious, obtained from the stationary distribution of the (K + 1)-state chain. The second is the distribution of the inter-infection time Delta_I, the time from the moment the node recovers to the moment it is next infected. At recovery the node is in S(K). Let R be the generator of the chain with I made absorbing: R[x, x - 1] = gamma for x at least 2, R[x, I] = lambda[x], the diagonal set so that every row of the susceptible block sums to zero, and the row of I zero. Then the entry of exp(t R) from S(K) to I is the probability that the node has been reinfected by time t, and the survival function is

P(Delta_I > t) = 1 - [exp(t R)]_{S(K), I}.

For K = 1 this is the exponential exp(-lambda t) of the ordinary pair approximation, whereas for K at least 2 it is a mixture that decays quickly at first, while the node is young and likely to have infectious neighbours, and more slowly later. The mean susceptible time E[Delta_I] is the expected time to absorption from S(K); because each infectious period has unit mean, renewal gives P(I) = 1 / (1 + E[Delta_I]), which ties the two quantities together.

Once the messages have converged, a single node i can be treated as an isolated Markov chain on its K + 1 states. In susceptible stage S(x) it is infected at the total stage-x rate lambda[x] = sum over j of A[i, j] phi[i, j, x], it ages from S(x) to S(x - 1) at rate gamma for x at least 2, and when infectious it recovers at unit rate into the youngest stage S(K). This chain carries the dynamic correlation that the pair closure has learned: the node's infection pressure depends on how long it has been susceptible.

Returns
-------
dict holding survival, an array of P(Delta_I > t) at the supplied times; the float infected_probability, the stationary probability that the node is infectious; and the float mean_susceptible_time, E[Delta_I].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reinfection_survival(stage_rates: np.ndarray, ageing_rate: float, times: np.ndarray) -> dict:
    """Evaluate a node's inter-infection survival function and stationary infectious probability.

    Parameters
    ----------
    stage_rates : np.ndarray
        Infection rates lambda[x] of the node in stages S(1) to S(K), length K.
    ageing_rate : float
        Ageing rate between successive susceptible stages.
    times : np.ndarray
        Non-negative times at which to evaluate P(Delta_I > t).

    Returns
    -------
    dict
        Under the keys survival, infected_probability and mean_susceptible_time.

    Raises
    ------
    ValueError
        When the stage rates fail to be a finite non-negative one-dimensional array of length at least 1 with a positive oldest-stage rate, when the ageing rate fails to be finite and non-negative or is zero with K at least 2, or when the times fail to be a finite non-negative one-dimensional array with at least one entry.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from scipy.linalg import expm


def _oracle_reinfection_survival(stage_rates: np.ndarray, ageing_rate: float, times: np.ndarray) -> dict:
    """Reference implementation."""
    lam = np.asarray(stage_rates)
    if lam.ndim != 1 or lam.size < 1 or np.iscomplexobj(lam):
        raise ValueError("stage_rates must be a real one-dimensional array of length at least 1")
    lam = lam.astype(float)
    if not np.all(np.isfinite(lam)) or np.any(lam < 0.0) or lam[0] <= 0.0:
        raise ValueError("stage_rates must be finite, non-negative and positive in the oldest stage")
    if isinstance(ageing_rate, bool):
        raise ValueError("ageing_rate must be a real number")
    gamma = float(ageing_rate)
    k = lam.size
    if not math.isfinite(gamma) or gamma < 0.0 or (k >= 2 and gamma == 0.0):
        raise ValueError("ageing_rate must be finite, non-negative, and above zero when K is at least 2")
    t = np.asarray(times)
    if t.ndim != 1 or t.size < 1 or np.iscomplexobj(t):
        raise ValueError("times must be a real one-dimensional array with at least one entry")
    t = t.astype(float)
    if not np.all(np.isfinite(t)) or np.any(t < 0.0):
        raise ValueError("times must be finite and non-negative")

    # states: 0 is I, x is S(x)
    size = k + 1
    absorbing = np.zeros((size, size))
    for x in range(1, size):
        absorbing[x, 0] = lam[x - 1]
        if x > 1:
            absorbing[x, x - 1] = gamma
    absorbing -= np.diag(absorbing.sum(axis=1))
    absorbing[0, :] = 0.0
    survival = np.array([1.0 - expm(s * absorbing)[k, 0] for s in t])

    # expected absorption time from S(K): solve (-R_SS) m = 1 on the susceptible block
    block = -absorbing[1:, 1:]
    mean_times = np.linalg.solve(block, np.ones(k))
    mean_susceptible = float(mean_times[k - 1])

    generator = absorbing.copy()
    generator[0, k] = 1.0
    generator[0, 0] = -1.0
    system = generator.T.copy()
    system[-1, :] = 1.0
    rhs = np.zeros(size)
    rhs[-1] = 1.0
    stationary = np.linalg.solve(system, rhs)
    return {
        "survival": survival,
        "infected_probability": float(stationary[0]),
        "mean_susceptible_time": mean_susceptible,
    }

# =============================================================================
# TEST CASES
# =============================================================================

SETUP = """
import math
import numpy as np

def digest(out):
    return tuple(round(float(v), 12) for v in out["survival"]) + (
        round(out["infected_probability"], 12), round(out["mean_susceptible_time"], 12))

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
            # eight stages with a young-stage excess of pressure, the shape the endemic closure
            # produces for a hub, evaluated on a grid of times
            "setup": SETUP + """
lam = np.array([0.97, 1.04, 1.07, 1.12, 1.19, 1.28, 1.40, 1.55])
t = np.array([0.0, 0.5, 1.0, 2.5, 6.0])
""",
            "call": "digest(reinfection_survival(lam, 2.85, t))",
            "gold_call": "digest(_oracle_reinfection_survival(lam, 2.85, t))",
        },
        {
            # K = 1 is exponential with P(I) = lambda / (1 + lambda); equal rates in every stage
            # make the ageing irrelevant; the renewal identity P(I) (1 + E[Delta_I]) = 1 holds for
            # arbitrary stage rates; all checked as residuals
            "setup": SETUP + """
def identities(fn):
    t = np.array([0.3, 1.7])
    one = fn(np.array([0.8]), 0.0, t)
    flat = fn(np.full(6, 0.8), 3.3, t)
    rough = fn(np.array([0.2, 1.9, 0.4, 2.6]), 0.7, t)
    return (round(float(np.abs(one["survival"] - np.exp(-0.8 * t)).max()), 12) + 0.0,
            round(one["infected_probability"] - 0.8 / 1.8, 12) + 0.0,
            round(float(np.abs(flat["survival"] - one["survival"]).max()), 12) + 0.0,
            round(rough["infected_probability"] * (1.0 + rough["mean_susceptible_time"]) - 1.0, 12) + 0.0,
            round(rough["mean_susceptible_time"], 12))
""",
            "call": "identities(reinfection_survival)",
            "gold_call": "identities(_oracle_reinfection_survival)",
        },
        {
            # boundary: at t = 0 the survival is one; very slow ageing leaves the node in S(K) so the
            # survival is exp(-lambda[K] t); very fast ageing sends it to S(1) so it approaches
            # exp(-lambda[1] t)
            "setup": SETUP + """
def limits(fn):
    lam = np.array([0.5, 0.9, 1.6])
    t = np.array([0.0, 1.2])
    slow = fn(lam, 1e-9, t)
    fast = fn(lam, 1e6, t)
    return (round(float(slow["survival"][0]), 12), round(float(slow["survival"][1] - math.exp(-1.6 * 1.2)), 7) + 0.0,
            round(float(fast["survival"][1] - math.exp(-0.5 * 1.2)), 4) + 0.0)
""",
            "call": "limits(reinfection_survival)",
            "gold_call": "limits(_oracle_reinfection_survival)",
        },
        {
            # malformed inputs and one valid call
            "setup": SETUP + """
lam = np.array([0.4, 0.6])
t = np.array([1.0])
""",
            "call": "(verdict(reinfection_survival, np.array([0.0, 0.6]), 1.0, t), verdict(reinfection_survival, lam, 0.0, t), verdict(reinfection_survival, lam, -1.0, t), verdict(reinfection_survival, lam, 1.0, np.array([-1.0])), verdict(reinfection_survival, lam, 1.0, np.array([])), verdict(reinfection_survival, np.array([[0.4]]), 1.0, t), verdict(reinfection_survival, lam, 1.0, t))",
            "gold_call": "(verdict(_oracle_reinfection_survival, np.array([0.0, 0.6]), 1.0, t), verdict(_oracle_reinfection_survival, lam, 0.0, t), verdict(_oracle_reinfection_survival, lam, -1.0, t), verdict(_oracle_reinfection_survival, lam, 1.0, np.array([-1.0])), verdict(_oracle_reinfection_survival, lam, 1.0, np.array([])), verdict(_oracle_reinfection_survival, np.array([[0.4]]), 1.0, t), verdict(_oracle_reinfection_survival, lam, 1.0, t))",
        },
    ]
