"""
The ageing rate of the susceptible stages is not a free parameter of the fit. It is tied to the time scale of infection so that the stage clock resolves the interval over which reinfection happens and that resolution sharpens as K grows:

gamma = beta q sqrt(K - 1), with q = (sum over i, j of A[i, j]) / N - 1,

where q is one less than the mean degree of the network. With K = 1 the ageing rate is zero and the stage structure is empty.

The iteration is sequential, edge by edge. A sweep visits every undirected edge (i, j) with i < j in row-major order of the adjacency matrix. For each edge it computes the cavity rates from the messages as they currently stand, including those already updated earlier in the same sweep, solves the pair with i first and j second, where the external rates on i are C[j, i] and those on j are C[i, j], and overwrites phi[i, j] with row 0 and phi[j, i] with row 1 of step 4. Updating in place reaches the same fixed point as a synchronous update, in which every edge of a sweep sees only the messages of the previous sweep, in about half as many sweeps. The sweep repeats until the largest absolute change of any message over a sweep falls below tol. The iteration starts from messages A[i, j] times K values evenly spaced from 0.1 to 0.2 (the single value 0.1 when K = 1). Above the epidemic threshold of the approximation this converges to the endemic fixed point, which is reached from any strictly positive start. Below it the oldest-stage messages decay geometrically towards zero, while the younger-stage messages become ratios of probabilities that vanish together and carry no information; as soon as every oldest-stage message after a sweep lies below 1e-12 the stage therefore stops and returns the disease-free solution, all messages zero, with residual zero. If neither has happened after max_sweeps sweeps the stage raises RuntimeError.

This stage solves the memory-augmented pair approximation on a whole network. Its unknowns are the stage-resolved messages phi[i, j, x], one vector of K rates per directed edge, and its equations close two maps on each other. Given the messages, the non-backtracking operator of step 1 gives, for every edge, the stage-dependent infection rates the rest of the network exerts on each end of the pair. Given those rates, steps 2 to 4 solve the pair chain for its stationary distribution and read new messages off it. A solution is a set of messages that reproduces itself.

Returns
-------
dict holding messages, the converged array phi[i, j, x] of shape (N, N, K) with x = 0 for S(1) and x = K - 1 for S(K), all zero for the disease-free solution; the float ageing_rate, gamma; the float mean_degree_less_one, q; the integer sweeps, the number of sweeps performed; and the float residual, the largest absolute message change in the last sweep.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def memory_pair_messages(
    adjacency: np.ndarray,
    beta: float,
    num_stages: int,
    tol: float,
    max_sweeps: int,
) -> dict:
    """Solve the self-consistent stage-resolved messages of the memory-augmented pair approximation.

    Parameters
    ----------
    adjacency : np.ndarray
        Symmetric 0/1 adjacency matrix of shape (N, N) with zero diagonal.
    beta : float
        Per-contact transmission rate, above zero, with unit recovery rate.
    num_stages : int
        Number K of susceptible stages, at least 1.
    tol : float
        Convergence tolerance on the largest absolute message change per sweep.
    max_sweeps : int
        Maximum number of sweeps.

    Returns
    -------
    dict
        Under the keys messages, ageing_rate, mean_degree_less_one, sweeps and residual.

    Raises
    ------
    ValueError
        When the adjacency is not a simple undirected graph with at least one edge, when beta or tol fails to be finite and above zero, or when num_stages or max_sweeps fails to be an integer at least 1.
    RuntimeError
        When the iteration has not converged after max_sweeps sweeps.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _check_adjacency(adjacency):
    a = np.asarray(adjacency)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 2:
        raise ValueError("adjacency must be a square matrix of size at least 2")
    if not np.all(np.isfinite(a)):
        raise ValueError("adjacency must be finite")
    a = a.astype(float)
    if not np.all((a == 0.0) | (a == 1.0)):
        raise ValueError("adjacency entries must be 0 or 1")
    if not np.array_equal(a, a.T):
        raise ValueError("adjacency must be symmetric")
    if np.any(np.diag(a) != 0.0):
        raise ValueError("adjacency must have a zero diagonal")
    return a


def _check_count(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(name + " must be an integer at least 1")
    return int(value)


def _oracle_memory_pair_messages(
    adjacency: np.ndarray,
    beta: float,
    num_stages: int,
    tol: float,
    max_sweeps: int,
) -> dict:
    """Reference implementation."""
    a = _check_adjacency(adjacency)
    if a.sum() == 0.0:
        raise ValueError("adjacency must contain at least one edge")
    k = _check_count(num_stages, "num_stages")
    sweeps_allowed = _check_count(max_sweeps, "max_sweeps")
    if isinstance(beta, bool) or isinstance(tol, bool):
        raise ValueError("beta and tol must be real numbers")
    beta = float(beta)
    tol = float(tol)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and above zero")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and above zero")

    n = a.shape[0]
    q = a.sum() / n - 1.0
    gamma = beta * q * math.sqrt(k - 1)
    phi = a[:, :, None] * np.linspace(0.1, 0.2, k)[None, None, :]
    first, second = np.nonzero(np.triu(a))

    residual = math.inf
    for sweep in range(1, sweeps_allowed + 1):
        previous = phi.copy()
        for i, j in zip(first, second):
            # cavity rates from the messages as they stand, including updates earlier in this sweep
            cavity = _oracle_nonbacktracking_cavity_rates(a, phi)  # noqa: F821
            rates = _oracle_stage_conditional_infection_rates(cavity[j, i], cavity[i, j], beta, gamma)  # noqa: F821
            phi[i, j] = rates[0]
            phi[j, i] = rates[1]
        residual = float(np.abs(phi - previous).max())
        if phi[:, :, 0].max() < 1e-12:
            # no endemic state: the oldest-stage messages have died out, and the younger-stage
            # messages are conditioned on events of vanishing probability
            return {
                "messages": np.zeros_like(phi),
                "ageing_rate": gamma,
                "mean_degree_less_one": float(q),
                "sweeps": sweep,
                "residual": 0.0,
            }
        if residual < tol:
            return {
                "messages": phi,
                "ageing_rate": gamma,
                "mean_degree_less_one": float(q),
                "sweeps": sweep,
                "residual": residual,
            }
    raise RuntimeError("the message iteration did not converge within max_sweeps")

# =============================================================================
# TEST CASES
# =============================================================================

SETUP = """
import numpy as np

def graph(n, edges):
    A = np.zeros((n, n))
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    return A

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
            # a heterogeneous nine-node graph with a hub, a triangle and a pendant chain, four
            # stages: the converged messages on every directed edge, the ageing rate and q
            "setup": SETUP + """
A = graph(9, [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 5), (3, 6), (6, 7), (4, 8), (5, 8), (1, 7)])
def digest(fn):
    out = fn(A, 0.9, 4, 1e-13, 5000)
    return tuple(round(float(v), 9) for v in out["messages"][A > 0].ravel()) + (
        round(float(out["ageing_rate"]), 12), round(float(out["mean_degree_less_one"]), 12), int(out["residual"] < 1e-13))
""",
            "call": "digest(memory_pair_messages)",
            "gold_call": "digest(_oracle_memory_pair_messages)",
        },
        {
            # closed forms on the Petersen graph, where every edge is equivalent and q = 2: one stage
            # gives phi = beta - 1 / q, and two stages with gamma = beta q satisfy the regular-graph
            # self-consistency for (phi1, phi2) in closed form; both checked as residuals
            "setup": SETUP + """
A = graph(10, [(k, (k + 1) % 5) for k in range(5)] + [(5 + k, 5 + (k + 2) % 5) for k in range(5)] + [(k, 5 + k) for k in range(5)])
def closed(fn, b):
    q = 2.0
    one = fn(A, b, 1, 1e-14, 20000)["messages"]
    two = fn(A, b, 2, 1e-14, 20000)
    p1, p2 = two["messages"][0, 1]
    n1 = b * q * p1 * (b + p2) * (b * q + q * (p1 + p2) + 1)
    d1 = q * p1 ** 2 * (b * q + q * p2 + 1) + p1 * (b + q * p2 * (2 * b * q + q * p2 + 2) + b * q * (b * q + 3) + 1) + b * (b + b * q + q * p2 + 1)
    n2 = b * q * (b + p2) * (b * (b + p2) + p1 * (b + b * q + q * p2 + 1) + q * p1 ** 2)
    d2 = q * p1 ** 2 * (b * q + q * p2 + 1) + p1 * (b + q * p2 * (b + 2 * b * q + q * p2 + 2) + b * q * (b + b * q + 3) + 1) + b * (p2 * (2 * b * q + q * p2 + q + 1) + b * (b * q + q + 2) + 1)
    return (round(float(np.abs(one[A > 0] - (b - 1.0 / q)).max()), 9) + 0.0, round(float(p1 - n1 / d1), 9) + 0.0,
            round(float(p2 - n2 / d2), 9) + 0.0, round(float(two["ageing_rate"] - b * q), 12) + 0.0, round(float(p1), 9), round(float(p2), 9))
""",
            "call": "closed(memory_pair_messages, 0.7) + closed(memory_pair_messages, 0.95)",
            "gold_call": "closed(_oracle_memory_pair_messages, 0.7) + closed(_oracle_memory_pair_messages, 0.95)",
        },
        {
            # boundary: a path of four nodes has q = 1 / 2 and at this rate both the one-stage and the
            # four-stage messages collapse to the disease-free solution, returned as exact zeros;
            # on a wheel of five spokes the eight-stage messages in both directions along a spoke
            # are returned in full, the rim node receiving the larger messages from the hub
            "setup": SETUP + """
P = graph(4, [(0, 1), (1, 2), (2, 3)])
S = graph(6, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
def boundary(fn):
    sub = fn(P, 0.3, 1, 1e-12, 20000)
    sub4 = fn(P, 0.3, 4, 1e-12, 20000)
    wheel = fn(S, 0.6, 8, 1e-13, 20000)
    m = np.concatenate([wheel["messages"][0, 1], wheel["messages"][1, 0]])
    return (round(float(sub["messages"].max()), 9) + 0.0, round(float(sub4["messages"].max()), 9) + 0.0,
            round(float(sub4["residual"]), 9) + 0.0, round(float(sub["mean_degree_less_one"]), 12),
            round(float(wheel["ageing_rate"]), 12)) + tuple(round(float(v), 9) for v in m)
""",
            "call": "boundary(memory_pair_messages)",
            "gold_call": "boundary(_oracle_memory_pair_messages)",
        },
        {
            # malformed inputs, a graph without edges, and one valid call
            "setup": SETUP + """
A = graph(3, [(0, 1), (1, 2)])
E = np.zeros((3, 3))
""",
            "call": "(verdict(memory_pair_messages, E, 0.5, 2, 1e-10, 100), verdict(memory_pair_messages, A, 0.0, 2, 1e-10, 100), verdict(memory_pair_messages, A, 0.5, 0, 1e-10, 100), verdict(memory_pair_messages, A, 0.5, 2.5, 1e-10, 100), verdict(memory_pair_messages, A, 0.5, 2, -1.0, 100), verdict(memory_pair_messages, A, 0.5, 2, 1e-10, 0), verdict(memory_pair_messages, 2.0 * A, 0.5, 2, 1e-10, 100), verdict(memory_pair_messages, A, 1.5, 2, 1e-10, 2000))",
            "gold_call": "(verdict(_oracle_memory_pair_messages, E, 0.5, 2, 1e-10, 100), verdict(_oracle_memory_pair_messages, A, 0.0, 2, 1e-10, 100), verdict(_oracle_memory_pair_messages, A, 0.5, 0, 1e-10, 100), verdict(_oracle_memory_pair_messages, A, 0.5, 2.5, 1e-10, 100), verdict(_oracle_memory_pair_messages, A, 0.5, 2, -1.0, 100), verdict(_oracle_memory_pair_messages, A, 0.5, 2, 1e-10, 0), verdict(_oracle_memory_pair_messages, 2.0 * A, 0.5, 2, 1e-10, 100), verdict(_oracle_memory_pair_messages, A, 1.5, 2, 1e-10, 2000))",
        },
    ]
