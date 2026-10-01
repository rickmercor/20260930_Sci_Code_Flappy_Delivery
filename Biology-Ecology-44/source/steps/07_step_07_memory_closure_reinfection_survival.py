"""
Before the survival function is evaluated, steps 1 to 4 are applied once more to every tie of the chosen node as a certificate of the fixed point: the cavity rates are recomputed from the converged messages, the pair generator and its stationary distribution are rebuilt, the distribution must be annihilated by the generator, and the stage-conditional messages read off it must reproduce the converged messages to within a hundred times tol, and never to less than 1e-8, failing which the stage raises RuntimeError.

The graded quantity is P(Delta_I > t) for the chosen node under the K-stage closure. The stage also returns the ageing rate and q, the node's infection rates in its oldest and youngest stages, its stationary infectious probability and mean susceptible time, the network average of the stationary infectious probability, and, as the memoryless baseline, the same survival probability from the one-stage closure run through steps 5 and 6 with K = 1.

The survival function describes reinfection in an endemic state, so the stage refuses a configuration whose K-stage messages have collapsed to the disease-free solution: it raises ValueError when the chosen node's oldest-stage infection rate is below 1e-8.

This stage answers the question the chain exists for: in the endemic state of the SIS contact process on a given network, as predicted by the pair approximation whose susceptible compartment carries a K-stage memory of the time since recovery, what is the probability that a chosen node, at the moment it recovers, remains uninfected for longer than a time t? It starts from the network and the rates and reruns every earlier stage.

1. Step 1 applies the stage-resolved non-backtracking operator to the edge messages, giving the infection rate each end of a pair feels from the rest of the network.

2. Step 2 assembles the generator of the (K + 1)^2-state chain of a connected pair, with recovery into the youngest susceptible stage, ageing towards the oldest, infection inside the pair at rate beta and stage-dependent infection from outside.

3. Step 3 solves that chain for its stationary distribution.

4. Step 4 reads the stage-conditional messages of both ends off the stationary distribution.

5. Step 5 fixes the ageing rate gamma = beta q sqrt(K - 1) from q, one less than the mean degree, and iterates steps 1 to 4 over every edge until the messages reproduce themselves.

6. Step 6 sums the converged messages into the chosen node's stage rates and evaluates, on its single-node chain, the inter-infection survival function from the youngest stage and the stationary infectious probability.

Returns
-------
dict holding the floats survival, the graded P(Delta_I > t) of the chosen node under the K-stage closure; ageing_rate and mean_degree_less_one; oldest_stage_rate and youngest_stage_rate, the node's infection rates in S(1) and S(K); infected_probability and mean_susceptible_time of the node; infected_fraction, the stationary infectious probability averaged over all nodes; standard_pair_survival, the one-stage survival probability of the same node at the same time; and the integer sweeps of the K-stage iteration.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def memory_closure_reinfection_survival(
    adjacency: np.ndarray,
    beta: float,
    num_stages: int,
    node: int,
    time: float,
    tol: float,
    max_sweeps: int,
) -> dict:
    """Predict a node's probability of staying uninfected for longer than t after recovery under the K-stage memory pair closure.

    Parameters
    ----------
    adjacency : np.ndarray
        Symmetric 0/1 adjacency matrix of shape (N, N) with zero diagonal and no isolated node.
    beta : float
        Per-contact transmission rate, above zero, with unit recovery rate.
    num_stages : int
        Number K of susceptible stages, at least 1.
    node : int
        Index of the chosen node.
    time : float
        Time t since recovery, non-negative.
    tol : float
        Convergence tolerance of the message iteration.
    max_sweeps : int
        Maximum number of sweeps of the message iteration.

    Returns
    -------
    dict
        Under the keys survival, ageing_rate, mean_degree_less_one, oldest_stage_rate, youngest_stage_rate, infected_probability, mean_susceptible_time, infected_fraction, standard_pair_survival and sweeps.

    Raises
    ------
    ValueError
        When the adjacency is not a simple undirected graph without isolated nodes, when beta or tol fails to be finite and above zero, when num_stages or max_sweeps fails to be an integer at least 1, when node fails to be an integer index of the graph, when time fails to be finite and non-negative, or when the closure predicts no endemic state at the chosen node.
    RuntimeError
        When the message iteration has not converged after max_sweeps sweeps, or when the converged messages fail the fixed-point certificate on the ties of the chosen node.
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


def _oracle_memory_closure_reinfection_survival(
    adjacency: np.ndarray,
    beta: float,
    num_stages: int,
    node: int,
    time: float,
    tol: float,
    max_sweeps: int,
) -> dict:
    """Reference implementation."""
    a = _check_adjacency(adjacency)
    n = a.shape[0]
    if np.any(a.sum(axis=1) == 0.0):
        raise ValueError("every node must have at least one edge")
    if isinstance(node, bool) or not isinstance(node, (int, np.integer)) or not 0 <= int(node) < n:
        raise ValueError("node must be an integer index of the graph")
    node = int(node)
    if isinstance(time, bool):
        raise ValueError("time must be a real number")
    t = float(time)
    if not math.isfinite(t) or t < 0.0:
        raise ValueError("time must be finite and non-negative")

    memory = _oracle_memory_pair_messages(a, beta, num_stages, tol, max_sweeps)  # noqa: F821
    phi = memory["messages"]
    rates = phi.sum(axis=1)
    if rates[node, 0] < 1e-8:
        raise ValueError("the closure predicts no endemic state at the chosen node")
    gamma = memory["ageing_rate"]

    # certificate: on every tie of the chosen node the pair distribution must be stationary under
    # its own generator and must reproduce the messages it was built from
    cavity = _oracle_nonbacktracking_cavity_rates(a, phi)  # noqa: F821
    # a converged sweep leaves the messages within tol of their own update, so the certificate is
    # held to a hundred times tol and never to less than 1e-8; the stationarity residual of the
    # linear solve does not depend on tol at all
    fixed_point_tolerance = max(1e-8, 100.0 * float(tol))
    for j in np.nonzero(a[node])[0]:
        first, second = (node, int(j)) if node < j else (int(j), node)
        generator = _oracle_pair_transition_matrix(cavity[second, first], cavity[first, second], beta, gamma)  # noqa: F821
        joint = _oracle_pair_stationary_distribution(cavity[second, first], cavity[first, second], beta, gamma)  # noqa: F821
        messages = _oracle_stage_conditional_infection_rates(cavity[second, first], cavity[first, second], beta, gamma)  # noqa: F821
        if np.abs(joint @ generator).max() > 1e-9:
            raise RuntimeError("a pair distribution of the chosen node is not stationary")
        if max(np.abs(messages[0] - phi[first, second]).max(), np.abs(messages[1] - phi[second, first]).max()) > fixed_point_tolerance:
            raise RuntimeError("the messages on a tie of the chosen node are not a fixed point")
    chosen = _oracle_reinfection_survival(rates[node], gamma, np.array([t]))  # noqa: F821
    fraction = float(np.mean([
        _oracle_reinfection_survival(rates[i], gamma, np.array([0.0]))["infected_probability"]  # noqa: F821
        for i in range(n)
    ]))

    baseline = _oracle_memory_pair_messages(a, beta, 1, tol, max_sweeps)  # noqa: F821
    baseline_rate = baseline["messages"].sum(axis=1)[node]
    if baseline_rate[0] < 1e-8:
        raise ValueError("the one-stage closure predicts no endemic state at the chosen node")
    standard = _oracle_reinfection_survival(baseline_rate, 0.0, np.array([t]))  # noqa: F821

    return {
        "survival": float(chosen["survival"][0]),
        "ageing_rate": float(gamma),
        "mean_degree_less_one": memory["mean_degree_less_one"],
        "oldest_stage_rate": float(rates[node, 0]),
        "youngest_stage_rate": float(rates[node, -1]),
        "infected_probability": chosen["infected_probability"],
        "mean_susceptible_time": chosen["mean_susceptible_time"],
        "infected_fraction": fraction,
        "standard_pair_survival": float(standard["survival"][0]),
        "sweeps": memory["sweeps"],
    }

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

KEYS = ("survival", "ageing_rate", "mean_degree_less_one", "oldest_stage_rate", "youngest_stage_rate",
        "infected_probability", "mean_susceptible_time", "infected_fraction", "standard_pair_survival")

def digest(out):
    return tuple(round(float(out[k]), 9) + 0.0 for k in KEYS)

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
            # a twelve-node contact network with a hub of degree six, two triangles and a chain,
            # eight stages: every returned quantity for the hub
            "setup": SETUP + """
A = graph(12, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 2), (3, 7), (7, 8), (8, 9), (4, 10), (10, 11), (5, 6), (9, 11), (2, 8)])
""",
            "call": "digest(memory_closure_reinfection_survival(A, 1.0, 8, 0, 1.5, 1e-13, 5000))",
            "gold_call": "digest(_oracle_memory_closure_reinfection_survival(A, 1.0, 8, 0, 1.5, 1e-13, 5000))",
        },
        {
            # the Petersen graph: with one stage the memory closure and its baseline coincide and
            # the survival is exp(-3 (beta - 1/2) t); with three stages the survival exceeds the
            # baseline at a late time, and every node has the same infectious probability
            "setup": SETUP + """
A = graph(10, [(k, (k + 1) % 5) for k in range(5)] + [(5 + k, 5 + (k + 2) % 5) for k in range(5)] + [(k, 5 + k) for k in range(5)])
def regular(fn):
    one = fn(A, 0.8, 1, 7, 2.0, 1e-14, 20000)
    three = fn(A, 0.8, 3, 7, 3.0, 1e-14, 20000)
    return (round(float(one["survival"] - np.exp(-3 * 0.3 * 2.0)), 9) + 0.0, round(float(one["survival"] - one["standard_pair_survival"]), 12) + 0.0,
            round(float(one["infected_fraction"] - one["infected_probability"]), 9) + 0.0, int(three["survival"] > three["standard_pair_survival"])) + digest(three)
""",
            "call": "regular(memory_closure_reinfection_survival)",
            "gold_call": "regular(_oracle_memory_closure_reinfection_survival)",
        },
        {
            # boundary: t = 0 gives survival one; a leaf of a star at the same rates; and a run at a
            # much looser convergence tolerance, which must still return and agree with the tight one
            "setup": SETUP + """
S = graph(7, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 2), (3, 4)])
def loose(fn):
    tight = fn(S, 0.9, 4, 6, 2.0, 1e-13, 20000)["survival"]
    return tuple(int(abs(fn(S, 0.9, 4, 6, 2.0, t, 20000)["survival"] - tight) < 100.0 * t)
                 for t in (1e-6, 1e-8, 1e-10))
""",
            "call": "digest(memory_closure_reinfection_survival(S, 0.9, 4, 0, 0.0, 1e-13, 20000)) + digest(memory_closure_reinfection_survival(S, 0.9, 4, 6, 2.0, 1e-13, 20000)) + loose(memory_closure_reinfection_survival)",
            "gold_call": "digest(_oracle_memory_closure_reinfection_survival(S, 0.9, 4, 0, 0.0, 1e-13, 20000)) + digest(_oracle_memory_closure_reinfection_survival(S, 0.9, 4, 6, 2.0, 1e-13, 20000)) + loose(_oracle_memory_closure_reinfection_survival)",
        },
        {
            # a subcritical ring, an isolated node, a bad node index, a negative time, and one valid call
            "setup": SETUP + """
R = graph(6, [(k, (k + 1) % 6) for k in range(6)])
Iso = graph(4, [(0, 1), (1, 2)])
T = graph(4, [(0, 1), (1, 2), (2, 0), (2, 3)])
""",
            "call": "(verdict(memory_closure_reinfection_survival, R, 0.4, 3, 0, 1.0, 1e-12, 20000), verdict(memory_closure_reinfection_survival, Iso, 0.9, 2, 0, 1.0, 1e-12, 2000), verdict(memory_closure_reinfection_survival, T, 1.5, 2, 4, 1.0, 1e-12, 2000), verdict(memory_closure_reinfection_survival, T, 1.5, 2, 0, -1.0, 1e-12, 2000), verdict(memory_closure_reinfection_survival, T, 1.5, 2, 0, 1.0, 1e-12, 2000))",
            "gold_call": "(verdict(_oracle_memory_closure_reinfection_survival, R, 0.4, 3, 0, 1.0, 1e-12, 20000), verdict(_oracle_memory_closure_reinfection_survival, Iso, 0.9, 2, 0, 1.0, 1e-12, 2000), verdict(_oracle_memory_closure_reinfection_survival, T, 1.5, 2, 4, 1.0, 1e-12, 2000), verdict(_oracle_memory_closure_reinfection_survival, T, 1.5, 2, 0, -1.0, 1e-12, 2000), verdict(_oracle_memory_closure_reinfection_survival, T, 1.5, 2, 0, 1.0, 1e-12, 2000))",
        },
    ]
