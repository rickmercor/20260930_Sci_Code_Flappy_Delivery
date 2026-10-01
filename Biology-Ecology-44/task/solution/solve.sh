#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
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


def nonbacktracking_cavity_rates(adjacency: np.ndarray, messages: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    a = _check_adjacency(adjacency)
    n = a.shape[0]
    phi = np.asarray(messages)
    if phi.ndim != 3 or phi.shape[0] != n or phi.shape[1] != n or phi.shape[2] < 1:
        raise ValueError("messages must have shape (N, N, K) with K at least 1")
    if np.iscomplexobj(phi) or not np.all(np.isfinite(phi)):
        raise ValueError("messages must be real and finite")
    phi = phi.astype(float)
    if np.any(phi < 0.0):
        raise ValueError("messages must be non-negative")
    if np.any(phi[a == 0.0] != 0.0):
        raise ValueError("messages must vanish off the edge set")

    # total[j, x] is the stage-x pressure on j from all its neighbours; removing phi[j, i, x]
    # leaves the pressure from outside the pair (i, j)
    total = phi.sum(axis=1)
    cavity = total[None, :, :] - np.transpose(phi, (1, 0, 2))
    return cavity * a[:, :, None]

import math

import numpy as np


def pair_transition_matrix(
    external_first: np.ndarray,
    external_second: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Reference implementation."""
    a = np.asarray(external_first)
    b = np.asarray(external_second)
    if a.ndim != 1 or b.ndim != 1 or a.size < 1 or a.size != b.size:
        raise ValueError("external rates must be one-dimensional arrays of equal length at least 1")
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        raise ValueError("external rates must be real")
    a = a.astype(float)
    b = b.astype(float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("external rates must be finite")
    if np.any(a < 0.0) or np.any(b < 0.0):
        raise ValueError("external rates must be non-negative")
    if isinstance(beta, bool) or isinstance(gamma, bool):
        raise ValueError("beta and gamma must be real numbers")
    beta = float(beta)
    gamma = float(gamma)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and above zero")
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    k = a.size
    size = k + 1
    q = np.zeros((size * size, size * size))

    def _index(u, v):
        return u * size + v

    # both infectious: either node recovers into the youngest stage
    q[_index(0, 0), _index(0, k)] += 1.0
    q[_index(0, 0), _index(k, 0)] += 1.0
    for x in range(1, k + 1):
        # one node infectious, the other in stage x
        q[_index(x, 0), _index(x, k)] += 1.0
        q[_index(0, x), _index(k, x)] += 1.0
        q[_index(x, 0), _index(0, 0)] += beta + a[x - 1]
        q[_index(0, x), _index(0, 0)] += beta + b[x - 1]
        if x > 1:
            q[_index(x, 0), _index(x - 1, 0)] += gamma
            q[_index(0, x), _index(0, x - 1)] += gamma
        # both susceptible
        for y in range(1, k + 1):
            q[_index(x, y), _index(0, y)] += a[x - 1]
            q[_index(x, y), _index(x, 0)] += b[y - 1]
            if x > 1:
                q[_index(x, y), _index(x - 1, y)] += gamma
            if y > 1:
                q[_index(x, y), _index(x, y - 1)] += gamma
    q -= np.diag(q.sum(axis=1))
    return q

import numpy as np


def pair_stationary_distribution(
    external_first: np.ndarray,
    external_second: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Reference implementation."""
    q = pair_transition_matrix(external_first, external_second, beta, gamma)  # noqa: F821
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

import numpy as np


def stage_conditional_infection_rates(
    external_first: np.ndarray,
    external_second: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Reference implementation."""
    k = np.asarray(external_first).size
    if np.asarray(external_first).ndim != 1 or np.asarray(external_second).shape != (k,) or k < 1:
        raise ValueError("external rates must be one-dimensional arrays of equal length at least 1")
    p = pair_stationary_distribution(external_first, external_second, beta, gamma)  # noqa: F821
    joint = p.reshape(k + 1, k + 1)
    beta = float(beta)
    # joint[u, v]: u is the state of the first node, v of the second; column 0 and row 0 are I
    to_first = beta * joint[1:, 0] / joint[1:, :].sum(axis=1)
    to_second = beta * joint[0, 1:] / joint[:, 1:].sum(axis=0)
    return np.vstack([to_first, to_second])

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


def memory_pair_messages(
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
            cavity = nonbacktracking_cavity_rates(a, phi)  # noqa: F821
            rates = stage_conditional_infection_rates(cavity[j, i], cavity[i, j], beta, gamma)  # noqa: F821
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

import math

import numpy as np
from scipy.linalg import expm


def reinfection_survival(stage_rates: np.ndarray, ageing_rate: float, times: np.ndarray) -> dict:
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


def memory_closure_reinfection_survival(
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

    memory = memory_pair_messages(a, beta, num_stages, tol, max_sweeps)  # noqa: F821
    phi = memory["messages"]
    rates = phi.sum(axis=1)
    if rates[node, 0] < 1e-8:
        raise ValueError("the closure predicts no endemic state at the chosen node")
    gamma = memory["ageing_rate"]

    # certificate: on every tie of the chosen node the pair distribution must be stationary under
    # its own generator and must reproduce the messages it was built from
    cavity = nonbacktracking_cavity_rates(a, phi)  # noqa: F821
    # a converged sweep leaves the messages within tol of their own update, so the certificate is
    # held to a hundred times tol and never to less than 1e-8; the stationarity residual of the
    # linear solve does not depend on tol at all
    fixed_point_tolerance = max(1e-8, 100.0 * float(tol))
    for j in np.nonzero(a[node])[0]:
        first, second = (node, int(j)) if node < j else (int(j), node)
        generator = pair_transition_matrix(cavity[second, first], cavity[first, second], beta, gamma)  # noqa: F821
        joint = pair_stationary_distribution(cavity[second, first], cavity[first, second], beta, gamma)  # noqa: F821
        messages = stage_conditional_infection_rates(cavity[second, first], cavity[first, second], beta, gamma)  # noqa: F821
        if np.abs(joint @ generator).max() > 1e-9:
            raise RuntimeError("a pair distribution of the chosen node is not stationary")
        if max(np.abs(messages[0] - phi[first, second]).max(), np.abs(messages[1] - phi[second, first]).max()) > fixed_point_tolerance:
            raise RuntimeError("the messages on a tie of the chosen node are not a fixed point")
    chosen = reinfection_survival(rates[node], gamma, np.array([t]))  # noqa: F821
    fraction = float(np.mean([
        reinfection_survival(rates[i], gamma, np.array([0.0]))["infected_probability"]  # noqa: F821
        for i in range(n)
    ]))

    baseline = memory_pair_messages(a, beta, 1, tol, max_sweeps)  # noqa: F821
    baseline_rate = baseline["messages"].sum(axis=1)[node]
    if baseline_rate[0] < 1e-8:
        raise ValueError("the one-stage closure predicts no endemic state at the chosen node")
    standard = reinfection_survival(baseline_rate, 0.0, np.array([t]))  # noqa: F821

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
SCICODE_GOLD_EOF
