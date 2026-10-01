"""
In densities, the production of explorers in patch j is b_j = sum_i c_i C_ij [S_i] / M_j = sum_i c_i (M_i / M_j) C_ij p_i, with p_i = [S_i] / M_i the fraction of occupied sites. With explorers at their quasi-stationary balance F x = b / lambda, the colonisation attempt rate per site is lambda x = F^(-1) b, and colonisation of patch i proceeds at lambda x_i (1 - p_i). The settled dynamics close on the occupancies alone,

dp_i/dt = -e_i p_i + (1 - p_i) sum_j K_ij c_j p_j,

with the effective colonisation kernel

K_ij = sum_k (F^(-1))_ik (M_j / M_k) C_jk.

The kernel is independent of the occupancies because an attempt on an occupied site also consumes the explorer, so explorer loss by attempts is lambda x_i whatever the occupancy. It depends on the dispersal dynamics through f and g and on the topology through the eigenbasis of the explorer operator. The ratio of site numbers sits inside the sum, at the patch k where the explorers are released, not outside it: the resolvent already carries the size weighting of movement, and a factor M_j / M_i outside the sum counts it twice.

The same kernel is often wanted in counts. At vanishing occupancy one settled individual of patch j generates new settled individuals in patch i at rate K_ij c_j M_i / M_j, which does not involve the site numbers at all. Every explorer ends either in a colonisation attempt or in death, so summing the count kernel over targets gives the colonisation budget sum_i K_ij M_i / M_j = xi q_j / [(1 + 1 / f)(1 + g)] of a source patch.

Settled individuals do not move. A settled individual of patch i releases explorers into patch j at rate c_i C_ij, where c_i is the fecundity of the patch and C_ij the feasibility of exploration from i to j. Explorers are released one exploration step away, along the links of the network, and the feasibility saturates in the exploration efficiency: it vanishes for immobile explorers and approaches a maximal explorability xi for very mobile ones,

C_ij = xi w_ij / (1 + 1 / f),   f = D / lambda.

Returns
-------
dict, the effective colonisation kernel and its release budget, keyed by kernel (the N by N K_ij), count_kernel (K_ij M_i / M_j), release_feasibility (the C_ij), colonisation_budget (the column sums of count_kernel), exploration_efficiency (f), mortality_ratio (g), and budget_residual (the largest modulus of colonisation_budget minus xi q_j / [(1 + 1 / f)(1 + g)]).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def effective_colonisation_kernel(
    weights: np.ndarray,
    sites: np.ndarray,
    max_explorability: float,
    exploration_rate: float,
    colonisation_rate: float,
    explorer_death_rate: float,
) -> dict:
    """Eliminate the quasi-stationary explorers and return the effective colonisation kernel of the settled occupancies.

    Parameters
    ----------
    weights : np.ndarray
        Link weights w_ij.
    sites : np.ndarray
        Site numbers M_i.
    max_explorability : float
        Maximal explorability xi, above zero.
    exploration_rate : float
        Explorer movement rate D, above zero.
    colonisation_rate : float
        Colonisation attempt rate lambda, above zero.
    explorer_death_rate : float
        Explorer death rate gamma, not below zero.

    Returns
    -------
    dict
        Under the keys kernel, count_kernel, release_feasibility, colonisation_budget, exploration_efficiency, mortality_ratio and budget_residual.

    Raises
    ------
    ValueError
        When the network or the site numbers are invalid, when xi, D or lambda fails to be finite and above zero, or when gamma fails to be finite and not below zero.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _check_network(weights, sites):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] < 2:
        raise ValueError("weights must be a square array of at least two patches")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0) or np.any(np.diag(w) != 0.0):
        raise ValueError("weights must be finite, non-negative and zero on the diagonal")
    m = np.asarray(sites, dtype=float)
    if m.shape != (w.shape[0],) or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
        raise ValueError("sites must be finite, above zero and one per patch")
    return w, m


def _oracle_effective_colonisation_kernel(
    weights: np.ndarray,
    sites: np.ndarray,
    max_explorability: float,
    exploration_rate: float,
    colonisation_rate: float,
    explorer_death_rate: float,
) -> dict:
    """Reference implementation."""
    w, m = _check_network(weights, sites)
    xi = float(max_explorability)
    d = float(exploration_rate)
    lam = float(colonisation_rate)
    gamma = float(explorer_death_rate)
    for value in (xi, d, lam):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError("max_explorability, exploration_rate and colonisation_rate must be finite and above zero")
    if not (math.isfinite(gamma) and gamma >= 0.0):
        raise ValueError("explorer_death_rate must be finite and not below zero")
    f = d / lam
    g = gamma / lam

    resolvent = _oracle_explorer_resolvent(w, m, f, g)["resolvent"]  # noqa: F821
    feasibility = xi * w / (1.0 + 1.0 / f)
    release = feasibility * (m[:, None] / m[None, :])
    kernel = resolvent @ release.T
    count_kernel = kernel * (m[:, None] / m[None, :])
    budget = count_kernel.sum(axis=0)
    expected = feasibility.sum(axis=1) / (1.0 + g)
    return {
        "kernel": kernel,
        "count_kernel": count_kernel,
        "release_feasibility": feasibility,
        "colonisation_budget": budget,
        "exploration_efficiency": f,
        "mortality_ratio": g,
        "budget_residual": float(np.max(np.abs(budget - expected))),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    FLAT = """
    def flat(x):
        # flatten to a tuple of plain numeric terminals
        if isinstance(x, (tuple, list)):
            out = []
            for v in x:
                out.extend(flat(v))
            return tuple(out)
        if hasattr(x, "tolist"):
            return flat(x.tolist())
        if isinstance(x, bool):
            return (int(x),)
        return (x,)
    """

    SETUP = """
    import numpy as np
    DOWN = (-1, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 6, 10)
    W = np.zeros((13, 13))
    for i, p in enumerate(DOWN):
        if p >= 0:
            W[i, p] = 1.0
            W[p, i] = 0.3
    W[7, 12] = 0.6
    W[11, 4] = 0.5
    DRAIN = np.array([13, 5, 7, 3, 1, 2, 4, 1, 1, 1, 2, 1, 1], dtype=float)
    M = 40.0 * DRAIN
    def digest(out):
        return (np.round(out["kernel"], 10), np.round(out["count_kernel"], 10), np.round(out["release_feasibility"], 12),
                np.round(out["colonisation_budget"], 10), round(out["exploration_efficiency"], 12),
                round(out["mortality_ratio"], 12), int(out["budget_residual"] < 1e-10))
    """
    import textwrap as _textwrap
    FLAT = _textwrap.dedent(FLAT)
    SETUP = _textwrap.dedent(SETUP)
    return [
        {
            # the river landscape at the graded rates
            "setup": SETUP + FLAT,
            "call": "flat(digest(effective_colonisation_kernel(W, M, 0.8, 1.6, 1.0, 0.25)))",
            "gold_call": "flat(digest(_oracle_effective_colonisation_kernel(W, M, 0.8, 1.6, 1.0, 0.25)))",
        },
        {
            # the count kernel does not depend on the site numbers, while the density kernel does, and
            # doubling lambda together with D and gamma leaves both unchanged
            "setup": SETUP + """
def invariance(fn):
    a = fn(W, M, 0.8, 1.6, 1.0, 0.25)
    b = fn(W, np.full(13, 40.0), 0.8, 1.6, 1.0, 0.25)
    c = fn(W, M, 0.8, 3.2, 2.0, 0.5)
    return (float(np.max(np.abs(a["count_kernel"] - b["count_kernel"]))) < 1e-12,
            float(np.max(np.abs(a["kernel"] - b["kernel"]))) > 1e-3,
            float(np.max(np.abs(a["kernel"] - c["kernel"]))) < 1e-12,
            np.round(b["kernel"][0], 10))
""" + FLAT,
            "call": "flat(invariance(effective_colonisation_kernel))",
            "gold_call": "flat(invariance(_oracle_effective_colonisation_kernel))",
        },
        {
            # boundary g = 0 on an all-to-all network of five identical patches: the kernel must equal the
            # direct solve of the quasi-stationary balance, and with no explorer death every source
            # patch spends its whole release on colonisation attempts
            "setup": """
import numpy as np
W5 = 0.4 * (np.ones((5, 5)) - np.eye(5))
M5 = np.full(5, 25.0)
def closed(fn):
    out = fn(W5, M5, 1.3, 2.5, 0.5, 0.0)
    f, g, xi, w = 5.0, 0.0, 1.3, 0.4
    L = 4 * w * np.eye(5) - W5
    direct = np.linalg.solve((1 + g) * np.eye(5) + f * L, (xi * W5 / (1 + 1 / f)).T)
    return (float(np.max(np.abs(out["kernel"] - direct))) < 1e-12, np.round(out["kernel"][0], 12),
            np.round(out["colonisation_budget"], 12))
""" + FLAT,
            "call": "flat(closed(effective_colonisation_kernel))",
            "gold_call": "flat(closed(_oracle_effective_colonisation_kernel))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(weights=W, sites=M, max_explorability=0.8, exploration_rate=1.6, colonisation_rate=1.0, explorer_death_rate=0.25)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "(verdict(effective_colonisation_kernel, max_explorability=0.0), verdict(effective_colonisation_kernel, exploration_rate=0.0), verdict(effective_colonisation_kernel, colonisation_rate=-1.0), verdict(effective_colonisation_kernel, explorer_death_rate=-0.1), verdict(effective_colonisation_kernel, explorer_death_rate=float('nan')), verdict(effective_colonisation_kernel, sites=M[:4]), verdict(effective_colonisation_kernel, weights=-W), verdict(effective_colonisation_kernel))",
            "gold_call": "(verdict(_oracle_effective_colonisation_kernel, max_explorability=0.0), verdict(_oracle_effective_colonisation_kernel, exploration_rate=0.0), verdict(_oracle_effective_colonisation_kernel, colonisation_rate=-1.0), verdict(_oracle_effective_colonisation_kernel, explorer_death_rate=-0.1), verdict(_oracle_effective_colonisation_kernel, explorer_death_rate=float('nan')), verdict(_oracle_effective_colonisation_kernel, sites=M[:4]), verdict(_oracle_effective_colonisation_kernel, weights=-W), verdict(_oracle_effective_colonisation_kernel))",
        },
    ]
