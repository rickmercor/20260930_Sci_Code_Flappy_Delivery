"""
Determine which nodes of a square-root lattice are reachable from the anchor at each numerical time step. Starting from the anchor alone, repeatedly apply the transition stencil selected by cir_transition_stencil, including its tie rule, at every node already reached and collect the successors. The reachable set is contiguous at every time level, so return its lowest and highest index at each level. The set does not simply widen by one node per step: stencils may point away from the parent, successors are clamped at the lattice floor, and strong mean reversion can stop the set widening altogether.

A recombining lattice for a square-root factor is defined on an unbounded index set, but only a finite part of it is visited over a finite horizon, and identifying that part is a prerequisite for allocating the backward recursion.

The naive expectation is that the reachable set widens by one index in each direction per step, giving 2n+1 nodes after n steps. That expectation fails for these lattices in both directions. Downward, node values are clamped at the lattice floor, where the Lamperti coordinate would otherwise go negative, so successors below the floor collapse onto it and the set stops descending. Upward and downward alike, the transition stencil is chosen for admissibility rather than symmetry and may lie entirely to one side of its parent, so an outer node can have all three successors strictly inside the current set. When mean reversion is strong relative to the step size, this happens at both extremes and the reachable set saturates: it stops growing after a few steps and remains fixed for the rest of the horizon however long the contract runs.

The practical consequence is not only that a naive enumeration wastes memory. It also asks for stencils at indices where the factor level is zero or the lattice is not defined, and the admissibility search has nothing to return there. The reachable set must therefore be built by iteration from the anchor, applying the same stencil selection the valuation itself will use, rather than assumed from the step count.

Because each stencil returns three ordered successors and the union over a contiguous parent set is itself contiguous, the reachable set at every level is an interval of indices and is fully described by its endpoints.

Returns
-------
np.ndarray of shape (2, n_steps + 1): row 0 the lowest reachable lattice index at each time level, row 1 the highest, as native floats
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reachable_factor_nodes(z_root: float, kappa: float, theta: float, sigma: float,
                           lam: float, h: float, n_steps: int) -> np.ndarray:
    '''Lowest and highest reachable lattice index at each time level.

    Parameters
    ----------
    z_root : float
        Strictly positive factor level at which the lattice is anchored.
    kappa : float
        Strictly positive mean-reversion speed.
    theta : float
        Strictly positive long-run level.
    sigma : float
        Strictly positive diffusion coefficient.
    lam : float
        Lattice scale parameter, strictly between 0 and 2.
    h : float
        Strictly positive time step.
    n_steps : int
        Non-negative number of steps to advance.

    Returns
    -------
    span : np.ndarray
        Shape (2, n_steps + 1). Row 0 holds the lowest reachable index at each
        time level and row 1 the highest, as floats.

    Raises
    ------
    ValueError
        If the arguments do not describe a valid square-root lattice.
    '''
    return span

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _node(z_root, sigma, lam, h, j):
    chi = 2.0 * np.sqrt(z_root) / sigma + j * lam * np.sqrt(h)
    return (sigma / 2.0 * max(chi, 0.0)) ** 2


def _moments(z, kappa, theta, sigma, h):
    e = np.exp(-kappa * h)
    return (theta + (z - theta) * e,
            sigma ** 2 * z / kappa * e * (1.0 - e) + theta * sigma ** 2 / (2.0 * kappa) * (1.0 - e) ** 2)


def _probs(Zs, m, v):
    d1, d2, d3 = (np.asarray(Zs, float) - m)
    den = np.array([(d1 - d2) * (d1 - d3), (d2 - d1) * (d2 - d3), (d3 - d1) * (d3 - d2)])
    if np.min(np.abs(den)) < 1e-300:
        return None
    return np.array([v + d2 * d3, v + d1 * d3, v + d1 * d2]) / den


def _children(z_root, kappa, theta, sigma, lam, h, j, j_min):
    z = _node(z_root, sigma, lam, h, j)
    m, v = _moments(z, kappa, theta, sigma, h)
    for span in range(2, 8):
        cands = [(lo, mid, lo + span) for lo in range(j - span, j + 1)
                 for mid in range(lo + 1, lo + span)]
        cands.sort(key=lambda t: abs((t[0] + t[2]) / 2.0 - j))
        for idx in cands:
            Zs = [_node(z_root, sigma, lam, h, i) for i in idx]
            if len(set(np.round(Zs, 15))) < 3:
                continue
            p = _probs(Zs, m, v)
            if p is None or p.min() < -1e-12:
                continue
            return [max(i, j_min) for i in idx]
    raise ValueError("no admissible stencil exists at a reachable node")


def _oracle_reachable_factor_nodes(z_root: float, kappa: float, theta: float, sigma: float,
                                   lam: float, h: float, n_steps: int) -> np.ndarray:
    for name, val in (("z_root", z_root), ("kappa", kappa), ("theta", theta),
                      ("sigma", sigma), ("h", h)):
        if not isinstance(val, (int, float, np.integer, np.floating)) or not float(val) > 0.0:
            raise ValueError(f"{name} must be a positive real number")
    if not isinstance(lam, (int, float, np.integer, np.floating)) or not 0.0 < float(lam) < 2.0:
        raise ValueError("lam must satisfy 0 < lam < 2")
    if not isinstance(n_steps, (int, np.integer)) or int(n_steps) < 0:
        raise ValueError("n_steps must be a non-negative integer")

    z_root, kappa, theta = float(z_root), float(kappa), float(theta)
    sigma, lam, h, N = float(sigma), float(lam), float(h), int(n_steps)
    j_min = int(np.ceil(-2.0 * np.sqrt(z_root) / (sigma * lam * np.sqrt(h))))

    span = np.zeros((2, N + 1), dtype=float)
    current = {0}
    span[0, 0] = span[1, 0] = 0.0
    for n in range(N):
        nxt = set()
        for j in sorted(current):
            nxt.update(_children(z_root, kappa, theta, sigma, lam, h, j, j_min))
        current = nxt
        span[0, n + 1] = float(min(current))
        span[1, n + 1] = float(max(current))
    return span

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pre = """import numpy as np
vv = (0.04, 2.0, 0.04, 0.30, 1.0, 0.25)
xx = (0.018, 0.45, 0.025, 0.10, 1.4, 0.25)
"""
    err = pre + """
def run_model(**kw):
    a = dict(z_root=0.04, kappa=2.0, theta=0.04, sigma=0.30, lam=1.0, h=0.25, n_steps=12)
    a.update(kw)
    try:
        reachable_factor_nodes(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold(**kw):
    a = dict(z_root=0.04, kappa=2.0, theta=0.04, sigma=0.30, lam=1.0, h=0.25, n_steps=12)
    a.update(kw)
    try:
        _oracle_reachable_factor_nodes(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        {"setup": pre, "call": "reachable_factor_nodes(*vv, 12)",
         "gold_call": "_oracle_reachable_factor_nodes(*vv, 12)"},
        {"setup": pre, "call": "reachable_factor_nodes(*xx, 12)",
         "gold_call": "_oracle_reachable_factor_nodes(*xx, 12)"},
        {"setup": pre, "call": "reachable_factor_nodes(*vv, 40)",
         "gold_call": "_oracle_reachable_factor_nodes(*vv, 40)"},
        {"setup": pre, "call": "reachable_factor_nodes(*vv, 0)",
         "gold_call": "_oracle_reachable_factor_nodes(*vv, 0)"},
        {"setup": pre, "call": "reachable_factor_nodes(*vv, 1)",
         "gold_call": "_oracle_reachable_factor_nodes(*vv, 1)"},
        {"setup": pre, "call": "reachable_factor_nodes(0.004, 0.20, 0.06, 0.15, 1.2, 1.0, 8)",
         "gold_call": "_oracle_reachable_factor_nodes(0.004, 0.20, 0.06, 0.15, 1.2, 1.0, 8)"},
        {"setup": err, "call": "run_model(n_steps=-1)", "gold_call": "run_gold(n_steps=-1)"},
        {"setup": err, "call": "run_model(lam=2.0)", "gold_call": "run_gold(lam=2.0)"},
        {"setup": err, "call": "run_model(sigma=-0.3)", "gold_call": "run_gold(sigma=-0.3)"},
    ]
