"""
Select the one-step transition stencil at a given node of a recombining square-root lattice. The lattice is uniform in the Lamperti coordinate of the factor, anchored at the initial factor level and stepping by lam*sqrt(h), with the factor level clipped at zero, the lattice floor, wherever the Lamperti coordinate is negative. A selected successor index whose Lamperti coordinate is negative is reported as the lowest index whose coordinate is non-negative. A parent index whose factor level is zero carries no transition and is rejected; every node with a strictly positive level, including the lowest one, is a valid parent. The transition from the parent node must reproduce the exact one-step conditional mean and variance of the square-root factor on three lattice nodes carrying non-negative probabilities. Return the three successor indices relative to the parent, in increasing order, together with their transition probabilities. The chosen triplet is the narrowest one admitting such a distribution, and its span may exceed the immediate neighbours of the parent. Only triplets whose lowest index is at or below the parent index and whose highest index is at or above it are considered, so the parent may itself be the lowest or the highest node of the triplet, and a triplet with a repeated factor level is skipped. When several admissible triplets share the smallest index span, take the one whose outer-pair midpoint is nearest the parent index; if that still ties, take the one with the lower outer index, then the lower middle index.

A square-root factor dZ = kappa(theta - Z)dt + sigma sqrt(Z) dW is discretised on a lattice uniform in the Lamperti coordinate chi = 2 sqrt(Z)/sigma, on which the diffusion coefficient is constant. With scale parameter lam and step h the grid is chi_j = chi_0 + j lam sqrt(h) and Z_j = (sigma/2 max{chi_j, 0})^2, so the nodes are spaced lam sigma sqrt(h)/2 apart in sqrt(Z).

The exact one-step conditional moments of the factor are

   m(z) = theta + (z - theta) exp(-kappa h)

    v(z) = sigma^2 z / kappa * exp(-kappa h)(1 - exp(-kappa h))

           + theta sigma^2 / (2 kappa) * (1 - exp(-kappa h))^2

Given three support points Z_1 < Z_2 < Z_3 and centred deviations d_i = Z_i - m(z), the probabilities reproducing unit mass, mean m(z) and variance v(z) are unique:

    p_1 = (v + d_2 d_3) / ((d_1 - d_2)(d_1 - d_3))

    p_2 = (v + d_1 d_3) / ((d_2 - d_1)(d_2 - d_3))

    p_3 = (v + d_1 d_2) / ((d_3 - d_1)(d_3 - d_2))

These are signed: for a given triplet they need not all be non-negative, and admissibility depends on the scale parameter, the factor parameters and the position of the parent node in the lattice. The scale restriction 0 < lam < 2 has a positivity interpretation: if the conditional mean lies between adjacent grid values Z_lo < m < Z_hi, the smallest variance attainable by a grid-supported distribution with that mean is (m - Z_lo)(Z_hi - m), and away from the lower boundary the worst-case ratio of this minimum to the leading conditional variance is asymptotically lam^2/4. Values of lam near two therefore leave little positivity margin, while smaller values increase the number of factor nodes.

Returns
-------
np.ndarray of shape (2, 3): row 0 the three successor indices relative to the parent in increasing order as floats, row 1 the corresponding transition probabilities
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cir_transition_stencil(z_root: float, kappa: float, theta: float, sigma: float,
                           lam: float, h: float, j: int) -> np.ndarray:
    '''Select the one-step transition stencil at node j of a square-root lattice.

    The lattice is uniform in the Lamperti coordinate of the square-root factor,
    anchored at z_root and stepping by lam*sqrt(h). The transition from the
    parent node reproduces the exact one-step conditional mean and variance of
    the factor on three lattice nodes carrying non-negative probabilities.

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
    j : int
        Index of the parent node relative to the lattice anchor. The factor
        level at this index must be strictly above the lattice floor.

    Returns
    -------
    out : np.ndarray
        Array of shape (2, 3). Row 0 holds the three successor indices relative
        to j, in increasing order, as floats. Successor indices below the
        lattice floor are clamped to it. Row 1 holds the corresponding
        transition probabilities.

    Raises
    ------
    ValueError
        If the arguments do not describe a valid square-root lattice, if the
        parent node lies at or below the lattice floor, or if no admissible
        stencil exists at the requested node.
    '''
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _node_value(z_root, sigma, lam, h, j):
    chi = 2.0 * np.sqrt(z_root) / sigma + j * lam * np.sqrt(h)
    return (sigma / 2.0 * max(chi, 0.0)) ** 2


def _cir_moments(z, kappa, theta, sigma, h):
    e = np.exp(-kappa * h)
    m = theta + (z - theta) * e
    v = sigma ** 2 * z / kappa * e * (1.0 - e) + theta * sigma ** 2 / (2.0 * kappa) * (1.0 - e) ** 2
    return m, v


def _triplet_probs(Zs, m, v):
    d1, d2, d3 = (np.asarray(Zs, float) - m)
    den = np.array([(d1 - d2) * (d1 - d3), (d2 - d1) * (d2 - d3), (d3 - d1) * (d3 - d2)])
    if np.min(np.abs(den)) < 1e-300:
        return None
    return np.array([v + d2 * d3, v + d1 * d3, v + d1 * d2]) / den


def _oracle_cir_transition_stencil(z_root: float, kappa: float, theta: float, sigma: float,
                                   lam: float, h: float, j: int) -> np.ndarray:
    for name, val in (("z_root", z_root), ("kappa", kappa), ("theta", theta),
                      ("sigma", sigma), ("h", h)):
        if not isinstance(val, (int, float, np.integer, np.floating)) or not float(val) > 0.0:
            raise ValueError(f"{name} must be a positive real number")
    if not isinstance(lam, (int, float, np.integer, np.floating)) or not 0.0 < float(lam) < 2.0:
        raise ValueError("lam must satisfy 0 < lam < 2")
    if not isinstance(j, (int, np.integer)):
        raise ValueError("j must be an integer node index")

    z_root, kappa, theta = float(z_root), float(kappa), float(theta)
    sigma, lam, h, j = float(sigma), float(lam), float(h), int(j)

    j_min = int(np.ceil(-2.0 * np.sqrt(z_root) / (sigma * lam * np.sqrt(h))))
    z = _node_value(z_root, sigma, lam, h, j)
    if z <= 0.0:
        raise ValueError("parent node lies at or below the lattice floor")
    m, v = _cir_moments(z, kappa, theta, sigma, h)

    for span in range(2, 8):
        cands = [(lo, mid, lo + span)
                 for lo in range(j - span, j + 1)
                 for mid in range(lo + 1, lo + span)]
        cands.sort(key=lambda t: abs((t[0] + t[2]) / 2.0 - j))
        for idx in cands:
            Zs = [_node_value(z_root, sigma, lam, h, i) for i in idx]
            if len(set(np.round(Zs, 15))) < 3:
                continue
            p = _triplet_probs(Zs, m, v)
            if p is None or p.min() < -1e-12:
                continue
            p = np.clip(p, 0.0, None)
            p = p / p.sum()
            rel = np.array([max(i, j_min) - j for i in idx], dtype=float)
            return np.vstack([rel, p])

    raise ValueError("no admissible stencil exists at the requested node")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
z_root, kappa, theta, sigma, lam, h = 0.04, 2.0, 0.04, 0.30, 1.0, 0.25
"""
    rate = """import numpy as np
z_root, kappa, theta, sigma, lam, h = 0.018, 0.45, 0.025, 0.10, 1.4, 0.25
"""
    err = """import numpy as np
def run_model(**kw):
    a = dict(z_root=0.04, kappa=2.0, theta=0.04, sigma=0.30, lam=1.0, h=0.25, j=0)
    a.update(kw)
    try:
        cir_transition_stencil(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold(**kw):
    a = dict(z_root=0.04, kappa=2.0, theta=0.04, sigma=0.30, lam=1.0, h=0.25, j=0)
    a.update(kw)
    try:
        _oracle_cir_transition_stencil(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        {"setup": base,
         "call": "cir_transition_stencil(z_root, kappa, theta, sigma, lam, h, 0)",
         "gold_call": "_oracle_cir_transition_stencil(z_root, kappa, theta, sigma, lam, h, 0)"},
        {"setup": base,
         "call": "cir_transition_stencil(z_root, kappa, theta, sigma, lam, h, 1)",
         "gold_call": "_oracle_cir_transition_stencil(z_root, kappa, theta, sigma, lam, h, 1)"},
        {"setup": base,
         "call": "cir_transition_stencil(z_root, kappa, theta, sigma, lam, h, -2)",
         "gold_call": "_oracle_cir_transition_stencil(z_root, kappa, theta, sigma, lam, h, -2)"},
        {"setup": base,
         "call": "cir_transition_stencil(z_root, kappa, theta, sigma, lam, h, 2)",
         "gold_call": "_oracle_cir_transition_stencil(z_root, kappa, theta, sigma, lam, h, 2)"},
        {"setup": rate,
         "call": "cir_transition_stencil(z_root, kappa, theta, sigma, lam, h, 4)",
         "gold_call": "_oracle_cir_transition_stencil(z_root, kappa, theta, sigma, lam, h, 4)"},
        {"setup": rate,
         "call": "cir_transition_stencil(z_root, kappa, theta, sigma, lam, h, 12)",
         "gold_call": "_oracle_cir_transition_stencil(z_root, kappa, theta, sigma, lam, h, 12)"},
        {"setup": err, "call": "run_model(sigma=0.0)", "gold_call": "run_gold(sigma=0.0)"},
        {"setup": err, "call": "run_model(lam=2.5)", "gold_call": "run_gold(lam=2.5)"},
        {"setup": err, "call": "run_model(j=-9)", "gold_call": "run_gold(j=-9)"},
    ]
