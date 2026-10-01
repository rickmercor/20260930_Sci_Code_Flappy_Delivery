"""
In single-orbital density matrix embedding, every localized orbital chi_i (site i) is embedded in turn into a quantum bath made of one delocalized orbital b^(i).

In single-orbital density matrix embedding, every localized orbital chi_i (site i) is embedded in turn into a quantum bath made of one delocalized orbital b^(i). The bath is fixed entirely by the reference idempotent one-electron reduced density matrix gamma (per spin) of the full system:

  b^(i) = ( sum_{j != i} gamma_{ij} chi_j ) / sqrt( sum_{j != i} gamma_{ij}^2 ).

Equivalently, the bath is the normalized i-th row of gamma with the impurity component removed, so that the bath is orthogonal to the impurity by construction and carries no amplitude on site i. The same orbital is what the Householder transformation of gamma produces in the conventional construction. Because gamma is idempotent and maps chi_i into the span of chi_i and b^(i), that two-dimensional impurity plus bath space is invariant under gamma, which is what makes the embedding exact for a mean-field state and what guarantees that the cluster holds exactly one electron per spin (a half-filled cluster) whenever 0 < gamma_ii < 1.

If the i-th row of gamma vanishes off the diagonal, site i is decoupled from the rest of the system and no bath exists; this is treated as invalid input.

Returns
-------
np.ndarray of float with shape (L,): normalized bath orbital coefficients, zero on the embedded site and proportional to gamma[site, j] elsewhere.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bath_orbital(gamma, site):
    '''Single bath orbital of site `site` built from the reference 1-RDM.

    Parameters
    ----------
    gamma : array_like of float, shape (L, L)
        Real symmetric per-spin one-electron reduced density matrix of the
        reference determinant.
    site : int
        Index of the embedded orbital, 0 <= site < L.

    Returns
    -------
    b : np.ndarray of float, shape (L,)
        Site-basis coefficients of the normalized bath orbital: b[site] = 0,
        b[j] proportional to gamma[site, j] for j != site, sum_j b[j]^2 = 1.
        Raises ValueError if gamma is not square and symmetric, if site is
        out of range, or if the off-diagonal part of row `site` vanishes
        (norm below 1e-12).
    '''
    return np.zeros(np.asarray(gamma).shape[0], dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bath_orbital(gamma, site):
    gamma = np.asarray(gamma, dtype=float)
    if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1] or gamma.shape[0] < 2:
        raise ValueError("gamma must be a square matrix")
    if not np.all(np.isfinite(gamma)) or not np.allclose(gamma, gamma.T, rtol=0.0, atol=1e-8):
        raise ValueError("gamma must be finite and symmetric")
    if isinstance(site, bool) or not isinstance(site, (int, np.integer)):
        raise ValueError("site must be an integer")
    L = gamma.shape[0]
    if site < 0 or site >= L:
        raise ValueError("site index out of range")
    b = gamma[int(site), :].copy()
    b[int(site)] = 0.0
    norm = np.linalg.norm(b)
    if norm < 1e-12:
        raise ValueError("the embedded site is decoupled: no bath orbital exists")
    return b / norm

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
def _ring(L, t, v):
    h = np.diag(np.asarray(v, dtype=float))
    for i in range(L):
        h[i, (i + 1) % L] -= t
        h[(i + 1) % L, i] -= t
    return h
def _gamma(h, n_occ):
    e, c = np.linalg.eigh(h)
    return c[:, :n_occ] @ c[:, :n_occ].T
g6 = _gamma(_ring(6, 1.0, [-1.0, 2.0, -2.0, 3.0, -3.0, 1.0]), 3)
"""
    return [
        # --- Normal: bath of site 2 of the task ring at the non-interacting level ---
        {
            "setup": setup,
            "call": "bath_orbital(g6, 2)",
            "gold_call": "_oracle_bath_orbital(g6, 2)",
        },
        # --- Normal: bath of site 0 (wraps around the periodic boundary) ---
        {
            "setup": setup,
            "call": "bath_orbital(g6, 0)",
            "gold_call": "_oracle_bath_orbital(g6, 0)",
        },
        # --- Boundary: five-site ring with a single doubly occupied orbital ---
        {
            "setup": setup + "g5 = _gamma(_ring(5, 1.0, [0.3, -0.4, 0.8, -0.2, 0.1]), 1)\n",
            "call": "bath_orbital(g5, 4)",
            "gold_call": "_oracle_bath_orbital(g5, 4)",
        },
        # --- Edge: the bath is normalized and orthogonal to the impurity (encoded check) ---
        {
            "setup": setup + """
def _check(fn):
    b = fn(g6, 3)
    return int(abs(np.sum(b * b) - 1.0) < 1e-12 and abs(b[3]) < 1e-15)
""",
            "call": "_check(bath_orbital)",
            "gold_call": "_check(_oracle_bath_orbital)",
        },
        # --- Invalid: decoupled site (diagonal density matrix) ---
        {
            "setup": setup + """
def run_model():
    try:
        bath_orbital(np.diag([1.0, 0.0, 1.0, 0.0]), 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_bath_orbital(np.diag([1.0, 0.0, 1.0, 0.0]), 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: site index out of range ---
        {
            "setup": setup + """
def run_model():
    try:
        bath_orbital(g6, 6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_bath_orbital(g6, 6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
