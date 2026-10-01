"""
Return the indices of the candidate geometries that the workflow's two-objective geometry-selection stage keeps, given one energy and one reactive separation per candidate.

A global search over adsorption geometries returns far more structures than a combinatorial pairing can use, and the structures that matter for a surface reaction are characterised by two competing properties. They should be low in energy, because a high-energy adsorption motif is not populated and a path through it is not the minimum-energy path. They should also place the two reacting centres close together, because a reaction between centres that are far apart proceeds through diffusion rather than through the elementary step that was asked about. The two properties genuinely compete, so there is no single best structure, only a trade-off between the two objectives.



Conventions fixed by this task. The energy window is measured from the lowest energy in the supplied set, and the distance window is an absolute cut-off on the separation. One candidate dominates another when it is no worse in both objectives and strictly better in at least one. A candidate lies within the tolerance window of a point when its energy and its separation each differ from that point's by no more than the respective tolerance, the bounds being inclusive. Each front is removed from the working set before the next one is taken, the requested number of fronts is taken or fewer if nothing is left, and the union is returned in ascending index order.

With $E_i$ and $d_i$ the energy and reactive separation of candidate $i$, candidate $j$ dominates $i$ when



$$E_j\\le E_i\\;\\wedge\\;d_j\\le d_i\\;\\wedge\\;(E_j<E_i\\;\\vee\\;d_j<d_i),$$



and $i$ lies within the tolerance window of $p$ when $|E_i-E_p|\\le \\varepsilon_E$ and $|d_i-d_p|\\le \\varepsilon_d$.

Returns
-------
A one-dimensional `numpy` integer array of the retained candidate indices, in ascending order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_pareto_geometries(energies: "np.ndarray",
                             distances: "np.ndarray",
                             e_window: float = 1.2, d_window: float = 4.5,
                             tol_e: float = 0.1, tol_d: float = 0.1,
                             n_fronts: int = 2) -> "np.ndarray":
    """Two-objective front selection of candidate adsorption geometries.

    Parameters
    ----------
    energies : numpy.ndarray
        Energy of each candidate in eV, shape (K,) with K >= 1.
    distances : numpy.ndarray
        Reactive separation of each candidate in angstrom, same length as
        energies.
    e_window : float
        Non-negative energy window in eV above the lowest energy in the set.
    d_window : float
        Non-negative absolute cut-off in angstrom on the reactive separation.
    tol_e, tol_d : float
        Non-negative tolerances in eV and angstrom defining the band kept
        around each undominated point.
    n_fronts : int
        Number of successive fronts to peel, at least 1.

    Returns
    -------
    keep : numpy.ndarray
        Integer indices of the retained candidates, ascending.

    Raises
    ------
    ValueError
        If energies is empty, if distances has a different length, if
        e_window, d_window, tol_e or tol_d is negative, or if n_fronts is
        below 1.
    """
    return keep

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_select_pareto_geometries(energies: "np.ndarray",
                                     distances: "np.ndarray",
                                     e_window: float = 1.2,
                                     d_window: float = 4.5,
                                     tol_e: float = 0.1,
                                     tol_d: float = 0.1,
                                     n_fronts: int = 2) -> "np.ndarray":
    energies = np.asarray(energies, dtype=float).reshape(-1)
    distances = np.asarray(distances, dtype=float).reshape(-1)
    if energies.size < 1:
        raise ValueError("energies must not be empty")
    if distances.shape != energies.shape:
        raise ValueError("distances must have the same length as energies")
    if not (e_window >= 0.0 and d_window >= 0.0):
        raise ValueError("e_window and d_window must be non-negative")
    if not (tol_e >= 0.0 and tol_d >= 0.0):
        raise ValueError("tol_e and tol_d must be non-negative")
    if int(n_fronts) < 1:
        raise ValueError("n_fronts must be at least 1")

    remaining = np.flatnonzero((energies <= energies.min() + e_window)
                               & (distances <= d_window))
    selected = []
    for _ in range(int(n_fronts)):
        if remaining.size == 0:
            break
        e, d = energies[remaining], distances[remaining]
        beaten = ((e[None, :] <= e[:, None]) & (d[None, :] <= d[:, None])
                  & ((e[None, :] < e[:, None]) | (d[None, :] < d[:, None])))
        core = remaining[~np.any(beaten, axis=1)]
        near = np.any((np.abs(e[:, None] - energies[core][None, :]) <= tol_e)
                      & (np.abs(d[:, None] - distances[core][None, :]) <= tol_d),
                      axis=1)
        selected.extend(remaining[near].tolist())
        remaining = remaining[~near]
    return np.array(sorted(selected), dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    shared = """import numpy as np

def spread(k, e0, de, d0, dd):
    i = np.arange(k, dtype=float)
    e = e0 + de * np.cos(1.7 * i) * (1.0 + 0.3 * np.sin(0.9 * i))
    d = d0 + dd * np.sin(1.1 * i + 0.4) ** 2
    return e, d
"""
    return [
        # normal: a scattered set where both fronts and both cut-offs bite
        {"setup": shared + """
e, d = spread(24, -33.8, 0.9, 1.8, 3.2)
""",
         "call": "select_pareto_geometries(e, d)",
         "gold_call": "_oracle_select_pareto_geometries(e, d)"},
        # normal: tighter windows and three fronts on the same set
        {"setup": shared + """
e, d = spread(24, -33.8, 0.9, 1.8, 3.2)
""",
         "call": "select_pareto_geometries(e, d, 0.6, 3.5, 0.02, 0.05, 3)",
         "gold_call":
             "_oracle_select_pareto_geometries(e, d, 0.6, 3.5, 0.02, 0.05, 3)"},
        # boundary: zero tolerances, so only the bare undominated points survive
        {"setup": shared + """
e, d = spread(18, -10.0, 1.4, 2.0, 2.8)
""",
         "call": "select_pareto_geometries(e, d, 1.2, 4.5, 0.0, 0.0, 2)",
         "gold_call":
             "_oracle_select_pareto_geometries(e, d, 1.2, 4.5, 0.0, 0.0, 2)"},
        # boundary: repeated points, which dominate nothing and all belong to
        # the same front
        {"setup": """import numpy as np
e = np.array([-2.0, -2.0, -2.0, -1.5, -1.0, -1.0])
d = np.array([3.0, 3.0, 3.0, 2.0, 1.0, 1.0])
""",
         "call": "select_pareto_geometries(e, d, 1.2, 4.5, 0.0, 0.0, 2)",
         "gold_call":
             "_oracle_select_pareto_geometries(e, d, 1.2, 4.5, 0.0, 0.0, 2)"},
        # boundary: a distance cut-off that removes every candidate
        {"setup": shared + """
e, d = spread(12, -5.0, 0.5, 6.0, 1.0)
""",
         "call": "select_pareto_geometries(e, d, 1.2, 4.5)",
         "gold_call": "_oracle_select_pareto_geometries(e, d, 1.2, 4.5)"},
        # edge: a single candidate, which is its own front
        {"setup": """import numpy as np
e = np.array([-7.25])
d = np.array([2.5])
""",
         "call": "select_pareto_geometries(e, d)",
         "gold_call": "_oracle_select_pareto_geometries(e, d)"},
        # edge: more fronts requested than the set can supply
        {"setup": """import numpy as np
e = np.array([-3.0, -2.0, -1.0])
d = np.array([1.0, 2.0, 3.0])
""",
         "call": "select_pareto_geometries(e, d, 1.2, 4.5, 0.0, 0.0, 9)",
         "gold_call":
             "_oracle_select_pareto_geometries(e, d, 1.2, 4.5, 0.0, 0.0, 9)"},
        # invalid: mismatched objective lengths
        {"setup": """import numpy as np
e = np.zeros(4)
d = np.zeros(5)
def run_model():
    try:
        select_pareto_geometries(e, d)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_select_pareto_geometries(e, d)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
        # invalid: a request for zero fronts
        {"setup": """import numpy as np
e = np.zeros(4)
d = np.zeros(4)
def run_model():
    try:
        select_pareto_geometries(e, d, 1.2, 4.5, 0.1, 0.1, 0)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_select_pareto_geometries(e, d, 1.2, 4.5, 0.1, 0.1, 0)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
    ]
