"""
Supply the carrier densities that enter the drift currents at the interval midpoints, under either the conventional or the field-selected treatment of the junction.

The drift part of the current between two nodes needs a carrier density at the interval midpoint, while the densities themselves live on the nodes, so a rule must supply the midpoint value. Away from the junction the neighbouring densities differ little and the arithmetic mean of the two nodal values is used for both carriers, whichever scheme is chosen. The two schemes differ only at the junction midpoint, the one between node n_hil (the last injection-layer node) and node n_hil + 1 (the first transport-layer node).

The conventional scheme, "mean", keeps the arithmetic mean at the junction as well. There the two nodal hole densities differ by orders of magnitude, so the mean sits close to the majority side, and when the field drives drift out of the sparse node the inflated density removes more carriers than that node holds.

The field-selected scheme, "field", instead takes each carrier's junction value from one of the two flanking nodes outright: the node on the side the carriers are arriving from, that is, the upwind node of the drift its own band-edge field drives. Holes drift in the direction of the valence-band field; electrons carry the opposite charge and drift against the conduction-band field. A field of exactly zero is treated as positive. Because the choice depends on the sign of each carrier's own field, the selected node can differ between holes and electrons at the same midpoint.

An N-node grid yields N - 1 midpoint values per carrier. Densities are in cm^-3 and fields in V/cm.

Returns
-------
np.ndarray of shape (2, N-1), row 0 the midpoint hole densities and row 1 the midpoint electron densities in cm^-3 as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def midpoint_densities(p: np.ndarray, n: np.ndarray, FV: np.ndarray, FC: np.ndarray,
                       n_hil: int, scheme: str) -> np.ndarray:
    '''Carrier densities at the interval midpoints.

    Parameters
    ----------
    p, n : np.ndarray
        Arrays of shape (N,) of nodal hole and electron densities in cm^-3.
    FV, FC : np.ndarray
        Arrays of shape (N-1,) of valence- and conduction-band driving fields at the midpoints in V/cm.
    n_hil : int
        Index of the last injection-layer node; the junction midpoint lies between nodes n_hil and
        n_hil + 1 and is midpoint index n_hil. Requires 0 <= n_hil <= N - 2.
    scheme : str
        "mean" for the arithmetic mean at every midpoint, or "field" for the field-selected value at
        the junction midpoint and the arithmetic mean elsewhere.

    Returns
    -------
    mid : np.ndarray
        Array of shape (2, N-1); row 0 the midpoint hole densities, row 1 the midpoint electron
        densities, in cm^-3.

    Raises
    ------
    ValueError
        If p and n are not one-dimensional of a common length of at least 2, if FV or FC does not have
        length N - 1, if n_hil is outside 0..N-2, or if scheme is neither "mean" nor "field".
    '''
    return mid

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_midpoint_densities(p: np.ndarray, n: np.ndarray, FV: np.ndarray, FC: np.ndarray,
                               n_hil: int, scheme: str) -> np.ndarray:
    p = np.array(p, dtype=float); n = np.array(n, dtype=float)
    FV = np.array(FV, dtype=float); FC = np.array(FC, dtype=float)
    if p.ndim != 1 or p.size < 2 or n.shape != p.shape:
        raise ValueError("p and n must be one-dimensional with a common length of at least 2")
    if FV.shape != (p.size - 1,) or FC.shape != (p.size - 1,):
        raise ValueError("FV and FC must have length N - 1")
    if isinstance(n_hil, bool) or not isinstance(n_hil, (int, np.integer)) or not 0 <= n_hil <= p.size - 2:
        raise ValueError("n_hil must be an integer index of a midpoint")
    if scheme not in ("mean", "field"):
        raise ValueError("scheme must be 'mean' or 'field'")
    p_mid = 0.5 * (p[:-1] + p[1:])
    n_mid = 0.5 * (n[:-1] + n[1:])
    if scheme == "field":
        j = int(n_hil)
        p_mid[j] = p[j] if FV[j] >= 0.0 else p[j + 1]
        n_mid[j] = n[j + 1] if FC[j] >= 0.0 else n[j]
    return np.vstack([p_mid, n_mid])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
p = np.concatenate([np.full(21, 2.81e19), np.full(80, 1.00e17)])
n = np.concatenate([np.full(21, 9.4552e-8), np.full(80, 2.5281e-29)])
FV = np.concatenate([np.full(20, -3.1e5), [-1.52e7], np.full(79, -1.4e5)])
FC = np.concatenate([np.full(20, -3.1e5), [3.61e7], np.full(79, -1.4e5)])""",
            "call": "midpoint_densities(p, n, FV, FC, 20, 'field')",
            "gold_call": "_oracle_midpoint_densities(p, n, FV, FC, 20, 'field')",
        },
        {
            "setup": """import numpy as np
p = np.array([6.0, 5.0, 1.0, 2.0, 3.0]); n = np.array([4.0, 7.0, 2.0, 9.0, 8.0])
FV = np.array([1.0, -2.0, 3.0, -1.0]); FC = np.array([-1.0, 2.0, -3.0, 1.0])""",
            "call": "midpoint_densities(p, n, FV, FC, 1, 'field')",
            "gold_call": "_oracle_midpoint_densities(p, n, FV, FC, 1, 'field')",
        },
        {
            "setup": """import numpy as np
p = np.array([6.0, 5.0, 1.0, 2.0, 3.0]); n = np.array([4.0, 7.0, 2.0, 9.0, 8.0])
FV = np.array([1.0, 2.0, 3.0, -1.0]); FC = np.array([-1.0, -2.0, -3.0, 1.0])""",
            "call": "midpoint_densities(p, n, FV, FC, 1, 'field')",
            "gold_call": "_oracle_midpoint_densities(p, n, FV, FC, 1, 'field')",
        },
        {
            "setup": """import numpy as np
p = np.array([6.0, 5.0, 1.0]); n = np.array([4.0, 7.0, 2.0])
FV = np.array([0.0, 0.0]); FC = np.array([0.0, 0.0])""",
            "call": "midpoint_densities(p, n, FV, FC, 1, 'field')",
            "gold_call": "_oracle_midpoint_densities(p, n, FV, FC, 1, 'field')",
        },
        {
            "setup": """import numpy as np
p = np.array([6.0, 5.0, 1.0, 2.0, 3.0]); n = np.array([4.0, 7.0, 2.0, 9.0, 8.0])
FV = np.array([1.0, -2.0, 3.0, -1.0]); FC = np.array([-1.0, 2.0, -3.0, 1.0])""",
            "call": "midpoint_densities(p, n, FV, FC, 1, 'mean')",
            "gold_call": "_oracle_midpoint_densities(p, n, FV, FC, 1, 'mean')",
        },
        {
            "setup": """import numpy as np
p = np.ones(4); n = np.ones(4); FV = np.ones(3); FC = np.ones(3)
def _probe(fn):
    hits = 0
    for kw in (dict(n_hil=3, scheme='field'), dict(n_hil=1, scheme='upwind'), dict(n_hil=-1, scheme='mean')):
        try:
            fn(p, n, FV, FC, **kw)
        except ValueError:
            hits += 1
    try:
        fn(p, n, FV, np.ones(4), 1, 'mean')
    except ValueError:
        hits += 1
    return float(hits)""",
            "call": "_probe(midpoint_densities)",
            "gold_call": "_probe(_oracle_midpoint_densities)",
        },
    ]
