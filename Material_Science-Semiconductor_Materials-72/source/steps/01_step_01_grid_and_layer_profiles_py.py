"""
Build the non-uniform node grid of the injection-layer/transport-layer stack and the piecewise-constant material profiles carried on it.

The device region is a hole-injection layer (HIL) stacked on a hole-transport layer (HTL). The HIL is resolved with n_hil intervals of width dz_hil and the HTL, where the interfacial carrier profile varies fastest, with n_htl finer intervals of width dz_htl. Node 0 sits at z = 0 and nodes 0 to n_hil lie in the HIL at spacing dz_hil. The interval that straddles the junction, between node n_hil and node n_hil + 1, belongs to the transport layer and therefore takes the transport-layer spacing, as does every interval after it. Nodes n_hil + 1 to n_hil + n_htl lie in the HTL, so the grid has N = n_hil + n_htl + 1 nodes and the HTL occupies n_htl * dz_htl.

Every material property is piecewise constant and takes the value of the layer its node belongs to. Nothing is smoothed across the junction, because the abruptness is the physical situation being modelled. Five per-node properties are needed downstream: the flat-band valence and conduction levels (eV), which set the band-offset contribution to the carrier driving fields; the acceptor doping density (cm^-3), the fixed space charge in Poisson's equation; the intrinsic carrier density (cm^-3), which fixes the equilibrium minority population; and the relative permittivity, which enters the electrostatics.

Lengths are in cm.

Returns
-------
np.ndarray of shape (6, N): row 0 the node positions z in cm, rows 1 to 5 the per-node [EV0, EC0, NA, ni, er] as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def grid_and_layer_profiles(n_hil: int, n_htl: int, dz_hil: float, dz_htl: float,
                            EV_hil: float, EV_htl: float, EC_hil: float, EC_htl: float,
                            NA_hil: float, NA_htl: float, ni_hil: float, ni_htl: float,
                            er_hil: float, er_htl: float) -> np.ndarray:
    '''Node positions and per-node material profiles of the HIL/HTL stack.

    Parameters
    ----------
    n_hil : int
        Number of HIL intervals; nodes 0..n_hil lie in the HIL. Must be at least 1.
    n_htl : int
        Number of HTL intervals, the junction interval included. Must be at least 1.
    dz_hil : float
        HIL grid spacing in cm, positive.
    dz_htl : float
        HTL grid spacing in cm, positive; also the width of the junction interval.
    EV_hil, EV_htl : float
        Flat-band valence levels of the HIL and HTL in eV.
    EC_hil, EC_htl : float
        Flat-band conduction levels of the HIL and HTL in eV.
    NA_hil, NA_htl : float
        Acceptor doping densities in cm^-3, positive.
    ni_hil, ni_htl : float
        Intrinsic carrier densities in cm^-3, positive.
    er_hil, er_htl : float
        Relative permittivities, positive.

    Returns
    -------
    out : np.ndarray
        Array of shape (6, N) with N = n_hil + n_htl + 1. Row 0 holds the node positions z in cm
        (z[0] = 0); rows 1 to 5 hold EV0, EC0, NA, ni and er at every node.

    Raises
    ------
    ValueError
        If n_hil or n_htl is not an integer of at least 1, if a spacing is not positive, or if any
        doping density, intrinsic density or relative permittivity is not positive.
    '''
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_grid_and_layer_profiles(n_hil: int, n_htl: int, dz_hil: float, dz_htl: float,
                                    EV_hil: float, EV_htl: float, EC_hil: float, EC_htl: float,
                                    NA_hil: float, NA_htl: float, ni_hil: float, ni_htl: float,
                                    er_hil: float, er_htl: float) -> np.ndarray:
    for m in (n_hil, n_htl):
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError("n_hil and n_htl must be integers of at least 1")
    if not (dz_hil > 0.0 and dz_htl > 0.0):
        raise ValueError("grid spacings must be positive")
    if min(NA_hil, NA_htl, ni_hil, ni_htl, er_hil, er_htl) <= 0.0:
        raise ValueError("doping, intrinsic densities and permittivities must be positive")
    n_hil = int(n_hil); n_htl = int(n_htl)
    z_hil = np.arange(n_hil + 1, dtype=float) * float(dz_hil)
    z_htl = z_hil[-1] + np.arange(1, n_htl + 1, dtype=float) * float(dz_htl)
    z = np.concatenate([z_hil, z_htl])
    in_hil = np.arange(z.size) <= n_hil
    rows = [np.where(in_hil, float(a), float(b)) for a, b in
            ((EV_hil, EV_htl), (EC_hil, EC_htl), (NA_hil, NA_htl), (ni_hil, ni_htl), (er_hil, er_htl))]
    return np.vstack([z] + rows)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
args = (20, 80, 1.0e-7, 0.25e-7, -5.17, -5.60, -3.60, -2.60, 2.81e19, 1.00e17, 1.63e6, 1.59e-6, 3.0, 9.4)""",
            "call": "grid_and_layer_profiles(*args)",
            "gold_call": "_oracle_grid_and_layer_profiles(*args)",
        },
        {
            "setup": """import numpy as np
args = (3, 5, 2.0, 0.5, -5.0, -6.0, -3.0, -2.0, 4.0, 1.5, 0.7, 0.2, 3.0, 9.0)""",
            "call": "grid_and_layer_profiles(*args)",
            "gold_call": "_oracle_grid_and_layer_profiles(*args)",
        },
        {
            "setup": """import numpy as np
args = (1, 1, 1.0, 3.0, -5.1, -5.4, -3.5, -2.7, 2.0, 5.0, 1.0, 0.1, 2.5, 4.0)""",
            "call": "grid_and_layer_profiles(*args)",
            "gold_call": "_oracle_grid_and_layer_profiles(*args)",
        },
        {
            "setup": """import numpy as np
args = (4, 4, 1.0, 1.0, -5.2, -5.2, -3.6, -3.6, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0)""",
            "call": "grid_and_layer_profiles(*args)",
            "gold_call": "_oracle_grid_and_layer_profiles(*args)",
        },
        {
            "setup": """import numpy as np
def _probe(fn):
    bad = [(0, 5, 1.0, 1.0), (5, 5, 1.0, -1.0), (2.5, 5, 1.0, 1.0)]
    hits = 0
    for n1, n2, d1, d2 in bad:
        try:
            fn(n1, n2, d1, d2, -5.0, -6.0, -3.0, -2.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0)
        except ValueError:
            hits += 1
    return float(hits)""",
            "call": "_probe(grid_and_layer_profiles)",
            "gold_call": "_probe(_oracle_grid_and_layer_profiles)",
        },
    ]
