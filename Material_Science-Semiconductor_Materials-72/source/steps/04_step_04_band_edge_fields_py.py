"""
Evaluate the valence- and conduction-band driving fields at every interval midpoint of the non-uniform grid.

Carriers in this model are driven by the slope of the band edge they occupy, not by the slope of the electrostatic potential alone. Each band edge is its flat-band level shifted by the electrostatic potential energy of an electron at that point, so it combines a material contribution that jumps at the junction with an electrostatic contribution that is continuous and comes from the Poisson solve. The driving field of a band is the slope of that band edge divided by the elementary charge, taken positive where the band edge rises with z. Inside either layer the flat-band levels are constant and the driving field reduces to the ordinary electrostatic field, the negative gradient of the potential.

Across the junction the flat-band step contributes as well, and it is concentrated entirely in the one interval between node n_hil and node n_hil + 1. A few tenths of an electronvolt dropped over a fraction of a nanometre gives a field of order 10^7 V/cm, far larger than anything the applied bias contributes, so omitting that step, or spreading it over the layer, changes the interfacial field by orders of magnitude. For this material pair the valence and conduction offsets have opposite signs, so the two bands are driven in opposite senses across the interface.

Each field is required at the midpoints between adjacent nodes, where the currents are evaluated, and is formed from the two flanking nodal values and that interval's own width, so an N-node grid yields N - 1 values per band. Levels are in eV, potential in V, positions in cm and fields in V/cm.

Returns
-------
np.ndarray of shape (2, N-1), row 0 the valence-band and row 1 the conduction-band driving field at each midpoint in V/cm as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def band_edge_fields(EV0: np.ndarray, EC0: np.ndarray, phi: np.ndarray, z: np.ndarray) -> np.ndarray:
    '''Band-edge driving fields at the interval midpoints.

    Parameters
    ----------
    EV0, EC0 : np.ndarray
        Arrays of shape (N,) of flat-band valence and conduction levels in eV.
    phi : np.ndarray
        Array of shape (N,) of electrostatic potentials in V.
    z : np.ndarray
        Array of shape (N,) of strictly increasing node positions in cm.

    Returns
    -------
    F : np.ndarray
        Array of shape (2, N-1); row 0 the valence-band field and row 1 the conduction-band field at
        each midpoint, in V/cm, positive where the corresponding band edge rises with z.

    Raises
    ------
    ValueError
        If the four arrays are not one-dimensional with a common length of at least 2, or if z is not
        strictly increasing.
    '''
    return F

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_band_edge_fields(EV0: np.ndarray, EC0: np.ndarray, phi: np.ndarray, z: np.ndarray) -> np.ndarray:
    EV0 = np.array(EV0, dtype=float); EC0 = np.array(EC0, dtype=float)
    phi = np.array(phi, dtype=float); z = np.array(z, dtype=float)
    if z.ndim != 1 or z.size < 2 or any(a.shape != z.shape for a in (EV0, EC0, phi)):
        raise ValueError("EV0, EC0, phi and z must be one-dimensional with a common length of at least 2")
    h = np.diff(z)
    if np.any(h <= 0.0):
        raise ValueError("z must be strictly increasing")
    # band edge E = E0 - q*phi (eV with phi in V); field = (1/q) dE/dz
    return np.vstack([np.diff(EV0 - phi) / h, np.diff(EC0 - phi) / h])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
z = np.concatenate([np.arange(21) * 1.0e-7, 20.0e-7 + np.arange(1, 81) * 0.25e-7])
EV0 = np.concatenate([np.full(21, -5.17), np.full(80, -5.60)])
EC0 = np.concatenate([np.full(21, -3.60), np.full(80, -2.60)])
phi = np.where(np.arange(101) <= 20, 0.2 * z / z[20], 0.2 + 0.8 * (z - z[20]) / (z[-1] - z[20]))""",
            "call": "band_edge_fields(EV0, EC0, phi, z)",
            "gold_call": "_oracle_band_edge_fields(EV0, EC0, phi, z)",
        },
        {
            "setup": """import numpy as np
z = np.linspace(0.0, 4.0e-6, 11); EV0 = np.full(11, -5.3); EC0 = np.full(11, -3.1)
phi = np.linspace(0.0, 1.0, 11)""",
            "call": "band_edge_fields(EV0, EC0, phi, z)",
            "gold_call": "_oracle_band_edge_fields(EV0, EC0, phi, z)",
        },
        {
            "setup": """import numpy as np
z = np.array([0.0, 1.0e-7, 1.25e-7]); EV0 = np.array([-5.17, -5.17, -5.60])
EC0 = np.array([-3.60, -3.60, -2.60]); phi = np.zeros(3)""",
            "call": "band_edge_fields(EV0, EC0, phi, z)",
            "gold_call": "_oracle_band_edge_fields(EV0, EC0, phi, z)",
        },
        {
            "setup": """import numpy as np
z = np.array([0.0, 2.0e-7, 2.5e-7, 3.0e-7]); EV0 = np.array([-5.0, -5.0, -5.3, -5.3])
EC0 = np.array([-3.0, -3.0, -2.2, -2.2]); phi = np.array([0.0, 0.1, 0.4, 0.3])""",
            "call": "band_edge_fields(EV0, EC0, phi, z)",
            "gold_call": "_oracle_band_edge_fields(EV0, EC0, phi, z)",
        },
        {
            "setup": """import numpy as np
def _probe(fn):
    hits = 0
    for z, e in ((np.array([0.0, 1.0, 1.0]), np.zeros(3)), (np.array([0.0, 1.0, 2.0]), np.zeros(2)),
                 (np.array([0.0]), np.zeros(1))):
        try:
            fn(e, np.zeros(z.size), np.zeros(z.size), z)
        except ValueError:
            hits += 1
    return float(hits)""",
            "call": "_probe(band_edge_fields)",
            "gold_call": "_probe(_oracle_band_edge_fields)",
        },
    ]
