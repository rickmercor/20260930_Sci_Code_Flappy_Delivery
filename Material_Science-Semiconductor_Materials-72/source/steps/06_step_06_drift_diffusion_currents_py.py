"""
Evaluate the hole and electron drift-diffusion current densities on every interval of the non-uniform grid.

With midpoint densities and midpoint fields in hand, the current across each interval is a drift term plus a diffusion term. The drift term is the elementary charge times the carrier mobility times the midpoint density times that carrier's band-edge driving field. The diffusion term is the mobility times the thermal energy k_B T times the difference of the two nodal densities divided by the interval's own width; this is the Einstein relation, which ties the diffusion coefficient to the mobility through the thermal voltage, and it is the only place the temperature enters this step. The two carriers take opposite relative signs on the diffusion term: it is subtracted from the drift term for holes and added to it for electrons. With these conventions a positive current is a flow of positive charge towards larger z.

The mobility acting on an interval is that of the layer the interval belongs to. Mobilities are supplied per node; every interval belongs to the layer of its right-hand node, which is the layer of both its nodes away from the junction, and the junction interval belongs to the transport layer, so it carries the transport-layer mobilities.

Use q = 1.602176634e-19 C and k_B = 1.380649e-23 J/K. Mobilities are in cm^2 V^-1 s^-1, densities in cm^-3, fields in V/cm, positions in cm, temperature in K and current densities in A cm^-2.

Returns
-------
np.ndarray of shape (2, N-1), row 0 the hole and row 1 the electron current density on each interval in A cm^-2 as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def drift_diffusion_currents(p: np.ndarray, n: np.ndarray, p_mid: np.ndarray, n_mid: np.ndarray,
                             FV: np.ndarray, FC: np.ndarray, mu_p: np.ndarray, mu_n: np.ndarray,
                             z: np.ndarray, T: float) -> np.ndarray:
    '''Drift-diffusion current densities on the grid intervals.

    Parameters
    ----------
    p, n : np.ndarray
        Arrays of shape (N,) of nodal hole and electron densities in cm^-3.
    p_mid, n_mid : np.ndarray
        Arrays of shape (N-1,) of midpoint hole and electron densities in cm^-3.
    FV, FC : np.ndarray
        Arrays of shape (N-1,) of valence- and conduction-band driving fields in V/cm.
    mu_p, mu_n : np.ndarray
        Arrays of shape (N,) of nodal hole and electron mobilities in cm^2 V^-1 s^-1.
    z : np.ndarray
        Array of shape (N,) of strictly increasing node positions in cm.
    T : float
        Temperature in K, positive.

    Returns
    -------
    J : np.ndarray
        Array of shape (2, N-1); row 0 the hole current density, row 1 the electron current density,
        in A cm^-2.

    Raises
    ------
    ValueError
        If the nodal arrays do not share a one-dimensional shape of length at least 2, if a midpoint
        array does not have length N - 1, if z is not strictly increasing, or if T is not positive.
    '''
    return J

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_drift_diffusion_currents(p: np.ndarray, n: np.ndarray, p_mid: np.ndarray, n_mid: np.ndarray,
                                     FV: np.ndarray, FC: np.ndarray, mu_p: np.ndarray, mu_n: np.ndarray,
                                     z: np.ndarray, T: float) -> np.ndarray:
    q = 1.602176634e-19
    kB = 1.380649e-23
    p = np.array(p, dtype=float); n = np.array(n, dtype=float); z = np.array(z, dtype=float)
    mu_p = np.array(mu_p, dtype=float); mu_n = np.array(mu_n, dtype=float)
    mids = [np.array(a, dtype=float) for a in (p_mid, n_mid, FV, FC)]
    if z.ndim != 1 or z.size < 2 or any(a.shape != z.shape for a in (p, n, mu_p, mu_n)):
        raise ValueError("nodal arrays must be one-dimensional with a common length of at least 2")
    if any(a.shape != (z.size - 1,) for a in mids):
        raise ValueError("midpoint arrays must have length N - 1")
    h = np.diff(z)
    if np.any(h <= 0.0):
        raise ValueError("z must be strictly increasing")
    if not T > 0.0:
        raise ValueError("T must be positive")
    p_mid, n_mid, FV, FC = mids
    mp = mu_p[1:]
    mn = mu_n[1:]
    Jp = q * mp * p_mid * FV - mp * kB * T * np.diff(p) / h
    Jn = q * mn * n_mid * FC + mn * kB * T * np.diff(n) / h
    return np.vstack([Jp, Jn])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
z = np.concatenate([np.arange(21) * 1.0e-7, 20.0e-7 + np.arange(1, 81) * 0.25e-7])
p = np.concatenate([np.full(21, 2.81e19), np.full(80, 1.00e17)])
n = np.concatenate([np.full(21, 9.4552e-8), np.full(80, 2.5281e-29)])
p_mid = 0.5 * (p[:-1] + p[1:]); p_mid[20] = 3.3e17; n_mid = 0.5 * (n[:-1] + n[1:]); n_mid[20] = 4.0e-20
FV = np.concatenate([np.full(20, -3.1e5), [-1.52e7], np.full(79, -1.4e5)])
FC = np.concatenate([np.full(20, -3.1e5), [3.61e7], np.full(79, -1.4e5)])
mu_p = np.concatenate([np.full(21, 2.0e-4), np.full(80, 2.0e-3)])
mu_n = np.concatenate([np.full(21, 2.0e-6), np.full(80, 2.0e-5)])""",
            "call": "drift_diffusion_currents(p, n, p_mid, n_mid, FV, FC, mu_p, mu_n, z, 300.0)",
            "gold_call": "_oracle_drift_diffusion_currents(p, n, p_mid, n_mid, FV, FC, mu_p, mu_n, z, 300.0)",
        },
        {
            "setup": """import numpy as np
z = np.array([0.0, 1.0e-7, 2.0e-7, 2.25e-7, 2.5e-7])
p = np.array([3.0e15, 2.0e15, 1.5e15, 1.0e15, 0.5e15]); n = np.array([1.0e15, 2.5e15, 0.5e15, 2.0e15, 3.0e15])
p_mid = np.array([2.5e15, 1.5e15, 1.25e15, 0.75e15]); n_mid = np.array([1.75e15, 2.5e15, 1.25e15, 2.5e15])
FV = np.array([2.0e5, -1.0e7, 3.0e5, -4.0e5]); FC = np.array([-1.0e5, 4.0e7, 2.0e5, 1.0e5])
mu_p = np.array([2.0e-4, 2.0e-4, 2.0e-4, 2.0e-3, 2.0e-3]); mu_n = np.array([3.0e-4, 3.0e-4, 3.0e-4, 5.0e-3, 5.0e-3])""",
            "call": "drift_diffusion_currents(p, n, p_mid, n_mid, FV, FC, mu_p, mu_n, z, 300.0)",
            "gold_call": "_oracle_drift_diffusion_currents(p, n, p_mid, n_mid, FV, FC, mu_p, mu_n, z, 300.0)",
        },
        {
            "setup": """import numpy as np
z = np.array([0.0, 0.5e-7, 1.5e-7]); p = np.full(3, 1.0e16); n = np.full(3, 1.0e16)
p_mid = np.full(2, 1.0e16); n_mid = np.full(2, 1.0e16); FV = np.array([3.0e5, 3.0e5]); FC = np.array([-2.0e5, -2.0e5])
mu_p = np.array([1.0e-3, 1.0e-3, 4.0e-3]); mu_n = np.array([2.0e-3, 2.0e-3, 6.0e-3])""",
            "call": "drift_diffusion_currents(p, n, p_mid, n_mid, FV, FC, mu_p, mu_n, z, 300.0)",
            "gold_call": "_oracle_drift_diffusion_currents(p, n, p_mid, n_mid, FV, FC, mu_p, mu_n, z, 300.0)",
        },
        {
            "setup": """import numpy as np
z = np.array([0.0, 1.0e-7, 1.3e-7]); p = np.array([4.0e15, 1.0e15, 2.0e15]); n = np.array([1.0e15, 3.0e15, 2.0e15])
p_mid = np.array([2.5e15, 1.0e15]); n_mid = np.array([3.0e15, 2.5e15]); FV = np.array([0.0, 0.0]); FC = np.array([0.0, 0.0])
mu_p = np.array([1.0e-4, 1.0e-4, 1.0e-3]); mu_n = np.array([2.0e-4, 2.0e-4, 5.0e-4])""",
            "call": "drift_diffusion_currents(p, n, p_mid, n_mid, FV, FC, mu_p, mu_n, z, 450.0)",
            "gold_call": "_oracle_drift_diffusion_currents(p, n, p_mid, n_mid, FV, FC, mu_p, mu_n, z, 450.0)",
        },
        {
            "setup": """import numpy as np
def _probe(fn):
    hits = 0
    z = np.array([0.0, 1.0, 2.0]); a = np.ones(3); m = np.ones(2)
    for args in ((a, a, m, m, m, m, a, a, np.array([0.0, 1.0, 1.0]), 300.0),
                 (a, a, m, m, m, m, a, a, z, 0.0),
                 (a, a, np.ones(3), m, m, m, a, a, z, 300.0)):
        try:
            fn(*args)
        except ValueError:
            hits += 1
    return float(hits)""",
            "call": "_probe(drift_diffusion_currents)",
            "gold_call": "_probe(_oracle_drift_diffusion_currents)",
        },
    ]
