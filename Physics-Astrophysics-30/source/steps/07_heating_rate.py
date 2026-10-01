"""
Implement heating_rate: the local volumetric heating rate of reflection-driven

Alfvenic turbulence.

Reflection converts a fraction of the outward flux into inward fluctuations,



and the two cascade against each other and dissipate where they meet. This



step returns that dissipation as a volumetric heating rate, given the mass



density, the rms outward amplitude, the field-aligned outflow speed, the Alfven speed and the reflection rate



per unit length (the same rate that damps the outward amplitude), all in cgs



units. The finite-flow closure is retained in the stationary solar frame,



so the outflow contribution is kept when the per-length rate is converted



to a dissipation rate.







Inputs



------



rho: (N,) mass density in g cm^-3



zp: (N,) rms outward amplitude in cm/s



u: (N,) field-aligned outflow speed in cm/s



va: (N,) Alfven speed in cm/s



eta: (N,) reflection rate per unit length in 1/cm







Returns



-------



q: (N,) float, heating rate in erg cm^-3 s^-1



Returns

-------

q : np.ndarray     (N,) heating rate in erg cm^-3 s^-1.

Returns
-------
q : np.ndarray     (N,) heating rate in erg cm^-3 s^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def heating_rate(rho: np.ndarray, zp: np.ndarray, u: np.ndarray,
                 va: np.ndarray, eta: np.ndarray) -> np.ndarray:
    '''Volumetric heating rate in erg cm^-3 s^-1 at each station.

    Parameters
    ----------
    rho : np.ndarray
        (N,) mass density in g cm^-3, all positive.
    zp : np.ndarray
        (N,) rms outward amplitude in cm/s, all non-negative.
    u : np.ndarray
        (N,) field-aligned outflow speed in cm/s, all non-negative.
    va : np.ndarray
        (N,) Alfven speed in cm/s, all positive.
    eta : np.ndarray
        (N,) reflection rate per unit length in 1/cm, all non-negative.

    Returns
    -------
    q : np.ndarray
        (N,) heating rate in erg cm^-3 s^-1.

    Raises
    ------
    ValueError
        If rho, zp, u, va and eta are not all 1-D arrays, if they do not all
        have the same length, if any entry is non-finite, if any entry of rho
        or va is not positive, or if any entry of zp, u or eta is negative.
    '''
    return q  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_heating_rate(
    rho: np.ndarray,
    zp: np.ndarray,
    u: np.ndarray,
    va: np.ndarray,
    eta: np.ndarray
) -> np.ndarray:
    r = np.asarray(rho, dtype=float)
    z = np.asarray(zp, dtype=float)
    uu = np.asarray(u, dtype=float)
    v = np.asarray(va, dtype=float)
    e = np.asarray(eta, dtype=float)

    if not (r.ndim == z.ndim == uu.ndim == v.ndim == e.ndim == 1):
        raise ValueError("all inputs must be 1-D arrays")
    if not (r.shape == z.shape == uu.shape == v.shape == e.shape):
        raise ValueError("all inputs must have the same length")
    if (not np.all(np.isfinite(r))
            or not np.all(np.isfinite(z))
            or not np.all(np.isfinite(uu))
            or not np.all(np.isfinite(v))
            or not np.all(np.isfinite(e))):
        raise ValueError("all inputs must contain only finite values")
    if np.any(r <= 0.0) or np.any(v <= 0.0):
        raise ValueError("rho and va must be positive")
    if np.any(z < 0.0) or np.any(uu < 0.0) or np.any(e < 0.0):
        raise ValueError("zp, u and eta must be non-negative")

    return 0.25 * r * z ** 2 * (uu + v) * e

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: coronal-magnitude profile ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(19)
rho = 1e-16 * (1.0 + rng.random(20))
zp = 1e6 * (1.0 + rng.random(20))
u = np.linspace(1e6, 6e7, 20)
va = 5e7 * (1.0 + 0.2 * rng.random(20))
eta = 1e-12 * rng.random(20)
""",
            "call": "heating_rate(rho, zp, u, va, eta)",
            "gold_call": "_oracle_heating_rate(rho, zp, u, va, eta)",
        },
        # --- Boundary: zero reflection rate -> zero heating everywhere ---
        {
            "setup": """import numpy as np
rho = np.full(7, 3e-20)
zp = np.full(7, 9e6)
u = np.linspace(1e6, 4e7, 7)
va = np.full(7, 5e7)
eta = np.zeros(7)
""",
            "call": "heating_rate(rho, zp, u, va, eta)",
            "gold_call": "_oracle_heating_rate(rho, zp, u, va, eta)",
        },
        # --- Normal: order-unity inputs, so the returned rate is O(1) and the
        #     comparison cannot be absorbed by an absolute tolerance ---
        {
            "setup": """import numpy as np
rho = np.array([1.0, 2.0, 0.5, 4.0])
zp = np.array([2.0, 1.0, 4.0, 0.25])
u = np.array([2.0, 7.0, 0.0, 5.0])
va = np.array([3.0, 5.0, 1.0, 8.0])
eta = np.array([4.0, 0.5, 2.0, 16.0])
""",
            "call": "heating_rate(rho, zp, u, va, eta)",
            "gold_call": "_oracle_heating_rate(rho, zp, u, va, eta)",
        },
        # --- Edge: single-station arrays ---
        {
            "setup": """import numpy as np
rho = np.array([3.05e-20])
zp = np.array([9.17e6])
u = np.array([6.06e7])
va = np.array([5.09e7])
eta = np.array([4.54e-14])
""",
            "call": "heating_rate(rho, zp, u, va, eta)",
            "gold_call": "_oracle_heating_rate(rho, zp, u, va, eta)",
        },
        # --- Invalid: negative amplitude ---
        {
            "setup": """import numpy as np
rho = np.array([1e-16])
zp = np.array([-1.0])
u = np.array([1e7])
va = np.array([5e7])
eta = np.array([1e-13])
def run_model():
    try:
        heating_rate(rho, zp, u, va, eta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_heating_rate(rho, zp, u, va, eta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-finite outflow speed ---
        {
            "setup": """import numpy as np
rho = np.array([1e-16])
zp = np.array([1e6])
u = np.array([np.nan])
va = np.array([5e7])
eta = np.array([1e-13])
def run_model():
    try:
        heating_rate(rho, zp, u, va, eta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_heating_rate(rho, zp, u, va, eta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative outflow speed ---
        {
            "setup": """import numpy as np
rho = np.array([1e-16])
zp = np.array([1e6])
u = np.array([-1.0])
va = np.array([5e7])
eta = np.array([1e-13])
def run_model():
    try:
        heating_rate(rho, zp, u, va, eta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_heating_rate(rho, zp, u, va, eta)
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
